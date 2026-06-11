# SPDX-License-Identifier: Apache-2.0
"""Vendored Triton kernels for the TriAttention KV-eviction pipeline.

These mirror the RocketKV idiom (one ``@triton.jit`` kernel + one ``triton_*``
launch wrapper per pipeline block). Each wrapper is a drop-in replacement for the
matching block in the PyTorch ``TriAttention`` manager, and is gated behind an A/B
flag by the caller -- the PyTorch reference stays as the ``else`` branch.

Blocks Triton-ized here:
  (1) ``triton_tri_score``   -- fuse read-K-from-paged-pool + complex product +
        offset phase (mean-collapsed, or explicit max over offsets) + sum-over-F +
        MLR term  ->  per-(query-head, token) scores ``[H, seq]``.
  (2) ``triton_tri_reduce_heads`` -- mean over query heads  ->  ``[seq]``.
  (3) ``triton_tri_select`` -- recency window + top-k over the older region, returns
        sorted-ascending kept slot indices ``[keep_count]``.
  (4) ``triton_tri_compact`` -- physically compact the kept tokens to the front of
        the request's pages (moves BOTH K and V), in-place on the HND pool.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every page/stride offset that can exceed 2^31 (paged-pool reads).
  * mask seq tails not divisible by ``tokens_per_block`` (and freq/dim tails).
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Block (1): score-from-paged-K                                               #
#   read K (HND paged pool) -> GQA gather -> complex product Q.conj(K) ->      #
#   position term (offset-collapsed cos/sin for 'mean', explicit O-loop for    #
#   'max') -> sum over F -> + MLR term  =>  [H, seq] per-query-head scores.    #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_score_kernel(
    pool_ptr,            # HND KV pool passed AS-IS (strided view, no reshape):
                         #   only its base data_ptr is used; every element
                         #   address is rebuilt from the explicit s_* strides.
    page_ids_ptr,        # [num_pages] int64: this request's physical page ids
    q_real_ptr,          # [H, F] fp32: real part of mean query (E_q.real)
    q_imag_ptr,          # [H, F] fp32: imag part of mean query (E_q.imag)
    mlr_coef_ptr,        # [H, F] fp32: (E_q_norm - |E_q|)  (the MLR coefficient)
    freq_scale_sq_ptr,   # [F]    fp32: per-freq RoPE amplitude^2
    omega_ptr,           # [F]    fp32: RoPE inverse frequencies (for 'max' path)
    mean_cos_ptr,        # [F]    fp32: mean_o cos((round_start+offset)*omega)  ('mean')
    mean_sin_ptr,        # [F]    fp32: mean_o sin((round_start+offset)*omega)  ('mean')
    offsets_ptr,         # [O]    fp32: geometric look-ahead offsets   (for 'max')
    out_ptr,             # [H, seq] fp32: output scores
    round_start,         # scalar fp32: current query absolute position
    seq_len,
    num_q_heads,
    num_kv_heads,
    num_freqs,           # F = head_dim // 2
    head_dim,
    tokens_per_block,
    kv_factor,
    num_offsets,         # O (only used on the 'max' path)
    # element strides of the HND pool (page, kv_factor, kv_head, slot, dim):
    s_page,
    s_kvf,
    s_kvh,
    s_slot,
    s_dim,
    USE_MAX: tl.constexpr,
    T_BLOCK: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    # grid = (num_q_heads, cdiv(seq_len, T_BLOCK))
    q_head = tl.program_id(0)
    t_blk = tl.program_id(1)
    if q_head >= num_q_heads:
        return

    # GQA map -- PORT EXACTLY from _score_layer (NOT RocketKV's head//(H//nkv)):
    #   qhead_to_kvhead = clamp(q_head * nkv // max(1, H), max=nkv-1)
    kv_head = (q_head * num_kv_heads) // num_q_heads
    kv_head = tl.minimum(kv_head, num_kv_heads - 1)

    f = tl.arange(0, F_BLOCK)
    f_mask = f < num_freqs

    # Per-head / per-freq stats (broadcast over tokens).
    qre = tl.load(q_real_ptr + q_head * num_freqs + f, mask=f_mask, other=0.0)
    qim = tl.load(q_imag_ptr + q_head * num_freqs + f, mask=f_mask, other=0.0)
    mlrc = tl.load(mlr_coef_ptr + q_head * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)
    mcos = tl.load(mean_cos_ptr + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + f, mask=f_mask, other=0.0)

    t = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
    t_mask = t < seq_len
    t64 = t.to(tl.int64)

    # Paged address reconstruction (RocketKV idiom; real HND pool, separate
    # kv_factor axis -- KEY half is kv_factor index 0, so its stride term is 0).
    blk_in_seq = t // tokens_per_block
    slot = (t % tokens_per_block).to(tl.int64)
    phys_page = tl.load(page_ids_ptr + blk_in_seq, mask=t_mask, other=0).to(tl.int64)
    # base offset into the flat pool for (page, KEY=0, kv_head, slot).  All int64.
    base = (phys_page * s_page
            + kv_head.to(tl.int64) * s_kvh
            + slot * s_slot)  # [T_BLOCK]

    # "half" RoPE layout: real = dims [0, F), imag = dims [F, 2F).
    f64 = f.to(tl.int64)
    load_mask = t_mask[:, None] & f_mask[None, :]
    addr_re = base[:, None] + f64[None, :] * s_dim
    addr_im = base[:, None] + (num_freqs + f64[None, :]) * s_dim
    k_re = tl.load(pool_ptr + addr_re, mask=load_mask, other=0.0).to(tl.float32)
    k_im = tl.load(pool_ptr + addr_im, mask=load_mask, other=0.0).to(tl.float32)

    # Complex product Q . conj(K)  (port of _score_layer:412-413):
    prod_real = qre[None, :] * k_re + qim[None, :] * k_im       # [T_BLOCK, F]
    prod_imag = qim[None, :] * k_re - qre[None, :] * k_im

    if USE_MAX:
        # max over the O offsets does NOT commute through the freq-sum, so we
        # carry the explicit offset loop and reduce max over the per-offset
        # F-sum (matches score_per_offset.max(dim=0)).
        score = tl.full((T_BLOCK,), -float("inf"), tl.float32)
        for o in tl.range(0, num_offsets):
            off = tl.load(offsets_ptr + o)
            phase = (round_start + off) * tl.load(omega_ptr + f, mask=f_mask, other=0.0)  # [F]
            cphase = tl.cos(phase)
            sphase = tl.sin(phase)
            per_f = fss[None, :] * (prod_real * cphase[None, :] - prod_imag * sphase[None, :])
            offset_score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)  # [T_BLOCK]
            score = tl.maximum(score, offset_score)
    else:
        # 'mean' aggregation: the O loop collapses analytically. mean_o cos/sin
        # are precomputed [F] vectors, so the position term is a per-(head,token)
        # dot over F.  (mean over offsets of  fss*(prod_real*cos - prod_imag*sin)).
        per_f = fss[None, :] * (prod_real * mcos[None, :] - prod_imag * msin[None, :])
        score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)  # [T_BLOCK]

    # position-INDEPENDENT MLR term:  sum_f |K|_f * mlr_coef_f * freq_scale_sq_f.
    kmag = tl.sqrt(k_re * k_re + k_im * k_im)
    mlr_f = kmag * mlrc[None, :] * fss[None, :]
    mlr = tl.sum(tl.where(f_mask[None, :], mlr_f, 0.0), axis=1)      # [T_BLOCK]

    out = score + mlr
    tl.store(out_ptr + q_head * seq_len + t, out, mask=t_mask)


def triton_tri_score(
    pool: torch.Tensor,
    page_ids: torch.Tensor,
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    mlr_coef: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    round_start: float,
    seq_len: int,
    num_q_heads: int,
    score_aggregation: str = "mean",
) -> torch.Tensor:
    """Drop-in for ``_read_request_k`` + ``_score_layer`` + (just the score, not
    the head-mean).  Returns ``[num_q_heads, seq_len]`` fp32 scores.

    Args:
        pool: HND KV pool view
            ``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``.
            (Pass the raw ``get_buffers(layer_idx, kv_layout="HND")`` view; its
            strides are read with ``pool.stride()`` so any padding is respected.)
        page_ids: 1-D tensor of this request's physical page ids (int).
        q_real/q_imag: ``[H, F]`` fp32 real/imag of the mean query (E_q).
        mlr_coef: ``[H, F]`` fp32 ``(E_q_norm - |E_q|)``.
        freq_scale_sq: ``[F]`` fp32.
        omega: ``[F]`` fp32 RoPE inverse frequencies.
        offsets: ``[O]`` fp32 geometric look-ahead offsets.
        round_start: current query absolute position.
        seq_len: committed token count for this request.
        num_q_heads: H.
        score_aggregation: ``"mean"`` (default, offset-collapsed) or ``"max"``.
    """
    device = pool.device
    num_pages_total, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape
    num_freqs = head_dim // 2

    page_ids = page_ids.to(device=device, dtype=torch.int64)
    omega = omega.to(device=device, dtype=torch.float32)
    offsets = offsets.to(device=device, dtype=torch.float32)
    num_offsets = int(offsets.numel())

    # Offset-collapsed cos/sin for the 'mean' path. Cheap [F] vectors on host;
    # for 'max' these are unused (kernel takes the explicit O loop instead).
    qp = (float(round_start) + offsets).view(-1, 1)           # [O, 1]
    phase = qp * omega.view(1, -1)                            # [O, F]
    mean_cos = torch.cos(phase).mean(dim=0).contiguous()      # [F]
    mean_sin = torch.sin(phase).mean(dim=0).contiguous()      # [F]

    out = torch.empty((num_q_heads, seq_len), dtype=torch.float32, device=device)

    F_BLOCK = triton.next_power_of_2(num_freqs)
    T_BLOCK = 64
    grid = (num_q_heads, triton.cdiv(seq_len, T_BLOCK))

    # Pass the strided HND view directly. Triton uses only ``pool.data_ptr()``
    # as the base; every element address inside the kernel is reconstructed from
    # the explicit element strides ``s_page/s_kvf/s_kvh/s_slot/s_dim`` below, so
    # a flatten is unnecessary. Crucially, the HND view is NON-contiguous (it
    # reinterprets the native bytes under a swapped axis order), so
    # ``pool.reshape(-1)`` would clone the WHOLE layer pool on every score call.
    _tri_score_kernel[grid](
        pool,
        page_ids,
        q_real.to(device=device, dtype=torch.float32).contiguous(),
        q_imag.to(device=device, dtype=torch.float32).contiguous(),
        mlr_coef.to(device=device, dtype=torch.float32).contiguous(),
        freq_scale_sq.to(device=device, dtype=torch.float32).contiguous(),
        omega,
        mean_cos,
        mean_sin,
        offsets,
        out,
        float(round_start),
        seq_len,
        num_q_heads,
        num_kv_heads,
        num_freqs,
        head_dim,
        tokens_per_block,
        kv_factor,
        num_offsets,
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        pool.stride(4),
        USE_MAX=(score_aggregation == "max"),
        T_BLOCK=T_BLOCK,
        F_BLOCK=F_BLOCK,
    )
    return out


# --------------------------------------------------------------------------- #
# Block (2): reduce over query heads -> [seq]                                 #
#   layer_score = head_scores.mean(dim=0)  (unweighted mean over ALL H heads). #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_reduce_heads_kernel(
    head_scores_ptr,   # [H, seq] fp32
    out_ptr,           # [seq]   fp32
    num_heads,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    # grid = (cdiv(seq_len, BLOCK_SIZE),)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for h in range(num_heads):
        v = tl.load(head_scores_ptr + h * seq_len + offs, mask=mask, other=0.0)
        acc += v
    tl.store(out_ptr + offs, acc / num_heads, mask=mask)


def triton_tri_reduce_heads(head_scores: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``head_scores.mean(dim=0)``.

    Args:
        head_scores: ``[H, seq_len]`` fp32 per-query-head scores.
    Returns:
        ``[seq_len]`` fp32 mean over heads.
    """
    assert head_scores.ndim == 2
    num_heads, seq_len = head_scores.shape
    out = torch.empty((seq_len,), dtype=torch.float32, device=head_scores.device)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(seq_len, BLOCK_SIZE),)
    _tri_reduce_heads_kernel[grid](
        head_scores.contiguous(), out, num_heads, seq_len, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# --------------------------------------------------------------------------- #
# Block (3): recency-window + top-k select                                    #
#   ALWAYS keep the most-recent recency_window slots; spend the leftover       #
#   budget on a top-k over the OLDER region; return SORTED-ASCENDING indices.  #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_older_argmax_kernel(
    scores_ptr,         # [seq_len] fp32 (full vector; only [0, older_len) is read)
    chosen_ptr,         # [older_budget] int32: output selected OLDER-region indices
    older_len,          # = seq_len - recency_window
    older_budget,       # = keep_count - recency_window
    NEG_INF,            # fp32 sentinel for "already taken"
    BLOCK_SIZE: tl.constexpr,
):
    # grid = (1,).  Selection-style top-k: pick the argmax `older_budget` times,
    # masking the running buffer in-place.  This is O(older_budget * older_len).
    #
    # TIE-BREAK: this kernel uses a DETERMINISTIC lowest-index tie-break -- on a
    # value tie it keeps the smallest slot index (`tl.min` within a block;
    # `blk_max > best_val` strict-greater keeps the first block across blocks).
    # ``torch.topk`` does NOT guarantee lowest-index ties, so to keep the kept
    # SET bit-identical between the two A/B paths the PyTorch reference
    # (``_select_with_recency`` / the test reference) must apply the SAME
    # lowest-index tie-break (see ``triton_tri_select`` below and the test). With
    # both sides made deterministic, the kept set matches exactly even under
    # exact-value ties; the production path may swap in the RocketKV histogram
    # top-k instead (which does not carry this guarantee).
    for sel in tl.range(0, older_budget):
        best_val = NEG_INF
        best_idx = 0
        for blk in tl.range(0, older_len, BLOCK_SIZE):
            offs = blk + tl.arange(0, BLOCK_SIZE)
            mask = offs < older_len
            vals = tl.load(scores_ptr + offs, mask=mask, other=NEG_INF)
            # block-local argmax:
            blk_max = tl.max(vals, axis=0)
            # lowest index achieving the block max (stable tie-break = first idx):
            is_max = vals == blk_max
            idx_candidates = tl.where(is_max & mask, offs, older_len)
            blk_idx = tl.min(idx_candidates, axis=0)
            if blk_max > best_val:
                best_val = blk_max
                best_idx = blk_idx
        tl.store(chosen_ptr + sel, best_idx)
        # mask out the chosen slot so it is not picked again:
        tl.store(scores_ptr + best_idx, NEG_INF)


def triton_tri_select(scores: torch.Tensor, seq_len: int,
                      top_B: int, window_size: int) -> torch.Tensor:
    """Drop-in for ``_select_with_recency``.

    Returns sorted-ascending int64 kept slot indices in ``[0, seq_len)``,
    length ``min(top_B, seq_len)``.  The two short-circuit branches (keep-all,
    pure-recency) are handled in Python exactly as the reference; only the
    genuine older-region top-k uses the kernel.

    TIE-BREAK NOTE: the older-region top-k kernel resolves exact-value ties by
    keeping the LOWEST slot index. ``torch.topk`` (used by the PyTorch A/B
    reference) does not guarantee this, so the kept SET is only provably
    bit-identical when scores in the older region are distinct, OR when the
    reference applies the same lowest-index tie-break. The GPU equivalence test
    feeds distinct scores AND uses a tie-stabilized reference, so it holds
    exactly; in production the two paths may pick different indices only on the
    rare exact-value tie at the budget boundary (same kept COUNT either way).
    """
    device = scores.device
    keep_count = min(top_B, seq_len)

    # Budget covers everything -> keep all.
    if keep_count >= seq_len:
        return torch.arange(seq_len, device=device, dtype=torch.long)

    recency_window = min(window_size, seq_len)

    # Budget <= recency window -> just the most-recent keep_count slots.
    if keep_count <= recency_window:
        return torch.arange(seq_len - keep_count, seq_len, device=device, dtype=torch.long)

    recent_indices = torch.arange(
        seq_len - recency_window, seq_len, device=device, dtype=torch.long
    )
    older_budget = keep_count - recency_window
    older_len = seq_len - recency_window

    # Selection top-k over the OLDER region. We copy the older scores into a
    # scratch buffer because the kernel destructively masks chosen slots.
    older_scratch = scores[:older_len].to(torch.float32).clone()
    chosen = torch.empty((older_budget,), dtype=torch.int32, device=device)
    NEG_INF = float("-inf")
    BLOCK_SIZE = 1024
    _tri_older_argmax_kernel[(1,)](
        older_scratch, chosen, older_len, older_budget,
        NEG_INF, BLOCK_SIZE=BLOCK_SIZE,
    )
    older_keep = chosen.to(torch.long)

    return torch.sort(torch.cat([older_keep, recent_indices])).values


# --------------------------------------------------------------------------- #
# Block (4): gather / compact kept tokens                                     #
#   apply the full permutation [kept..., dropped...] to the (page, slot) token #
#   axis, for BOTH K and V, writing back the touched pages.  Two-pass          #
#   (gather -> scratch -> scatter) to match the reference .clone() semantics   #
#   and avoid the in-place read/write aliasing hazard.                         #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_compact_kernel(
    pool_ptr,            # HND KV pool passed AS-IS (read AND write, in place):
                         #   strided view, only its base data_ptr is used; every
                         #   element address is rebuilt from the s_* strides.
    new_order_ptr,       # [seq_len] int64: source token for each dest slot
    page_ids_ptr,        # [num_pages] int64: this request's physical page ids
    scratch_ptr,         # [kv_factor, num_kv_heads, seq_len, head_dim] fp-native scratch
    seq_len,
    num_kv_heads,
    tokens_per_block,
    head_dim,
    kv_factor,
    s_page,
    s_kvf,
    s_kvh,
    s_slot,
    s_dim,
    PHASE: tl.constexpr,   # 0 = gather pool->scratch, 1 = scatter scratch->pool
    D_BLOCK: tl.constexpr,
):
    # grid = (seq_len, kv_factor, num_kv_heads). One program moves one head_dim
    # vector for one (dest token, kv_factor half, kv head).
    dest = tl.program_id(0)
    kvf = tl.program_id(1)
    kvh = tl.program_id(2)
    if dest >= seq_len:
        return

    d = tl.arange(0, D_BLOCK)
    d_mask = d < head_dim
    d64 = d.to(tl.int64)

    # scratch is packed [kv_factor, num_kv_heads, seq_len, head_dim]:
    scratch_base = (((kvf.to(tl.int64) * num_kv_heads + kvh.to(tl.int64)) * seq_len
                     + dest.to(tl.int64)) * head_dim)

    if PHASE == 0:
        # PASS 1: read SOURCE token = new_order[dest] from the live pool, write
        # into scratch at dest. This realizes index_select on the token axis.
        src = tl.load(new_order_ptr + dest)
        src_blk = src // tokens_per_block
        src_slot = (src % tokens_per_block).to(tl.int64)
        src_page = tl.load(page_ids_ptr + src_blk).to(tl.int64)
        src_base = (src_page * s_page
                    + kvf.to(tl.int64) * s_kvf
                    + kvh.to(tl.int64) * s_kvh
                    + src_slot * s_slot)
        val = tl.load(pool_ptr + src_base + d64 * s_dim, mask=d_mask, other=0.0)
        tl.store(scratch_ptr + scratch_base + d64, val, mask=d_mask)
    else:
        # PASS 2: read scratch[dest], write to the DEST token's physical slot in
        # the live pool.  dest token lives at page_ids[dest // tpb], slot dest%tpb.
        dst_blk = dest // tokens_per_block
        dst_slot = (dest % tokens_per_block).to(tl.int64)
        dst_page = tl.load(page_ids_ptr + dst_blk).to(tl.int64)
        dst_base = (dst_page * s_page
                    + kvf.to(tl.int64) * s_kvf
                    + kvh.to(tl.int64) * s_kvh
                    + dst_slot * s_slot)
        val = tl.load(scratch_ptr + scratch_base + d64, mask=d_mask, other=0.0)
        tl.store(pool_ptr + dst_base + d64 * s_dim, val, mask=d_mask)


def triton_tri_compact(pool: torch.Tensor, page_ids: torch.Tensor,
                       keep: torch.Tensor, seq_len: int) -> None:
    """Drop-in for ``_evict_layer``'s physical compaction (in place on ``pool``).

    Args:
        pool: HND KV pool view
            ``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``.
        page_ids: this request's physical page ids (int).
        keep: SORTED-ascending kept slot indices in ``[0, seq_len)``.
        seq_len: committed token count.
    """
    device = pool.device
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape

    keep = keep.to(device=device, dtype=torch.long)
    keep_count = int(keep.numel())
    if keep_count >= seq_len:
        return  # nothing to drop

    page_ids_t = page_ids.to(device=device, dtype=torch.int64)

    # Full permutation: kept first (in given order), then dropped -- so no slot
    # in [0, seq_len) is left stale (matches the reference exactly).
    all_ids = torch.arange(seq_len, device=device, dtype=torch.long)
    is_dropped = torch.ones(seq_len, device=device, dtype=torch.bool)
    is_dropped[keep] = False
    new_order = torch.cat([keep, all_ids[is_dropped]]).to(torch.int64)

    scratch = torch.empty(
        (kv_factor, num_kv_heads, seq_len, head_dim),
        dtype=pool.dtype, device=device,
    )

    D_BLOCK = triton.next_power_of_2(head_dim)
    grid = (seq_len, kv_factor, num_kv_heads)

    # CRITICAL: pass the strided HND view ``pool`` DIRECTLY (do not flatten).
    # The HND view is non-contiguous (reinterpreted bytes under a swapped axis
    # order), so ``pool.reshape(-1)`` falls back to a contiguous CLONE. Pass-2's
    # in-place scatter would then write into that throwaway copy and the live
    # pool would never be compacted -- eviction silently lost. Triton uses only
    # ``pool.data_ptr()`` as the base and the kernel rebuilds every address from
    # the explicit element strides, so the strided view is the correct target.
    # ``scratch`` IS contiguous (freshly allocated), so flattening it is safe.

    # Pass 1: gather pool[new_order] -> scratch (packed dest order).
    _tri_compact_kernel[grid](
        pool, new_order, page_ids_t, scratch.reshape(-1),
        seq_len, num_kv_heads, tokens_per_block, head_dim, kv_factor,
        pool.stride(0), pool.stride(1), pool.stride(2), pool.stride(3), pool.stride(4),
        PHASE=0, D_BLOCK=D_BLOCK,
    )
    # Pass 2: scatter scratch -> pool at dest positions (front-compacted).
    _tri_compact_kernel[grid](
        pool, new_order, page_ids_t, scratch.reshape(-1),
        seq_len, num_kv_heads, tokens_per_block, head_dim, kv_factor,
        pool.stride(0), pool.stride(1), pool.stride(2), pool.stride(3), pool.stride(4),
        PHASE=1, D_BLOCK=D_BLOCK,
    )
