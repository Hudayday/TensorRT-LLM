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
        sorted-ascending kept slot indices ``[keep_count]``. The older-region
        top-k is the fast two-stage histogram top-k (vendored verbatim from the
        RocketKV kernels, with its bitonic-argsort + histogram helpers), run as a
        single-request / single-head call. NOTE: it bins scores in the fp16 value
        domain, so the kept SET only matches an fp32 ``torch.topk`` reference when
        the scores are distinct in fp16 (the kept COUNT is always exact).
  (4) ``triton_tri_compact`` -- physically compact the kept tokens to the front of
        the request's pages (moves BOTH K and V), in-place on the HND pool.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every page/stride offset that can exceed 2^31 (paged-pool reads).
  * mask seq tails not divisible by ``tokens_per_block`` (and freq/dim tails).
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

import math
import os

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like
# Histogram top-k reused from the RocketKV kernels in the sparse-attention
# package (kernel.py): layout-agnostic pure top-k, V2-safe -- imported, not
# re-vendored. (Moved out of the sparse package, so this is now an absolute
# import rather than a sibling relative import.)
from tensorrt_llm._torch.attention_backend.sparse.kernel import triton_topk

# A/B toggle for the per-head score kernel re-parallelization ("Rank 1: kv_head
# grid axis"). When "1", the per-head score wrapper launches the kvgrid kernel
# variant that promotes kv_head from an inner while-loop to a 3rd grid axis (more
# blocks -> better B200 occupancy); the math / store layout are bit-identical
# (see _tri_score_batched_perhead_kvgrid_kernel). Default ("0"/unset) keeps the
# original kv_head-loop kernel. Read at CALL time in the wrapper (not just here at
# import) so a single process can flip it for equivalence testing.
_TRI_KVGRID = os.environ.get("TRI_KVGRID") == "1"


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
    s_kv_factor,
    s_kv_head,
    s_slot,
    s_dim,
    USE_MAX: tl.constexpr,
    T_BLOCK: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    # grid = (num_q_heads, cdiv(seq_len, T_BLOCK))
    q_head = tl.program_id(0)
    t_blk = tl.program_id(1)

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
            + kv_head.to(tl.int64) * s_kv_head
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
    # the explicit element strides ``s_page/s_kv_factor/s_kv_head/s_slot/s_dim`` below, so
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
# Block (3a): histogram top-k -- imported from .kernel (RocketKV's, V2-safe),  #
# see `from .kernel import triton_topk` at the top; not re-vendored here.       #
# --------------------------------------------------------------------------- #

def triton_tri_select(scores: torch.Tensor, seq_len: int,
                      top_B: int, window_size: int) -> torch.Tensor:
    """Drop-in for ``_select_with_recency``.

    Returns sorted-ascending int64 kept slot indices in ``[0, seq_len)``,
    length ``min(top_B, seq_len)``.  The two short-circuit branches (keep-all,
    pure-recency) are handled in Python exactly as the reference; only the
    genuine older-region top-k uses the vendored histogram kernel.

    TIE-BREAK / PRECISION NOTE: the histogram top-k bins scores in the fp16
    value domain (``convert_to_sortable_uint16`` casts to fp16 before binning),
    so the comparison key is the fp16 round of each score, not the raw fp32
    value. Two consequences for the kept SET:
      * Distinct *fp32* scores that round to the SAME fp16 value become ties
        and may be split arbitrarily at the budget boundary (e.g. the test's
        ``randperm + 0.5`` integers above ~2048 are no longer fp16-distinct).
      * Within a tied final bin the kernel resolves ties by a bitonic argsort
        on (fp16 value, original index) -- NOT by the lowest original index, so
        the tie-break differs from both ``torch.topk`` and the old
        selection-argmax kernel.
    The kept COUNT is always exact (= ``older_budget``). The kept SET matches a
    reference iff that reference selects in the SAME fp16 domain. The GPU
    equivalence test therefore quantizes its reference through fp16 before
    ``torch.topk`` (see ``ref_select_with_recency``); on scores that are
    distinct in fp16 the two SETs are identical.
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

    # Histogram top-k over the OLDER region, set up as a SINGLE request / SINGLE
    # head: input is scores[:older_len] reshaped to [num_kv_heads=1, older_len];
    # one batch entry spanning [0, older_len) -> [0, older_budget) outputs. The
    # kernel reads scores by element offset, so contiguous fp32 is required.
    older_scores = scores[:older_len].to(torch.float32).contiguous().view(1, older_len)
    input_offsets = torch.tensor([0, older_len], dtype=torch.int32, device=device)
    output_offsets = torch.tensor([0, older_budget], dtype=torch.int32, device=device)

    chosen = triton_topk(
        older_scores, input_offsets, output_offsets,
        total_output_tokens=older_budget, topk=older_budget,
    )                                       # [1, older_budget] int32
    older_keep = chosen[0].to(torch.long)   # indices into [0, older_len)

    return torch.sort(torch.cat([older_keep, recent_indices])).values


def triton_tri_select_batched(scores_list, top_B, window_size):
    """Batched ``triton_tri_select`` over many (request x layer) segments in ONE
    ``triton_topk`` launch. Each segment is a genuine eviction
    (``seq_len > top_B``, layer-uniform), so ``older_budget = top_B -
    window_size`` is uniform; only the older-region length is ragged (fed via
    ``input_offsets``). Uses the imported (kernel.py) histogram top-k.

    ``scores_list``: list of 1-D fp32 score tensors ([seq_len_i]) -- returns one
    sorted-ascending int64 kept-index tensor per segment, each identical to
    ``triton_tri_select(scores_i, seq_len_i, top_B, window_size)``. Falls back to
    the per-segment scalar path for any non-genuine segment.
    """
    if not scores_list:
        return []
    device = scores_list[0].device
    older_budget = top_B - window_size
    if older_budget <= 0 or any(s.numel() <= top_B for s in scores_list):
        return [triton_tri_select(s, s.numel(), top_B, window_size)
                for s in scores_list]

    older_lens = [s.numel() - window_size for s in scores_list]
    older_flat = torch.cat(
        [s[:ol].to(torch.float32) for s, ol in zip(scores_list, older_lens)]
    ).contiguous().view(1, -1)
    n_seg = len(scores_list)
    input_offsets = torch.zeros(n_seg + 1, dtype=torch.int32, device=device)
    input_offsets[1:] = torch.tensor(
        older_lens, dtype=torch.int32, device=device).cumsum(0)
    output_offsets = torch.arange(
        n_seg + 1, dtype=torch.int32, device=device) * older_budget

    chosen = triton_topk(
        older_flat, input_offsets, output_offsets,
        total_output_tokens=n_seg * older_budget, topk=older_budget,
    )[0].to(torch.long)   # segment-local indices into each [0, older_len_i)

    out = []
    for i, s in enumerate(scores_list):
        sl = s.numel()
        older_keep = chosen[i * older_budget:(i + 1) * older_budget]
        recent = torch.arange(sl - window_size, sl, device=device,
                              dtype=torch.long)
        out.append(torch.sort(torch.cat([older_keep, recent])).values)
    return out


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
    s_kv_factor,
    s_kv_head,
    s_slot,
    s_dim,
    PHASE: tl.constexpr,   # 0 = gather pool->scratch, 1 = scatter scratch->pool
    D_BLOCK: tl.constexpr,
):
    # grid = (seq_len, kv_factor, num_kv_heads). One program moves one head_dim
    # vector for one (dest token, kv_factor half, kv head).
    dest = tl.program_id(0)
    kv_half = tl.program_id(1)
    kv_head = tl.program_id(2)

    d = tl.arange(0, D_BLOCK)
    d_mask = d < head_dim
    d64 = d.to(tl.int64)

    # scratch is packed [kv_factor, num_kv_heads, seq_len, head_dim]:
    scratch_base = (((kv_half.to(tl.int64) * num_kv_heads + kv_head.to(tl.int64)) * seq_len
                     + dest.to(tl.int64)) * head_dim)

    if PHASE == 0:
        # PASS 1: read SOURCE token = new_order[dest] from the live pool, write
        # into scratch at dest. This realizes index_select on the token axis.
        src = tl.load(new_order_ptr + dest)
        src_blk = src // tokens_per_block
        src_slot = (src % tokens_per_block).to(tl.int64)
        src_page = tl.load(page_ids_ptr + src_blk).to(tl.int64)
        src_base = (src_page * s_page
                    + kv_half.to(tl.int64) * s_kv_factor
                    + kv_head.to(tl.int64) * s_kv_head
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
                    + kv_half.to(tl.int64) * s_kv_factor
                    + kv_head.to(tl.int64) * s_kv_head
                    + dst_slot * s_slot)
        val = tl.load(scratch_ptr + scratch_base + d64, mask=d_mask, other=0.0)
        tl.store(pool_ptr + dst_base + d64 * s_dim, val, mask=d_mask)


# R3 perf: reuse a per-(dtype,device) compaction scratch buffer instead of
# allocating [kv_factor,nkv,seq_len,hd] on every call (L per eviction). Grows to
# the max size seen; sliced [:need] (contiguous) so scratch.reshape(-1) never clones.
_COMPACT_SCRATCH = {}


def _get_compact_scratch(need: int, dtype, device) -> torch.Tensor:
    key = (dtype, str(device))
    buf = _COMPACT_SCRATCH.get(key)
    if buf is None or buf.numel() < need:
        buf = torch.empty(need, dtype=dtype, device=device)
        _COMPACT_SCRATCH[key] = buf
    return buf


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

    _need = kv_factor * num_kv_heads * seq_len * head_dim
    scratch = _get_compact_scratch(_need, pool.dtype, device)[:_need].view(
        kv_factor, num_kv_heads, seq_len, head_dim)

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


@triton.jit
def _tri_compact_batched_kernel(
    pool_ptr,            # one layer's HND pool (strided view; in place)
    new_order_ptr,       # [total] int64: per-dest-token SOURCE local idx (within its request)
    dest_local_ptr,      # [total] int64: per-dest-token LOCAL dest idx (within its request)
    page_base_ptr,       # [total] int64: this token's request's base offset into cat_page_ids
    page_ids_ptr,        # [sum_pages] int64: concatenated per-request physical page ids
    scratch_ptr,         # [kv_factor, num_kv_heads, total, head_dim] scratch
    total,
    num_kv_heads,
    tokens_per_block,
    head_dim,
    s_page, s_kv_factor, s_kv_head, s_slot, s_dim,
    PHASE: tl.constexpr,
    D_BLOCK: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    # grid = (cdiv(total, BLOCK_TOKENS), kv_factor, num_kv_heads). Each program
    # moves BLOCK_TOKENS dest tokens' head_dim vectors for one (kv_factor half,
    # kv head). BLOCK_TOKENS=1 reproduces the original one-token-per-program grid
    # byte-for-byte; >1 tiles to amortize index/launch overhead + raise occupancy.
    # Tail tokens (g >= total) are clamped to index 0 and masked out (no-op).
    g_base = tl.program_id(0) * BLOCK_TOKENS
    kv_half = tl.program_id(1)
    kv_head = tl.program_id(2)

    d = tl.arange(0, D_BLOCK)
    d_mask = d < head_dim
    d64 = d.to(tl.int64)

    for tt in tl.static_range(BLOCK_TOKENS):
        g = g_base + tt
        g_ok = g < total
        gs = tl.where(g_ok, g, 0)
        m = d_mask & g_ok
        pbase = tl.load(page_base_ptr + gs)
        scratch_base = (((kv_half.to(tl.int64) * num_kv_heads + kv_head.to(tl.int64))
                         * total + gs.to(tl.int64)) * head_dim)
        if PHASE == 0:
            # gather: SOURCE local token (new_order[g]) of this token's request.
            src = tl.load(new_order_ptr + gs)
            src_blk = src // tokens_per_block
            src_slot = (src % tokens_per_block).to(tl.int64)
            src_page = tl.load(page_ids_ptr + pbase + src_blk).to(tl.int64)
            src_base = (src_page * s_page + kv_half.to(tl.int64) * s_kv_factor
                        + kv_head.to(tl.int64) * s_kv_head + src_slot * s_slot)
            val = tl.load(pool_ptr + src_base + d64 * s_dim, mask=m, other=0.0)
            tl.store(scratch_ptr + scratch_base + d64, val, mask=m)
        else:
            # scatter: write to the LOCAL dest slot of this token's request.
            dst = tl.load(dest_local_ptr + gs)
            dst_blk = dst // tokens_per_block
            dst_slot = (dst % tokens_per_block).to(tl.int64)
            dst_page = tl.load(page_ids_ptr + pbase + dst_blk).to(tl.int64)
            dst_base = (dst_page * s_page + kv_half.to(tl.int64) * s_kv_factor
                        + kv_head.to(tl.int64) * s_kv_head + dst_slot * s_slot)
            val = tl.load(scratch_ptr + scratch_base + d64, mask=m, other=0.0)
            tl.store(pool_ptr + dst_base + d64 * s_dim, val, mask=m)


def triton_tri_compact_batched(pool, page_ids_list, keep_list, seq_len_list):
    """Batched in-place compaction over many requests sharing ONE layer ``pool``.
    For each request, kept tokens are gathered to the front (slots
    [0, keep_count_r)), matching per-request ``triton_tri_compact`` exactly.

    pool:          one layer's HND view [num_pages, kv_factor, nkv, tpb, hd].
    page_ids_list: list of 1-D int page-id tensors, one per request.
    keep_list:     list of SORTED-ascending kept local-index tensors, one per request.
    seq_len_list:  list of committed token counts, one per request.
    Requests with keep_count >= seq_len are skipped (nothing to drop).
    """
    device = pool.device
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape

    flat_new_order = []
    flat_dest_local = []
    flat_page_base = []
    cat_page_ids = []
    page_cursor = 0
    for page_ids, keep, seq_len in zip(page_ids_list, keep_list, seq_len_list):
        keep = keep.to(device=device, dtype=torch.long)
        if int(keep.numel()) >= seq_len:
            continue  # nothing to drop for this request
        all_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        is_dropped = torch.ones(seq_len, device=device, dtype=torch.bool)
        is_dropped[keep] = False
        new_order = torch.cat([keep, all_ids[is_dropped]]).to(torch.int64)  # [seq_len]
        flat_new_order.append(new_order)
        flat_dest_local.append(all_ids.to(torch.int64))                     # 0..seq_len-1
        pids = page_ids.to(device=device, dtype=torch.int64)
        flat_page_base.append(torch.full((seq_len,), page_cursor,
                                         device=device, dtype=torch.int64))
        cat_page_ids.append(pids)
        page_cursor += int(pids.numel())

    if not flat_new_order:
        return
    new_order_t = torch.cat(flat_new_order)
    dest_local_t = torch.cat(flat_dest_local)
    page_base_t = torch.cat(flat_page_base)
    page_ids_t = torch.cat(cat_page_ids)
    total = int(new_order_t.numel())

    _need = kv_factor * num_kv_heads * total * head_dim
    scratch = _get_compact_scratch(_need, pool.dtype, device)[:_need].view(
        kv_factor, num_kv_heads, total, head_dim)
    D_BLOCK = triton.next_power_of_2(head_dim)
    # BLOCK_TOKENS tiles dest tokens per program to amortize the per-token index
    # overhead (the compact is overhead-bound: scattered 128-elem moves, ~40x its
    # memory-traffic floor). Baked default = 16: bench-measured ~14% faster than the
    # one-token-per-program grid (BT=1) and BYTE-IDENTICAL to the per-request
    # reference at BT in {8,16,32,64} (test_compact_tune.py / test_compact_equiv.py).
    BLOCK_TOKENS = int(os.environ.get("TRI_COMPACT_BT", "16") or "16")
    grid = (triton.cdiv(total, BLOCK_TOKENS), kv_factor, num_kv_heads)
    for phase in (0, 1):
        _tri_compact_batched_kernel[grid](
            pool, new_order_t, dest_local_t, page_base_t, page_ids_t,
            scratch.reshape(-1),
            total, num_kv_heads, tokens_per_block, head_dim,
            pool.stride(0), pool.stride(1), pool.stride(2),
            pool.stride(3), pool.stride(4),
            PHASE=phase, D_BLOCK=D_BLOCK, BLOCK_TOKENS=BLOCK_TOKENS,
        )


# --------------------------------------------------------------------------- #
# Batched fused trig-score over (request x layer) segments -- one launch,      #
# head-mean fused in; outputs flat scores + segment offsets (feeds the batched  #
# select). GPU-verified allclose vs per-request triton_tri_score+reduce.        #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_score_batched_kernel(
    pool_storage_ptr,     # ONE typed base pointer; data_ptr() == TRUE storage base.
                          #   element-typed view aliasing all scored layers'
                          #   shared storage (no copy).
    layer_base_ptrs,      # [num_layers] int64: ELEMENT offset of each layer's
                          #   HND view base relative to the TRUE storage base.
    page_ids_ptr,         # [sum_pages] int64: all requests' page ids concat.
    page_ids_offsets,     # [num_requests+1] int64: cum #pages per request.
    # per-SEGMENT metadata (seg = req_slot*L_scored + layer_slot), idx by pid(0):
    seg_req_id,           # [nseg] int32: request slot (into page_ids_offsets etc.)
    seg_layer_id,         # [nseg] int32: ABSOLUTE layer id (indexes layer_base_ptrs + calib)
    seg_seq_len,          # [nseg] int32
    seg_round_start,      # [nseg] fp32
    seg_out_offset,       # [nseg] int64: write base into flat out
    seg_n_tblk,           # [nseg] int32: cdiv(seq_len, T_BLOCK)
    # per-LAYER calibration, [L,H,F] flattened layer-major:
    q_real_ptr,           # [L*H*F] fp32
    q_imag_ptr,           # [L*H*F] fp32
    mlr_coef_ptr,         # [L*H*F] fp32
    # per-SEGMENT offset-collapsed phase ('mean' path), [nseg,F] flattened:
    mean_cos_ptr,         # [nseg*F] fp32
    mean_sin_ptr,         # [nseg*F] fp32
    # shared freq vectors:
    freq_scale_sq_ptr,    # [F] fp32
    omega_ptr,            # [F] fp32 ('max' path only)
    offsets_ptr,          # [O] fp32 ('max' path only)
    out_ptr,              # [sum_seq] fp32 (logically [1, sum_seq])
    # scalars uniform across the batch:
    num_q_heads,
    num_kv_heads,
    num_freqs,            # F = head_dim // 2
    head_dim,
    tokens_per_block,
    kv_factor,
    num_offsets,          # O ('max' path only)
    # per-layer HND element strides (uniform across scored segments):
    s_page,
    s_kv_factor,
    s_kv_head,
    s_slot,
    s_dim,
    USE_MAX: tl.constexpr,
    T_BLOCK: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    t_blk = tl.program_id(1)

    # tiles beyond this segment's own block count do nothing (ragged grid).
    n_tblk = tl.load(seg_n_tblk + seg)
    if t_blk >= n_tblk:
        return

    seq_len = tl.load(seg_seq_len + seg)
    layer_id = tl.load(seg_layer_id + seg)
    req_id = tl.load(seg_req_id + seg)
    rstart = tl.load(seg_round_start + seg)
    out_base = tl.load(seg_out_offset + seg)
    layer_elt = tl.load(layer_base_ptrs + layer_id).to(tl.int64)
    page_off = tl.load(page_ids_offsets + req_id)

    f = tl.arange(0, F_BLOCK)
    f_mask = f < num_freqs
    f64 = f.to(tl.int64)

    # ---- token tile of THIS segment ----
    t = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
    t_mask = t < seq_len
    blk_in_seq = t // tokens_per_block
    slot = (t % tokens_per_block).to(tl.int64)
    # page_off + blk_in_seq stays within this request's page slice for in-range
    # tokens (t_mask). For masked tail tokens the load is masked (other=0) so the
    # possibly-out-of-this-request index is never dereferenced.
    phys_page = tl.load(page_ids_ptr + page_off + blk_in_seq,
                        mask=t_mask, other=0).to(tl.int64)

    # element offset into pool_storage for (layer, page, KEY=0, *, slot).
    # KEY half is kv_factor index 0 -> its stride term is 0 (matches reference).
    tok_base = layer_elt + phys_page * s_page + slot * s_slot     # [T_BLOCK] int64

    # per-segment 'mean'-path phase + shared freq scale.
    mcos = tl.load(mean_cos_ptr + seg * num_freqs + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + seg * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)

    # ---- head-sum of (position + mlr) over ALL q_heads, GQA-deduped ----
    # The original loop reloaded paged K for every q-head; with GQA each KV head
    # is shared by ``group_size = num_q_heads // num_kv_heads`` q-heads, so K was
    # fetched group_size times redundantly. Here we load K (and |K|) ONCE per KV
    # head and inner-loop only the q-heads sharing it. Iterating kv_head outer /
    # q-in-group inner reproduces h = kv_head*group_size + qg -- the SAME h order
    # as the original h=0..num_q_heads-1 -- so the fp32 head-accumulation is
    # bit-identical. Requires num_q_heads % num_kv_heads == 0 (asserted host-side
    # in triton_tri_score_batched; standard for MHA/GQA/MQA).
    acc = tl.zeros((T_BLOCK,), dtype=tl.float32)
    group_size = num_q_heads // num_kv_heads
    load_mask = t_mask[:, None] & f_mask[None, :]
    off_re = f64[None, :] * s_dim
    off_im = (num_freqs + f64[None, :]) * s_dim

    kv_head = 0
    while kv_head < num_kv_heads:
        base = tok_base + kv_head.to(tl.int64) * s_kv_head         # [T_BLOCK]
        # paged K loaded ONCE for this KV head (shared by group_size q-heads).
        k_re = tl.load(pool_storage_ptr + base[:, None] + off_re,
                       mask=load_mask, other=0.0).to(tl.float32)
        k_im = tl.load(pool_storage_ptr + base[:, None] + off_im,
                       mask=load_mask, other=0.0).to(tl.float32)
        kmag = tl.sqrt(k_re * k_re + k_im * k_im)                  # once per KV head

        qg = 0
        while qg < group_size:
            h = kv_head * group_size + qg
            calib_off = (layer_id.to(tl.int64) * num_q_heads + h) * num_freqs
            qre = tl.load(q_real_ptr + calib_off + f, mask=f_mask, other=0.0)
            qim = tl.load(q_imag_ptr + calib_off + f, mask=f_mask, other=0.0)
            mlrc = tl.load(mlr_coef_ptr + calib_off + f, mask=f_mask, other=0.0)

            # complex product Q . conj(K)  (port of _score_layer).
            prod_real = qre[None, :] * k_re + qim[None, :] * k_im  # [T_BLOCK, F]
            prod_imag = qim[None, :] * k_re - qre[None, :] * k_im

            if USE_MAX:
                # max over O offsets does NOT commute through the freq-sum;
                # explicit O loop reducing max over the per-offset F-sum.
                score = tl.full((T_BLOCK,), -float("inf"), tl.float32)
                o = 0
                while o < num_offsets:
                    off = tl.load(offsets_ptr + o)
                    om = tl.load(omega_ptr + f, mask=f_mask, other=0.0)
                    phase = (rstart + off) * om
                    cphase = tl.cos(phase)
                    sphase = tl.sin(phase)
                    per_f = fss[None, :] * (prod_real * cphase[None, :]
                                            - prod_imag * sphase[None, :])
                    offset_score = tl.sum(
                        tl.where(f_mask[None, :], per_f, 0.0), axis=1)
                    score = tl.maximum(score, offset_score)
                    o += 1
            else:
                # 'mean': offset loop collapsed into mean_cos/mean_sin.
                per_f = fss[None, :] * (prod_real * mcos[None, :]
                                        - prod_imag * msin[None, :])
                score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)

            # position-INDEPENDENT MLR term (reuses the per-KV-head |K|).
            mlr_f = kmag * mlrc[None, :] * fss[None, :]
            mlr = tl.sum(tl.where(f_mask[None, :], mlr_f, 0.0), axis=1)

            acc += score + mlr
            qg += 1
        kv_head += 1

    out = acc / num_q_heads
    tl.store(out_ptr + out_base + t, out, mask=t_mask)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _flat_from_true_storage_base(anchor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return (flat, storage_base_ptr) where ``flat`` is a 1-D element-typed
    tensor whose ``data_ptr()`` is the TRUE underlying storage base, spanning the
    whole storage WITHOUT copying. Gives the kernel a single typed base pointer;
    per-layer element offsets index into it.

    Anchors on the TRUE storage base (NOT on the min layer data_ptr), so it is
    correct even when the scored layers are a SUBSET of a larger allocation whose
    lowest scored layer has storage_offset > 0 (VSWA grouping). Built via
    ``set_(storage, storage_offset=0, ...)`` so ``flat.data_ptr()`` IS the true
    base regardless of which layer we anchored on.
    """
    st = anchor.untyped_storage()
    elt_size = anchor.element_size()
    nelem = st.nbytes() // elt_size
    flat = torch.empty(0, dtype=anchor.dtype, device=anchor.device)
    flat.set_(st, storage_offset=0, size=(nelem,), stride=(1,))
    storage_base = int(flat.data_ptr())
    # Cross-check the offset arithmetic: anchor's data_ptr must equal
    # storage_base + storage_offset*elt_size.
    assert int(anchor.data_ptr()) == storage_base + anchor.storage_offset() * elt_size, (
        "unexpected storage layout: anchor data_ptr != base + offset*elt_size")
    return flat, storage_base


class SegMeta(NamedTuple):
    seg_index: int       # position in the request-major (req_slot*L + layer_slot) order
    request_index: int   # slot into page_ids_list / seq_lens / round_starts
    layer_index: int     # ABSOLUTE layer id (from layer_indices)
    seq_len: int         # seq_len_r
    round_start: float   # per-request round_start


# --------------------------------------------------------------------------- #
# Python wrapper                                                               #
# --------------------------------------------------------------------------- #
def triton_tri_score_batched(
    layer_pools,            # list[L_all] of HND views get_buffers(l,'HND').
    page_ids_list,          # list[K] of 1-D int tensors: each request's page ids.
    seq_lens,               # list[K] int: per-request committed seq_len (>top_B).
    round_starts,           # list[K] int/float: per-request round_start.
    q_real_LHF,             # [L_all,H,F] fp32 (stack of per-layer E_q.real).
    q_imag_LHF,             # [L_all,H,F] fp32.
    mlr_coef_LHF,           # [L_all,H,F] fp32 (E_q_norm - |E_q|).
    freq_scale_sq,          # [F] fp32.
    omega,                  # [F] fp32.
    offsets,                # [O] fp32.
    num_q_heads,            # H.
    score_aggregation="mean",
    layer_indices=None,     # optional list of ABSOLUTE layer ids to score (default all).
) -> Tuple[torch.Tensor, torch.Tensor, List[SegMeta]]:
    """Returns (flat_scores[1,sum_seg_seq], seg_offsets[nseg+1] int32, seg_meta).

    seg_meta is a list of SegMeta (seg_index, request_index, layer_index,
    seq_len, round_start) in segment order (REQUEST-MAJOR, then layer). The slice
    flat_scores[0, seg_offsets[s]:seg_offsets[s+1]] equals
    triton_tri_reduce_heads(triton_tri_score(pool=layer_pools[layer_index],
    page_ids=page_ids_list[request_index], ... same calib/round_start ...)).

    To feed the existing list-based ``triton_tri_select_batched``, use
    ``flat_scores_to_list(flat_scores, seg_offsets)``.
    """
    assert score_aggregation in ("mean", "max")
    device = layer_pools[0].device
    L_all = len(layer_pools)
    layer_ids = list(range(L_all)) if layer_indices is None else list(layer_indices)
    K = len(page_ids_list)
    assert len(seq_lens) == K and len(round_starts) == K

    # ---- HND geometry (uniform across scored layers; assert it) ----
    p0 = layer_pools[layer_ids[0]]
    _num_pages0, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
    num_freqs = head_dim // 2
    # GQA-deduped score kernel loads K once per KV head and inner-loops the
    # group_size = H // nkv q-heads sharing it; needs exact divisibility so the
    # h order (and thus the fp32 head-accumulation) is preserved bit-for-bit.
    assert num_q_heads % num_kv_heads == 0, (
        f"score kernel requires num_q_heads ({num_q_heads}) % num_kv_heads "
        f"({num_kv_heads}) == 0")
    s_page, s_kv_factor, s_kv_head, s_slot, s_dim = (
        p0.stride(0), p0.stride(1), p0.stride(2), p0.stride(3), p0.stride(4))
    for lid in layer_ids:
        pl = layer_pools[lid]
        assert tuple(pl.shape[1:]) == tuple(p0.shape[1:]), (
            "uniform (kv_factor,nkv,tpb,hd) required across scored layers; "
            "group by config otherwise.")
        assert pl.stride() == p0.stride(), "non-uniform HND strides across layers"
        assert pl.dtype == p0.dtype, "non-uniform pool dtype across layers"

    # ---- single typed base ptr (TRUE storage base) + per-layer element offsets ----
    elt_size = p0.element_size()
    anchor = layer_pools[layer_ids[0]]
    pool_storage, storage_base = _flat_from_true_storage_base(anchor)

    layer_base_ptrs = torch.zeros(L_all, dtype=torch.int64, device=device)
    for lid in layer_ids:
        pl = layer_pools[lid]
        # All scored layers must alias the SAME underlying storage.
        assert pl.untyped_storage().data_ptr() == anchor.untyped_storage().data_ptr(), (
            "scored layer views do not share one storage (VSWA multi-pool); group "
            "segments by pool group and call triton_tri_score_batched per group.")
        off_bytes = int(pl.data_ptr()) - storage_base
        assert off_bytes >= 0 and off_bytes % elt_size == 0, (
            "layer base not within/aligned to chosen storage")
        layer_base_ptrs[lid] = off_bytes // elt_size

    # ---- page ids: concat per request, request-indexed offsets ----
    page_ids_offsets = torch.zeros(K + 1, dtype=torch.int64, device=device)
    pid_parts = []
    for r in range(K):
        pid = torch.as_tensor(page_ids_list[r], device=device, dtype=torch.int64).reshape(-1)
        pid_parts.append(pid)
        page_ids_offsets[r + 1] = page_ids_offsets[r] + pid.numel()
    page_ids_flat = (torch.cat(pid_parts).contiguous() if pid_parts
                     else torch.zeros(0, dtype=torch.int64, device=device))

    # ---- segments: request-major then layer ----
    L = len(layer_ids)
    nseg = K * L
    seg_req = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_layer = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_seq = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_rstart = torch.empty(nseg, dtype=torch.float32, device=device)
    seg_out_off = torch.empty(nseg, dtype=torch.int64, device=device)
    seg_ntblk = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_offsets = torch.zeros(nseg + 1, dtype=torch.int32, device=device)
    seg_meta: List[SegMeta] = []

    # T_BLOCK is the TOKEN tiling. Each token's score is a single per-tile tl.sum
    # over F (F_BLOCK covers all freqs) with no cross-tile/cross-token fp32 fold, so
    # the stored value is INVARIANT to T_BLOCK -> tuning it is byte-identical (only
    # launch/index overhead changes). num_warps/num_stages are scheduling-only
    # (numerics-neutral). All three are env-tunable at call time for the bench/sweep;
    # the winning constants get baked in once validated byte-identical.
    T_BLOCK = int(os.environ.get("TRI_TBLOCK", "64") or "64")
    F_BLOCK = triton.next_power_of_2(num_freqs)

    mean_cos = torch.empty(nseg, num_freqs, dtype=torch.float32, device=device)
    mean_sin = torch.empty(nseg, num_freqs, dtype=torch.float32, device=device)
    omega_d = omega.to(device=device, dtype=torch.float32)
    offsets_d = offsets.to(device=device, dtype=torch.float32)

    run = 0
    max_ntblk = 1
    s = 0
    for r in range(K):
        sl = int(seq_lens[r])
        rs = float(round_starts[r])
        # offset-collapsed cos/sin: same for all layers of this request, built
        # EXACTLY as triton_tri_score does (mean over offsets of cos/sin(phase)).
        qp = (rs + offsets_d).view(-1, 1)            # [O,1]
        phase = qp * omega_d.view(1, -1)             # [O,F]
        mc = torch.cos(phase).mean(dim=0)            # [F]
        ms = torch.sin(phase).mean(dim=0)            # [F]
        ntblk = (sl + T_BLOCK - 1) // T_BLOCK
        for layer_slot, lid in enumerate(layer_ids):
            seg_req[s] = r
            seg_layer[s] = lid
            seg_seq[s] = sl
            seg_rstart[s] = rs
            seg_out_off[s] = run
            seg_ntblk[s] = ntblk
            mean_cos[s] = mc
            mean_sin[s] = ms
            seg_offsets[s + 1] = run + sl
            seg_meta.append(SegMeta(
                seg_index=s, request_index=r, layer_index=lid,
                seq_len=sl, round_start=rs))
            run += sl
            max_ntblk = max(max_ntblk, ntblk)
            s += 1

    sum_seq = run
    out = torch.empty(sum_seq, dtype=torch.float32, device=device)

    # calib flattened layer-major [L_all*H*F].
    q_real_f = q_real_LHF.to(device=device, dtype=torch.float32).contiguous().view(-1)
    q_imag_f = q_imag_LHF.to(device=device, dtype=torch.float32).contiguous().view(-1)
    mlr_f = mlr_coef_LHF.to(device=device, dtype=torch.float32).contiguous().view(-1)
    fss_d = freq_scale_sq.to(device=device, dtype=torch.float32).contiguous()

    # grid axis 0 = nseg (on the 2^31-1 axis); axis 1 = max token-blocks per
    # segment (small: cdiv(seq_len, T_BLOCK), <=1024 even at 64k tokens). The
    # unbounded token count NEVER lands on a 65535-limited grid axis.
    grid = (nseg, max_ntblk)
    _tri_score_batched_kernel[grid](
        pool_storage,
        layer_base_ptrs,
        page_ids_flat,
        page_ids_offsets,
        seg_req, seg_layer, seg_seq, seg_rstart, seg_out_off, seg_ntblk,
        q_real_f, q_imag_f, mlr_f,
        mean_cos.view(-1).contiguous(), mean_sin.view(-1).contiguous(),
        fss_d, omega_d, offsets_d,
        out,
        num_q_heads, num_kv_heads, num_freqs, head_dim, tokens_per_block,
        kv_factor, int(offsets_d.numel()),
        s_page, s_kv_factor, s_kv_head, s_slot, s_dim,
        USE_MAX=(score_aggregation == "max"),
        T_BLOCK=T_BLOCK, F_BLOCK=F_BLOCK,
    )
    return out.view(1, -1), seg_offsets, seg_meta


def flat_scores_to_list(flat_scores: torch.Tensor,
                        seg_offsets: torch.Tensor) -> List[torch.Tensor]:
    """Zero-copy bridge: slice flat_scores[1,sum_seq] into the per-segment list
    of 1-D [seq_i] views the existing ``triton_tri_select_batched(scores_list,
    ...)`` consumer expects. Each slice is a VIEW (no copy); the only host work is
    ``nseg`` narrow calls, unavoidable given the list-based consumer API.
    """
    flat = flat_scores.view(-1)
    off = seg_offsets.tolist()
    return [flat[off[s]:off[s + 1]] for s in range(len(off) - 1)]


# --------------------------------------------------------------------------- #
# Batched fused trig-score over (request x layer) segments -- PER-HEAD variant. #
# Identical to _tri_score_batched_kernel EXCEPT the per-query-head scores are    #
# written out WITHOUT the head-mean reduction: out is [H, sum_seq] and each      #
# head h writes (score + mlr) at row h. Each per-(layer,request) head slice is    #
# bit-identical to triton_tri_score(...)[h]. Feeds the per_head / per_layer_perhead#
# / union eviction modes that need un-reduced per-query-head scores.             #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_score_batched_perhead_kernel(
    pool_storage_ptr,     # ONE typed base pointer; data_ptr() == TRUE storage base.
                          #   element-typed view aliasing all scored layers'
                          #   shared storage (no copy).
    layer_base_ptrs,      # [num_layers] int64: ELEMENT offset of each layer's
                          #   HND view base relative to the TRUE storage base.
    page_ids_ptr,         # [sum_pages] int64: all requests' page ids concat.
    page_ids_offsets,     # [num_requests+1] int64: cum #pages per request.
    # per-SEGMENT metadata (seg = req_slot*L_scored + layer_slot), idx by pid(0):
    seg_req_id,           # [nseg] int32: request slot (into page_ids_offsets etc.)
    seg_layer_id,         # [nseg] int32: ABSOLUTE layer id (indexes layer_base_ptrs + calib)
    seg_seq_len,          # [nseg] int32
    seg_round_start,      # [nseg] fp32
    seg_out_offset,       # [nseg] int64: write base (column) into each [H, sum_seq] row
    seg_n_tblk,           # [nseg] int32: cdiv(seq_len, T_BLOCK)
    # per-LAYER calibration, [L,H,F] flattened layer-major:
    q_real_ptr,           # [L*H*F] fp32
    q_imag_ptr,           # [L*H*F] fp32
    mlr_coef_ptr,         # [L*H*F] fp32
    # per-SEGMENT offset-collapsed phase ('mean' path), [nseg,F] flattened:
    mean_cos_ptr,         # [nseg*F] fp32
    mean_sin_ptr,         # [nseg*F] fp32
    # shared freq vectors:
    freq_scale_sq_ptr,    # [F] fp32
    omega_ptr,            # [F] fp32 ('max' path only)
    offsets_ptr,          # [O] fp32 ('max' path only)
    out_ptr,              # [num_q_heads * sum_seq] fp32 (logically [H, sum_seq])
    sum_seq,              # int: total tokens across all segments == row stride of out.
    # scalars uniform across the batch:
    num_q_heads,
    num_kv_heads,
    num_freqs,            # F = head_dim // 2
    head_dim,
    tokens_per_block,
    kv_factor,
    num_offsets,          # O ('max' path only)
    # per-layer HND element strides (uniform across scored segments):
    s_page,
    s_kv_factor,
    s_kv_head,
    s_slot,
    s_dim,
    USE_MAX: tl.constexpr,
    T_BLOCK: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    t_blk = tl.program_id(1)

    # tiles beyond this segment's own block count do nothing (ragged grid).
    n_tblk = tl.load(seg_n_tblk + seg)
    if t_blk >= n_tblk:
        return

    seq_len = tl.load(seg_seq_len + seg)
    layer_id = tl.load(seg_layer_id + seg)
    req_id = tl.load(seg_req_id + seg)
    rstart = tl.load(seg_round_start + seg)
    out_base = tl.load(seg_out_offset + seg)
    layer_elt = tl.load(layer_base_ptrs + layer_id).to(tl.int64)
    page_off = tl.load(page_ids_offsets + req_id)

    f = tl.arange(0, F_BLOCK)
    f_mask = f < num_freqs
    f64 = f.to(tl.int64)

    # ---- token tile of THIS segment ----
    t = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
    t_mask = t < seq_len
    blk_in_seq = t // tokens_per_block
    slot = (t % tokens_per_block).to(tl.int64)
    # page_off + blk_in_seq stays within this request's page slice for in-range
    # tokens (t_mask). For masked tail tokens the load is masked (other=0) so the
    # possibly-out-of-this-request index is never dereferenced.
    phys_page = tl.load(page_ids_ptr + page_off + blk_in_seq,
                        mask=t_mask, other=0).to(tl.int64)

    # element offset into pool_storage for (layer, page, KEY=0, *, slot).
    # KEY half is kv_factor index 0 -> its stride term is 0 (matches reference).
    tok_base = layer_elt + phys_page * s_page + slot * s_slot     # [T_BLOCK] int64

    # per-segment 'mean'-path phase + shared freq scale.
    mcos = tl.load(mean_cos_ptr + seg * num_freqs + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + seg * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)

    # ---- PER-HEAD (position + mlr), GQA-deduped, NO head reduction ----
    # Identical loop nest / load schedule / fp32 math to _tri_score_batched_kernel,
    # but instead of accumulating into a single head-mean we WRITE each head's own
    # (score + mlr) to its row in the [H, sum_seq] output. Iterating kv_head outer /
    # q-in-group inner reproduces h = kv_head*group_size + qg -- the SAME h order as
    # the per-request _tri_score_kernel's q_head = 0..num_q_heads-1 -- so the value
    # stored at row h is bit-identical to triton_tri_score(...)[h]. K (and |K|) is
    # loaded ONCE per KV head, shared by the group_size = H // nkv q-heads.
    group_size = num_q_heads // num_kv_heads
    load_mask = t_mask[:, None] & f_mask[None, :]
    off_re = f64[None, :] * s_dim
    off_im = (num_freqs + f64[None, :]) * s_dim

    kv_head = 0
    while kv_head < num_kv_heads:
        base = tok_base + kv_head.to(tl.int64) * s_kv_head         # [T_BLOCK]
        # paged K loaded ONCE for this KV head (shared by group_size q-heads).
        k_re = tl.load(pool_storage_ptr + base[:, None] + off_re,
                       mask=load_mask, other=0.0).to(tl.float32)
        k_im = tl.load(pool_storage_ptr + base[:, None] + off_im,
                       mask=load_mask, other=0.0).to(tl.float32)
        kmag = tl.sqrt(k_re * k_re + k_im * k_im)                  # once per KV head

        qg = 0
        while qg < group_size:
            h = kv_head * group_size + qg
            calib_off = (layer_id.to(tl.int64) * num_q_heads + h) * num_freqs
            qre = tl.load(q_real_ptr + calib_off + f, mask=f_mask, other=0.0)
            qim = tl.load(q_imag_ptr + calib_off + f, mask=f_mask, other=0.0)
            mlrc = tl.load(mlr_coef_ptr + calib_off + f, mask=f_mask, other=0.0)

            # complex product Q . conj(K)  (port of _score_layer).
            prod_real = qre[None, :] * k_re + qim[None, :] * k_im  # [T_BLOCK, F]
            prod_imag = qim[None, :] * k_re - qre[None, :] * k_im

            if USE_MAX:
                # max over O offsets does NOT commute through the freq-sum;
                # explicit O loop reducing max over the per-offset F-sum.
                score = tl.full((T_BLOCK,), -float("inf"), tl.float32)
                o = 0
                while o < num_offsets:
                    off = tl.load(offsets_ptr + o)
                    om = tl.load(omega_ptr + f, mask=f_mask, other=0.0)
                    phase = (rstart + off) * om
                    cphase = tl.cos(phase)
                    sphase = tl.sin(phase)
                    per_f = fss[None, :] * (prod_real * cphase[None, :]
                                            - prod_imag * sphase[None, :])
                    offset_score = tl.sum(
                        tl.where(f_mask[None, :], per_f, 0.0), axis=1)
                    score = tl.maximum(score, offset_score)
                    o += 1
            else:
                # 'mean': offset loop collapsed into mean_cos/mean_sin.
                per_f = fss[None, :] * (prod_real * mcos[None, :]
                                        - prod_imag * msin[None, :])
                score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)

            # position-INDEPENDENT MLR term (reuses the per-KV-head |K|).
            mlr_f = kmag * mlrc[None, :] * fss[None, :]
            mlr = tl.sum(tl.where(f_mask[None, :], mlr_f, 0.0), axis=1)

            # write this head's score at row h, column (out_base + t). No mean.
            tl.store(out_ptr + h.to(tl.int64) * sum_seq + out_base + t,
                     score + mlr, mask=t_mask)
            qg += 1
        kv_head += 1


# --------------------------------------------------------------------------- #
# kv_head-GRID variant of _tri_score_batched_perhead_kernel ("Rank 1").          #
# IDENTICAL to _tri_score_batched_perhead_kernel EXCEPT kv_head is read from a    #
# 3rd grid axis (tl.program_id(2)) instead of an inner while-loop -- one block    #
# per (seg, t_blk, kv_head) => more blocks => better B200 occupancy. The qg loop  #
# stays inner so the group_size q-heads still share ONE paged-K load. Each block   #
# computes the identical per-(kv_head,qg,t) value and writes the SAME disjoint     #
# output row h = kv_head*group_size + qg; the F-sum is a single per-tile tl.sum    #
# with no cross-head/cross-kv_head/cross-tile fp32 fold, so the output is byte-     #
# identical whether kv_head is a loop or a grid axis. Launched with               #
# grid=(nseg, max_ntblk, num_kv_heads). Same arg list as the loop kernel.          #
# --------------------------------------------------------------------------- #
@triton.jit
def _tri_score_batched_perhead_kvgrid_kernel(
    pool_storage_ptr,     # ONE typed base pointer; data_ptr() == TRUE storage base.
                          #   element-typed view aliasing all scored layers'
                          #   shared storage (no copy).
    layer_base_ptrs,      # [num_layers] int64: ELEMENT offset of each layer's
                          #   HND view base relative to the TRUE storage base.
    page_ids_ptr,         # [sum_pages] int64: all requests' page ids concat.
    page_ids_offsets,     # [num_requests+1] int64: cum #pages per request.
    # per-SEGMENT metadata (seg = req_slot*L_scored + layer_slot), idx by pid(0):
    seg_req_id,           # [nseg] int32: request slot (into page_ids_offsets etc.)
    seg_layer_id,         # [nseg] int32: ABSOLUTE layer id (indexes layer_base_ptrs + calib)
    seg_seq_len,          # [nseg] int32
    seg_round_start,      # [nseg] fp32
    seg_out_offset,       # [nseg] int64: write base (column) into each [H, sum_seq] row
    seg_n_tblk,           # [nseg] int32: cdiv(seq_len, T_BLOCK)
    # per-LAYER calibration, [L,H,F] flattened layer-major:
    q_real_ptr,           # [L*H*F] fp32
    q_imag_ptr,           # [L*H*F] fp32
    mlr_coef_ptr,         # [L*H*F] fp32
    # per-SEGMENT offset-collapsed phase ('mean' path), [nseg,F] flattened:
    mean_cos_ptr,         # [nseg*F] fp32
    mean_sin_ptr,         # [nseg*F] fp32
    # shared freq vectors:
    freq_scale_sq_ptr,    # [F] fp32
    omega_ptr,            # [F] fp32 ('max' path only)
    offsets_ptr,          # [O] fp32 ('max' path only)
    out_ptr,              # [num_q_heads * sum_seq] fp32 (logically [H, sum_seq])
    sum_seq,              # int: total tokens across all segments == row stride of out.
    # scalars uniform across the batch:
    num_q_heads,
    num_kv_heads,
    num_freqs,            # F = head_dim // 2
    head_dim,
    tokens_per_block,
    kv_factor,
    num_offsets,          # O ('max' path only)
    # per-layer HND element strides (uniform across scored segments):
    s_page,
    s_kv_factor,
    s_kv_head,
    s_slot,
    s_dim,
    USE_MAX: tl.constexpr,
    T_BLOCK: tl.constexpr,
    F_BLOCK: tl.constexpr,
):
    seg = tl.program_id(0)
    t_blk = tl.program_id(1)

    # tiles beyond this segment's own block count do nothing (ragged grid).
    n_tblk = tl.load(seg_n_tblk + seg)
    if t_blk >= n_tblk:
        return

    seq_len = tl.load(seg_seq_len + seg)
    layer_id = tl.load(seg_layer_id + seg)
    req_id = tl.load(seg_req_id + seg)
    rstart = tl.load(seg_round_start + seg)
    out_base = tl.load(seg_out_offset + seg)
    layer_elt = tl.load(layer_base_ptrs + layer_id).to(tl.int64)
    page_off = tl.load(page_ids_offsets + req_id)

    f = tl.arange(0, F_BLOCK)
    f_mask = f < num_freqs
    f64 = f.to(tl.int64)

    # ---- token tile of THIS segment ----
    t = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
    t_mask = t < seq_len
    blk_in_seq = t // tokens_per_block
    slot = (t % tokens_per_block).to(tl.int64)
    # page_off + blk_in_seq stays within this request's page slice for in-range
    # tokens (t_mask). For masked tail tokens the load is masked (other=0) so the
    # possibly-out-of-this-request index is never dereferenced.
    phys_page = tl.load(page_ids_ptr + page_off + blk_in_seq,
                        mask=t_mask, other=0).to(tl.int64)

    # element offset into pool_storage for (layer, page, KEY=0, *, slot).
    # KEY half is kv_factor index 0 -> its stride term is 0 (matches reference).
    tok_base = layer_elt + phys_page * s_page + slot * s_slot     # [T_BLOCK] int64

    # per-segment 'mean'-path phase + shared freq scale.
    mcos = tl.load(mean_cos_ptr + seg * num_freqs + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + seg * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)

    # ---- PER-HEAD (position + mlr), GQA-deduped, NO head reduction ----
    # Identical loop nest / load schedule / fp32 math to _tri_score_batched_kernel,
    # but instead of accumulating into a single head-mean we WRITE each head's own
    # (score + mlr) to its row in the [H, sum_seq] output. kv_head comes from the
    # 3rd grid axis here (one block per kv_head); the q-in-group inner loop yields
    # h = kv_head*group_size + qg -- the SAME h order as the per-request
    # _tri_score_kernel's q_head = 0..num_q_heads-1 -- so the value stored at row h
    # is bit-identical to triton_tri_score(...)[h]. K (and |K|) is loaded ONCE per
    # KV head, shared by the group_size = H // nkv q-heads.
    group_size = num_q_heads // num_kv_heads
    load_mask = t_mask[:, None] & f_mask[None, :]
    off_re = f64[None, :] * s_dim
    off_im = (num_freqs + f64[None, :]) * s_dim

    kv_head = tl.program_id(2)
    base = tok_base + kv_head.to(tl.int64) * s_kv_head         # [T_BLOCK]
    # paged K loaded ONCE for this KV head (shared by group_size q-heads).
    k_re = tl.load(pool_storage_ptr + base[:, None] + off_re,
                   mask=load_mask, other=0.0).to(tl.float32)
    k_im = tl.load(pool_storage_ptr + base[:, None] + off_im,
                   mask=load_mask, other=0.0).to(tl.float32)
    kmag = tl.sqrt(k_re * k_re + k_im * k_im)                  # once per KV head

    qg = 0
    while qg < group_size:
        h = kv_head * group_size + qg
        calib_off = (layer_id.to(tl.int64) * num_q_heads + h) * num_freqs
        qre = tl.load(q_real_ptr + calib_off + f, mask=f_mask, other=0.0)
        qim = tl.load(q_imag_ptr + calib_off + f, mask=f_mask, other=0.0)
        mlrc = tl.load(mlr_coef_ptr + calib_off + f, mask=f_mask, other=0.0)

        # complex product Q . conj(K)  (port of _score_layer).
        prod_real = qre[None, :] * k_re + qim[None, :] * k_im  # [T_BLOCK, F]
        prod_imag = qim[None, :] * k_re - qre[None, :] * k_im

        if USE_MAX:
            # max over O offsets does NOT commute through the freq-sum;
            # explicit O loop reducing max over the per-offset F-sum.
            score = tl.full((T_BLOCK,), -float("inf"), tl.float32)
            o = 0
            while o < num_offsets:
                off = tl.load(offsets_ptr + o)
                om = tl.load(omega_ptr + f, mask=f_mask, other=0.0)
                phase = (rstart + off) * om
                cphase = tl.cos(phase)
                sphase = tl.sin(phase)
                per_f = fss[None, :] * (prod_real * cphase[None, :]
                                        - prod_imag * sphase[None, :])
                offset_score = tl.sum(
                    tl.where(f_mask[None, :], per_f, 0.0), axis=1)
                score = tl.maximum(score, offset_score)
                o += 1
        else:
            # 'mean': offset loop collapsed into mean_cos/mean_sin.
            per_f = fss[None, :] * (prod_real * mcos[None, :]
                                    - prod_imag * msin[None, :])
            score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)

        # position-INDEPENDENT MLR term (reuses the per-KV-head |K|).
        mlr_f = kmag * mlrc[None, :] * fss[None, :]
        mlr = tl.sum(tl.where(f_mask[None, :], mlr_f, 0.0), axis=1)

        # write this head's score at row h, column (out_base + t). No mean.
        tl.store(out_ptr + h.to(tl.int64) * sum_seq + out_base + t,
                 score + mlr, mask=t_mask)
        qg += 1


def triton_tri_score_batched_perhead(
    layer_pools,            # list[L_all] of HND views get_buffers(l,'HND').
    page_ids_list,          # list[K] of 1-D int tensors: each request's page ids.
    seq_lens,               # list[K] int: per-request committed seq_len (>top_B).
    round_starts,           # list[K] int/float: per-request round_start.
    q_real_LHF,             # [L_all,H,F] fp32 (stack of per-layer E_q.real).
    q_imag_LHF,             # [L_all,H,F] fp32.
    mlr_coef_LHF,           # [L_all,H,F] fp32 (E_q_norm - |E_q|).
    freq_scale_sq,          # [F] fp32.
    omega,                  # [F] fp32.
    offsets,                # [O] fp32.
    num_q_heads,            # H.
    score_aggregation="mean",
    layer_indices=None,     # optional list of ABSOLUTE layer ids to score (default all).
) -> Tuple[torch.Tensor, torch.Tensor, List[SegMeta]]:
    """PER-HEAD sibling of ``triton_tri_score_batched``: same args / same host
    setup, but returns un-reduced per-query-head scores.

    Returns (perhead_scores[H, sum_seq], seg_offsets[nseg+1] int32, seg_meta).

    seg_meta is a list of SegMeta (seg_index, request_index, layer_index,
    seq_len, round_start) in segment order (REQUEST-MAJOR, then layer). The slice
    perhead_scores[:, seg_offsets[s]:seg_offsets[s+1]] equals
    triton_tri_score(pool=layer_pools[layer_index],
    page_ids=page_ids_list[request_index], ... same calib/round_start ...) (i.e.
    [H, seq_len], NO head-mean reduction).

    To split into the per-segment list of [H, seq_i] views, use
    ``flat_perhead_to_list(perhead_scores, seg_offsets)``.
    """
    assert score_aggregation in ("mean", "max")
    device = layer_pools[0].device
    L_all = len(layer_pools)
    layer_ids = list(range(L_all)) if layer_indices is None else list(layer_indices)
    K = len(page_ids_list)
    assert len(seq_lens) == K and len(round_starts) == K

    # ---- HND geometry (uniform across scored layers; assert it) ----
    p0 = layer_pools[layer_ids[0]]
    _num_pages0, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
    num_freqs = head_dim // 2
    # GQA-deduped score kernel loads K once per KV head and inner-loops the
    # group_size = H // nkv q-heads sharing it; needs exact divisibility so the
    # h order (and thus the per-head math) is preserved bit-for-bit.
    assert num_q_heads % num_kv_heads == 0, (
        f"score kernel requires num_q_heads ({num_q_heads}) % num_kv_heads "
        f"({num_kv_heads}) == 0")
    s_page, s_kv_factor, s_kv_head, s_slot, s_dim = (
        p0.stride(0), p0.stride(1), p0.stride(2), p0.stride(3), p0.stride(4))
    for lid in layer_ids:
        pl = layer_pools[lid]
        assert tuple(pl.shape[1:]) == tuple(p0.shape[1:]), (
            "uniform (kv_factor,nkv,tpb,hd) required across scored layers; "
            "group by config otherwise.")
        assert pl.stride() == p0.stride(), "non-uniform HND strides across layers"
        assert pl.dtype == p0.dtype, "non-uniform pool dtype across layers"

    # ---- single typed base ptr (TRUE storage base) + per-layer element offsets ----
    elt_size = p0.element_size()
    anchor = layer_pools[layer_ids[0]]
    pool_storage, storage_base = _flat_from_true_storage_base(anchor)

    layer_base_ptrs = torch.zeros(L_all, dtype=torch.int64, device=device)
    for lid in layer_ids:
        pl = layer_pools[lid]
        # All scored layers must alias the SAME underlying storage.
        assert pl.untyped_storage().data_ptr() == anchor.untyped_storage().data_ptr(), (
            "scored layer views do not share one storage (VSWA multi-pool); group "
            "segments by pool group and call triton_tri_score_batched_perhead per group.")
        off_bytes = int(pl.data_ptr()) - storage_base
        assert off_bytes >= 0 and off_bytes % elt_size == 0, (
            "layer base not within/aligned to chosen storage")
        layer_base_ptrs[lid] = off_bytes // elt_size

    # ---- page ids: concat per request, request-indexed offsets ----
    page_ids_offsets = torch.zeros(K + 1, dtype=torch.int64, device=device)
    pid_parts = []
    for r in range(K):
        pid = torch.as_tensor(page_ids_list[r], device=device, dtype=torch.int64).reshape(-1)
        pid_parts.append(pid)
        page_ids_offsets[r + 1] = page_ids_offsets[r] + pid.numel()
    page_ids_flat = (torch.cat(pid_parts).contiguous() if pid_parts
                     else torch.zeros(0, dtype=torch.int64, device=device))

    # ---- segments: request-major then layer ----
    L = len(layer_ids)
    nseg = K * L
    seg_req = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_layer = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_seq = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_rstart = torch.empty(nseg, dtype=torch.float32, device=device)
    seg_out_off = torch.empty(nseg, dtype=torch.int64, device=device)
    seg_ntblk = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_offsets = torch.zeros(nseg + 1, dtype=torch.int32, device=device)
    seg_meta: List[SegMeta] = []

    # T_BLOCK is the TOKEN tiling. Each token's score is a single per-tile tl.sum
    # over F (F_BLOCK covers all freqs) with no cross-tile/cross-token fp32 fold, so
    # the stored value is INVARIANT to T_BLOCK -> tuning it is byte-identical (only
    # launch/index overhead changes). num_warps/num_stages are scheduling-only
    # (numerics-neutral). All three are env-tunable at call time for the bench/sweep;
    # the winning constants get baked in once validated byte-identical.
    T_BLOCK = int(os.environ.get("TRI_TBLOCK", "64") or "64")
    F_BLOCK = triton.next_power_of_2(num_freqs)

    mean_cos = torch.empty(nseg, num_freqs, dtype=torch.float32, device=device)
    mean_sin = torch.empty(nseg, num_freqs, dtype=torch.float32, device=device)
    omega_d = omega.to(device=device, dtype=torch.float32)
    offsets_d = offsets.to(device=device, dtype=torch.float32)

    run = 0
    max_ntblk = 1
    s = 0
    for r in range(K):
        sl = int(seq_lens[r])
        rs = float(round_starts[r])
        # offset-collapsed cos/sin: same for all layers of this request, built
        # EXACTLY as triton_tri_score does (mean over offsets of cos/sin(phase)).
        qp = (rs + offsets_d).view(-1, 1)            # [O,1]
        phase = qp * omega_d.view(1, -1)             # [O,F]
        mc = torch.cos(phase).mean(dim=0)            # [F]
        ms = torch.sin(phase).mean(dim=0)            # [F]
        ntblk = (sl + T_BLOCK - 1) // T_BLOCK
        for layer_slot, lid in enumerate(layer_ids):
            seg_req[s] = r
            seg_layer[s] = lid
            seg_seq[s] = sl
            seg_rstart[s] = rs
            seg_out_off[s] = run
            seg_ntblk[s] = ntblk
            mean_cos[s] = mc
            mean_sin[s] = ms
            seg_offsets[s + 1] = run + sl
            seg_meta.append(SegMeta(
                seg_index=s, request_index=r, layer_index=lid,
                seq_len=sl, round_start=rs))
            run += sl
            max_ntblk = max(max_ntblk, ntblk)
            s += 1

    sum_seq = run
    # PER-HEAD output: [H, sum_seq] (sum_seq is the row stride).
    out = torch.empty(num_q_heads * sum_seq, dtype=torch.float32, device=device)

    # calib flattened layer-major [L_all*H*F].
    q_real_f = q_real_LHF.to(device=device, dtype=torch.float32).contiguous().view(-1)
    q_imag_f = q_imag_LHF.to(device=device, dtype=torch.float32).contiguous().view(-1)
    mlr_f = mlr_coef_LHF.to(device=device, dtype=torch.float32).contiguous().view(-1)
    fss_d = freq_scale_sq.to(device=device, dtype=torch.float32).contiguous()

    # grid axis 0 = nseg (on the 2^31-1 axis); axis 1 = max token-blocks per
    # segment (small: cdiv(seq_len, T_BLOCK), <=1024 even at 64k tokens). The
    # unbounded token count NEVER lands on a 65535-limited grid axis.
    #
    # A/B toggle read at CALL time (not just module import) so a single process
    # can flip TRI_KVGRID for equivalence testing. When "1", launch the kvgrid
    # variant with a 3rd grid axis = num_kv_heads (one block per kv_head, more
    # blocks for B200 occupancy); else the original kv_head-loop kernel. Both
    # kernels take the IDENTICAL arg list and produce byte-identical output.
    use_kvgrid = os.environ.get("TRI_KVGRID") == "1"
    # Scheduling-only launch params (numerics-neutral -> byte-identical); env-tunable
    # for the sweep, baked once validated. Empty -> Triton's autochosen defaults.
    _lkw = {}
    _nw = os.environ.get("TRI_NW")
    _ns = os.environ.get("TRI_NS")
    if _nw:
        _lkw["num_warps"] = int(_nw)
    if _ns:
        _lkw["num_stages"] = int(_ns)
    if use_kvgrid:
        grid = (nseg, max_ntblk, num_kv_heads)
        _tri_score_batched_perhead_kvgrid_kernel[grid](
            pool_storage,
            layer_base_ptrs,
            page_ids_flat,
            page_ids_offsets,
            seg_req, seg_layer, seg_seq, seg_rstart, seg_out_off, seg_ntblk,
            q_real_f, q_imag_f, mlr_f,
            mean_cos.view(-1).contiguous(), mean_sin.view(-1).contiguous(),
            fss_d, omega_d, offsets_d,
            out,
            sum_seq,
            num_q_heads, num_kv_heads, num_freqs, head_dim, tokens_per_block,
            kv_factor, int(offsets_d.numel()),
            s_page, s_kv_factor, s_kv_head, s_slot, s_dim,
            USE_MAX=(score_aggregation == "max"),
            T_BLOCK=T_BLOCK, F_BLOCK=F_BLOCK,
            **_lkw,
        )
    else:
        grid = (nseg, max_ntblk)
        _tri_score_batched_perhead_kernel[grid](
            pool_storage,
            layer_base_ptrs,
            page_ids_flat,
            page_ids_offsets,
            seg_req, seg_layer, seg_seq, seg_rstart, seg_out_off, seg_ntblk,
            q_real_f, q_imag_f, mlr_f,
            mean_cos.view(-1).contiguous(), mean_sin.view(-1).contiguous(),
            fss_d, omega_d, offsets_d,
            out,
            sum_seq,
            num_q_heads, num_kv_heads, num_freqs, head_dim, tokens_per_block,
            kv_factor, int(offsets_d.numel()),
            s_page, s_kv_factor, s_kv_head, s_slot, s_dim,
            USE_MAX=(score_aggregation == "max"),
            T_BLOCK=T_BLOCK, F_BLOCK=F_BLOCK,
            **_lkw,
        )
    return out.view(num_q_heads, sum_seq), seg_offsets, seg_meta


def flat_perhead_to_list(perhead_scores: torch.Tensor,
                         seg_offsets: torch.Tensor) -> List[torch.Tensor]:
    """Per-head sibling of ``flat_scores_to_list``: slice perhead_scores[H,sum_seq]
    into the per-segment list of [H, seq_i] views, KEEPING the head dim. Each slice
    is a column-range VIEW (no copy); the only host work is ``nseg`` narrow calls.
    """
    off = seg_offsets.tolist()
    return [perhead_scores[:, off[s]:off[s + 1]] for s in range(len(off) - 1)]


# --------------------------------------------------------------------------- #
# Standalone equivalence test                                                  #
# --------------------------------------------------------------------------- #
