# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendored Triton kernels for the TriAttention KV-eviction pipeline.

The production eviction path is fully fused over (request x layer): one fused score launch over
(request x layer), torch.topk SELECT, and one per-layer compaction.
The kernels here back that path:

  * Fused trig-score over (request x layer) segments, writing un-reduced
    per-query-head rows
    (``triton_tri_score_perhead`` / ``_tri_score_perhead_kernel``
  * In-place compaction over many requests sharing one layer pool
    (``triton_tri_compact`` / ``_tri_compact_kernel``), reusing a
    per-(dtype,device) scratch buffer (``_get_compact_scratch`` /
    ``_COMPACT_SCRATCH``).
  * Flat->list bridge (``flat_perhead_to_list``) and
    the single-storage-base helper (``_flat_from_true_storage_base`` / ``SegMeta``).

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every page/stride offset that can exceed 2^31 (paged-pool reads).
  * mask seq tails not divisible by ``tokens_per_block`` (and freq/dim tails).
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

import os
from typing import List, NamedTuple, Tuple

import torch
import triton
import triton.language as tl

# Move only retained tokens to the compacted prefix. Once the compacted length is
# published, the tail is unreachable, so reordering dropped tokens only adds traffic.
# The opt-out exists for controlled in-process A/B comparisons.
_COMPACT_KEPT_ONLY = os.environ.get("TRIATTN_COMPACT_KEPT_ONLY", "1") == "1"

# --------------------------------------------------------------------------- #
# Block (4): gather / compact kept tokens                                     #
#   gather retained tokens into the compacted (page, slot) prefix             #
#   axis, for BOTH K and V, writing back the touched pages.  Two-pass          #
#   (gather -> scratch -> scatter) to match the reference .clone() semantics   #
#   and avoid the in-place read/write aliasing hazard.                         #
# --------------------------------------------------------------------------- #
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


def _build_compaction_indices(
    keep: torch.Tensor, seq_len: int, *, kept_only: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build source and destination token indices for one compaction segment."""
    keep = keep.to(dtype=torch.int64)
    if kept_only:
        dest = torch.arange(keep.numel(), device=keep.device, dtype=torch.int64)
        return keep, dest

    all_ids = torch.arange(seq_len, device=keep.device, dtype=torch.int64)
    is_dropped = torch.ones(seq_len, device=keep.device, dtype=torch.bool)
    is_dropped[keep] = False
    return torch.cat([keep, all_ids[is_dropped]]), all_ids


@triton.jit
def _tri_compact_kernel(
    pool_ptr,  # one layer's HND pool (strided view; in place)
    new_order_ptr,  # [total] int64: per-dest-token SOURCE local idx (within its request)
    dest_local_ptr,  # [total] int64: per-dest-token LOCAL dest idx (within its request)
    page_base_ptr,  # [total] int64: this token's request's base offset into cat_page_ids
    page_ids_ptr,  # [sum_pages] int64: concatenated per-request physical page ids
    scratch_ptr,  # [kv_factor, num_kv_heads, total, head_dim] scratch
    total,
    num_kv_heads,
    tokens_per_block,
    head_dim,
    s_page,
    s_kv_factor,
    s_kv_head,
    s_slot,
    s_dim,
    PHASE: tl.constexpr,
    D_BLOCK: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PER_HEAD: tl.constexpr,
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
        scratch_base = (
            (kv_half.to(tl.int64) * num_kv_heads + kv_head.to(tl.int64)) * total + gs.to(tl.int64)
        ) * head_dim
        if PHASE == 0:
            # gather: SOURCE local token of this token's request. union (shared
            # reorder) reads new_order[g]; per_head reads new_order[kv_head, g]
            # (a different [kept...,dropped...] permutation per KV head), laid out
            # row-major [num_kv_heads, total].
            if PER_HEAD:
                src = tl.load(new_order_ptr + kv_head.to(tl.int64) * total + gs)
            else:
                src = tl.load(new_order_ptr + gs)
            src_blk = src // tokens_per_block
            src_slot = (src % tokens_per_block).to(tl.int64)
            src_page = tl.load(page_ids_ptr + pbase + src_blk).to(tl.int64)
            src_base = (
                src_page * s_page
                + kv_half.to(tl.int64) * s_kv_factor
                + kv_head.to(tl.int64) * s_kv_head
                + src_slot * s_slot
            )
            val = tl.load(pool_ptr + src_base + d64 * s_dim, mask=m, other=0.0)
            tl.store(scratch_ptr + scratch_base + d64, val, mask=m)
        else:
            # scatter: write to the LOCAL dest slot of this token's request.
            dst = tl.load(dest_local_ptr + gs)
            dst_blk = dst // tokens_per_block
            dst_slot = (dst % tokens_per_block).to(tl.int64)
            dst_page = tl.load(page_ids_ptr + pbase + dst_blk).to(tl.int64)
            dst_base = (
                dst_page * s_page
                + kv_half.to(tl.int64) * s_kv_factor
                + kv_head.to(tl.int64) * s_kv_head
                + dst_slot * s_slot
            )
            val = tl.load(scratch_ptr + scratch_base + d64, mask=m, other=0.0)
            tl.store(pool_ptr + dst_base + d64 * s_dim, val, mask=m)


def triton_tri_compact(pool, page_ids_list, keep_list, seq_len_list, *, dest_list=None):
    """In-place compaction over many requests sharing ONE layer ``pool``.
    By default each request's kept tokens are gathered to the front. An explicit
    ``dest_list`` copies only the supplied source/destination ranges.

    pool:          one layer's HND view [num_pages, kv_factor, nkv, tpb, hd].
    page_ids_list: list of 1-D int page-id tensors, one per request.
    keep_list:     source local-index tensors, one per request.
    seq_len_list:  list of committed token counts, one per request.
    dest_list:     optional explicit destination tensors. When omitted, each
                   source is a complete keep set compacted to [0, keep_count).
    """
    device = pool.device
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape
    if not (len(page_ids_list) == len(keep_list) == len(seq_len_list)):
        raise ValueError("page_ids_list, keep_list, and seq_len_list must have equal lengths")
    if dest_list is not None and len(dest_list) != len(keep_list):
        raise ValueError("dest_list and keep_list must have equal lengths")

    flat_new_order = []
    flat_dest_local = []
    flat_page_base = []
    cat_page_ids = []
    page_cursor = 0
    for request_index, (page_ids, keep, seq_len) in enumerate(
        zip(page_ids_list, keep_list, seq_len_list)
    ):
        keep = keep.to(device=device, dtype=torch.long)
        keep_count = int(keep.numel())
        if dest_list is None:
            if keep_count >= seq_len:
                continue  # nothing to drop for this request
            new_order, dest = _build_compaction_indices(keep, seq_len, kept_only=_COMPACT_KEPT_ONLY)
        else:
            new_order = keep.to(torch.int64)
            dest = dest_list[request_index].to(device=device, dtype=torch.int64)
            if new_order.numel() != dest.numel():
                raise ValueError("Each explicit source and destination must have equal lengths")
            if new_order.numel() == 0:
                continue
        flat_new_order.append(new_order)
        flat_dest_local.append(dest)
        pids = page_ids.to(device=device, dtype=torch.int64)
        flat_page_base.append(
            torch.full((new_order.numel(),), page_cursor, device=device, dtype=torch.int64)
        )
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
        kv_factor, num_kv_heads, total, head_dim
    )
    D_BLOCK = triton.next_power_of_2(head_dim)
    # BLOCK_TOKENS tiles dest tokens per program to amortize the per-token index
    # overhead (the compact is overhead-bound: scattered 128-elem moves, ~40x its
    # memory-traffic floor). Baked default = 16: bench-measured ~14% faster than the
    # one-token-per-program grid (BT=1), and BYTE-IDENTICAL across BT in
    # {8,16,32,64} (test_compact_tune.py / test_compact_equiv.py).
    BLOCK_TOKENS = 16  # verified-best compaction tile (BT sweep: 16 > 32 > 64 tok/s)
    grid = (triton.cdiv(total, BLOCK_TOKENS), kv_factor, num_kv_heads)
    for phase in (0, 1):
        _tri_compact_kernel[grid](
            pool,
            new_order_t,
            dest_local_t,
            page_base_t,
            page_ids_t,
            scratch.reshape(-1),
            total,
            num_kv_heads,
            tokens_per_block,
            head_dim,
            pool.stride(0),
            pool.stride(1),
            pool.stride(2),
            pool.stride(3),
            pool.stride(4),
            PHASE=phase,
            D_BLOCK=D_BLOCK,
            BLOCK_TOKENS=BLOCK_TOKENS,
            PER_HEAD=False,
        )


def triton_tri_compact_fixed(
    pool: torch.Tensor,
    page_ids: torch.Tensor,
    keep: torch.Tensor,
    destination: torch.Tensor,
    page_base: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    """Compact one request with caller-owned fixed-shape buffers.

    The regular wrapper remains the general batched/ragged path. This entry
    point only removes its per-call cat/arange/full allocation chain after the
    manager has validated a stable one-request bucket.
    """
    if pool.ndim != 5:
        raise ValueError("pool must use the five-dimensional HND layout")
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape
    total = int(keep.numel())
    tensors = (page_ids, keep, destination, page_base, scratch)
    if any(tensor.device != pool.device for tensor in tensors):
        raise ValueError("fixed compaction buffers must be on the pool device")
    if (
        page_ids.ndim != 1
        or keep.ndim != 1
        or destination.shape != keep.shape
        or page_base.shape != keep.shape
    ):
        raise ValueError("fixed compaction index buffers have incompatible shapes")
    if any(tensor.dtype != torch.int64 for tensor in tensors[:-1]):
        raise ValueError("fixed compaction indices must use int64")
    need = kv_factor * num_kv_heads * total * head_dim
    if scratch.dtype != pool.dtype or scratch.numel() < need:
        raise ValueError("fixed compaction scratch does not match the pool layout")

    block_tokens = 16
    dim_block = triton.next_power_of_2(head_dim)
    grid = (triton.cdiv(total, block_tokens), kv_factor, num_kv_heads)
    for phase in (0, 1):
        _tri_compact_kernel[grid](
            pool,
            keep,
            destination,
            page_base,
            page_ids,
            scratch,
            total,
            num_kv_heads,
            tokens_per_block,
            head_dim,
            pool.stride(0),
            pool.stride(1),
            pool.stride(2),
            pool.stride(3),
            pool.stride(4),
            PHASE=phase,
            D_BLOCK=dim_block,
            BLOCK_TOKENS=block_tokens,
            PER_HEAD=False,
        )


def triton_tri_compact_perhead(pool, page_ids_list, keep_2d_list, seq_len_list):
    """In-place PER-HEAD compaction over many requests sharing ONE layer ``pool``.

    Same as ``triton_tri_compact`` but each KV head keeps a DIFFERENT token set:
    ``keep_2d_list`` holds one ``[num_kv_heads, keep_count]`` tensor per request
    (sorted-ascending kept local indices per head). For each request+head the
    token axis is reordered ``[kept(head order)..., dropped(ascending)...]`` so
    every slot in ``[0, seq_len)`` still holds a real key/value. This collapses
    the old per-(layer,head) PyTorch gather/scatter loop into ONE batched launch
    per layer (matching the union path's launch count). Byte-identical to the
    PyTorch per-head reorder (validated by test_compact_perhead_equiv.py).
    """
    device = pool.device
    _, kv_factor, num_kv_heads, tokens_per_block, head_dim = pool.shape

    flat_new_order = []  # per request: [num_kv_heads, seq_len]
    flat_dest_local = []  # per request: [seq_len]
    flat_page_base = []
    cat_page_ids = []
    page_cursor = 0
    for page_ids, keep_2d, seq_len in zip(page_ids_list, keep_2d_list, seq_len_list):
        keep_2d = keep_2d.to(device=device, dtype=torch.long)  # [nkv, keep_count]
        keep_count = int(keep_2d.shape[1])
        if keep_count >= seq_len:
            continue  # nothing to drop for this request
        all_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        # dropped slots per head, ascending: mask out the kept slots, then gather
        # the survivors. keep_count is uniform across heads, so the boolean mask
        # selects exactly (seq_len - keep_count) per head -> reshape to [nkv, .].
        is_dropped = torch.ones((num_kv_heads, seq_len), device=device, dtype=torch.bool)
        is_dropped[torch.arange(num_kv_heads, device=device).unsqueeze(1), keep_2d] = False
        dropped = (
            all_ids.unsqueeze(0)
            .expand(num_kv_heads, seq_len)[is_dropped]
            .view(num_kv_heads, seq_len - keep_count)
        )
        new_order = torch.cat([keep_2d, dropped], dim=1).to(torch.int64)  # [nkv, seq_len]
        flat_new_order.append(new_order)
        flat_dest_local.append(all_ids.to(torch.int64))  # 0..seq_len-1 (shared by all heads)
        pids = page_ids.to(device=device, dtype=torch.int64)
        flat_page_base.append(torch.full((seq_len,), page_cursor, device=device, dtype=torch.int64))
        cat_page_ids.append(pids)
        page_cursor += int(pids.numel())

    if not flat_new_order:
        return
    # [num_kv_heads, total] row-major -> kernel reads new_order[kv_head*total + g].
    new_order_t = torch.cat(flat_new_order, dim=1).contiguous()
    dest_local_t = torch.cat(flat_dest_local)
    page_base_t = torch.cat(flat_page_base)
    page_ids_t = torch.cat(cat_page_ids)
    total = int(dest_local_t.numel())

    _need = kv_factor * num_kv_heads * total * head_dim
    scratch = _get_compact_scratch(_need, pool.dtype, device)[:_need].view(
        kv_factor, num_kv_heads, total, head_dim
    )
    D_BLOCK = triton.next_power_of_2(head_dim)
    BLOCK_TOKENS = 16  # same verified-best tile as the union path
    grid = (triton.cdiv(total, BLOCK_TOKENS), kv_factor, num_kv_heads)
    for phase in (0, 1):
        _tri_compact_kernel[grid](
            pool,
            new_order_t.reshape(-1),
            dest_local_t,
            page_base_t,
            page_ids_t,
            scratch.reshape(-1),
            total,
            num_kv_heads,
            tokens_per_block,
            head_dim,
            pool.stride(0),
            pool.stride(1),
            pool.stride(2),
            pool.stride(3),
            pool.stride(4),
            PHASE=phase,
            D_BLOCK=D_BLOCK,
            BLOCK_TOKENS=BLOCK_TOKENS,
            PER_HEAD=True,
        )


def _flat_from_true_storage_base(anchor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Return (flat, storage_base_ptr) where ``flat`` is a 1-D element-typed
    tensor whose ``data_ptr()`` is the TRUE underlying storage base, spanning the
    whole storage WITHOUT copying. Gives the kernel a single typed base pointer;
    per-layer element offsets index into it.

    Anchors on the TRUE storage base (NOT on the min layer data_ptr), so it is
    correct even when the scored layers are a subset of a larger allocation whose
    lowest scored layer has storage_offset > 0. Built via
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
        "unexpected storage layout: anchor data_ptr != base + offset*elt_size"
    )
    return flat, storage_base


class SegMeta(NamedTuple):
    seg_index: int  # position in the request-major (req_slot*L + layer_slot) order
    request_index: int  # slot into page_ids_list / seq_lens / round_starts
    layer_index: int  # ABSOLUTE layer id (from layer_indices)
    seq_len: int  # seq_len_r
    round_start: float  # per-request round_start


@triton.jit
def _tri_score_perhead_kernel(
    pool_storage_ptr,  # ONE typed base pointer; data_ptr() == TRUE storage base.
    #   element-typed view aliasing all scored layers'
    #   shared storage (no copy).
    layer_base_ptrs,  # [num_layers] int64: ELEMENT offset of each layer's
    #   HND view base relative to the TRUE storage base.
    page_ids_ptr,  # [sum_pages] int64: all requests' page ids concat.
    page_ids_offsets,  # [num_requests+1] int64: cum #pages per request.
    # per-SEGMENT metadata (seg = req_slot*L_scored + layer_slot), idx by pid(0):
    seg_req_id,  # [nseg] int32: request slot (into page_ids_offsets etc.)
    seg_layer_id,  # [nseg] int32: ABSOLUTE layer id (indexes layer_base_ptrs + calib)
    seg_seq_len,  # [nseg] int32
    req_round_start,  # [num_requests] fp32
    seg_out_offset,  # [nseg] int64: write base (column) into each [H, sum_seq] row
    seg_n_tblk,  # [nseg] int32: cdiv(seq_len, T_BLOCK)
    # per-LAYER calibration, [L,H,F] flattened layer-major:
    q_real_ptr,  # [L*H*F] fp32
    q_imag_ptr,  # [L*H*F] fp32
    mlr_coef_ptr,  # [L*H*F] fp32
    # per-REQUEST offset-collapsed phase ('mean' path), [num_requests,F] flattened:
    mean_cos_ptr,  # [num_requests*F] fp32
    mean_sin_ptr,  # [num_requests*F] fp32
    # shared freq vectors:
    freq_scale_sq_ptr,  # [F] fp32
    omega_ptr,  # [F] fp32 ('max' path only)
    offsets_ptr,  # [O] fp32 ('max' path only)
    out_ptr,  # [num_q_heads * sum_seq] fp32 (logically [H, sum_seq])
    sum_seq,  # int: total tokens across all segments == row stride of out.
    # scalars uniform across the batch:
    num_q_heads,
    num_kv_heads,
    num_freqs,  # F = head_dim // 2
    head_dim,
    tokens_per_block,
    kv_factor,
    num_offsets,  # O ('max' path only)
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
    rstart = tl.load(req_round_start + req_id)
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
    phys_page = tl.load(page_ids_ptr + page_off + blk_in_seq, mask=t_mask, other=0).to(tl.int64)

    # element offset into pool_storage for (layer, page, KEY=0, *, slot).
    # KEY half is kv_factor index 0 -> its stride term is 0 (matches reference).
    tok_base = layer_elt + phys_page * s_page + slot * s_slot  # [T_BLOCK] int64

    # per-request 'mean'-path phase + shared freq scale.
    mcos = tl.load(mean_cos_ptr + req_id * num_freqs + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + req_id * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)

    # ---- PER-HEAD (position + mlr), GQA-deduped, NO head reduction ----
    # Per-segment trig-score (fp32 math, int64 paged offsets): instead of
    # accumulating into a single head-mean we WRITE each head's own
    # (score + mlr) to its row in the [H, sum_seq] output. Iterating kv_head outer /
    # q-in-group inner gives h = kv_head*group_size + qg, i.e. query-head order
    # 0..num_q_heads-1, so row h holds query head h's score. K (and |K|) is
    # loaded ONCE per KV head, shared by the group_size = H // nkv q-heads.
    group_size = num_q_heads // num_kv_heads
    load_mask = t_mask[:, None] & f_mask[None, :]
    off_re = f64[None, :] * s_dim
    off_im = (num_freqs + f64[None, :]) * s_dim

    kv_head = 0
    while kv_head < num_kv_heads:
        base = tok_base + kv_head.to(tl.int64) * s_kv_head  # [T_BLOCK]
        # paged K loaded ONCE for this KV head (shared by group_size q-heads).
        k_re = tl.load(pool_storage_ptr + base[:, None] + off_re, mask=load_mask, other=0.0).to(
            tl.float32
        )
        k_im = tl.load(pool_storage_ptr + base[:, None] + off_im, mask=load_mask, other=0.0).to(
            tl.float32
        )
        kmag = tl.sqrt(k_re * k_re + k_im * k_im)  # once per KV head

        qg = 0
        while qg < group_size:
            h = kv_head * group_size + qg
            calib_off = (layer_id.to(tl.int64) * num_q_heads + h) * num_freqs
            qre = tl.load(q_real_ptr + calib_off + f, mask=f_mask, other=0.0)
            qim = tl.load(q_imag_ptr + calib_off + f, mask=f_mask, other=0.0)
            mlrc = tl.load(mlr_coef_ptr + calib_off + f, mask=f_mask, other=0.0)

            # complex product Q . conj(K) -- the trig importance score.
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
                    per_f = fss[None, :] * (
                        prod_real * cphase[None, :] - prod_imag * sphase[None, :]
                    )
                    offset_score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)
                    score = tl.maximum(score, offset_score)
                    o += 1
            else:
                # 'mean': offset loop collapsed into mean_cos/mean_sin.
                per_f = fss[None, :] * (prod_real * mcos[None, :] - prod_imag * msin[None, :])
                score = tl.sum(tl.where(f_mask[None, :], per_f, 0.0), axis=1)

            # position-INDEPENDENT MLR term (reuses the per-KV-head |K|).
            mlr_f = kmag * mlrc[None, :] * fss[None, :]
            mlr = tl.sum(tl.where(f_mask[None, :], mlr_f, 0.0), axis=1)

            # write this head's score at row h, column (out_base + t). No mean.
            tl.store(out_ptr + h.to(tl.int64) * sum_seq + out_base + t, score + mlr, mask=t_mask)
            qg += 1
        kv_head += 1


def _launch_tri_score_perhead(
    grid: tuple,
    pointer_args: tuple,
    geometry_args: tuple,
    *,
    score_aggregation: str,
    token_block: int,
    num_freqs: int,
) -> None:
    """Launch the shared score ABI for eager and fixed metadata owners."""
    if score_aggregation not in ("mean", "max"):
        raise ValueError(f"unsupported score aggregation: {score_aggregation}")
    _tri_score_perhead_kernel[grid](
        *pointer_args,
        *geometry_args,
        USE_MAX=(score_aggregation == "max"),
        T_BLOCK=token_block,
        F_BLOCK=triton.next_power_of_2(num_freqs),
    )


class _FixedScoreGroup:
    """Persistent score metadata/output for one storage group and sequence bucket."""

    def __init__(
        self,
        layer_pools: List[torch.Tensor],
        layer_indices: List[int],
        max_requests: int,
        page_count: int,
        seq_len: int,
        num_q_heads: int,
        page_ids: torch.Tensor,
        q_real_LHF: torch.Tensor,
        q_imag_LHF: torch.Tensor,
        mlr_coef_LHF: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        omega: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        if not layer_indices or min(max_requests, page_count, seq_len) <= 0:
            raise ValueError("fixed score group requires non-empty positive geometry")
        self.max_requests = max_requests
        self.seq_len = seq_len
        self.num_q_heads = num_q_heads
        self.num_layers = len(layer_indices)
        p0 = layer_pools[layer_indices[0]]
        if p0.ndim != 5:
            raise ValueError("fixed score group requires HND pools")
        device = p0.device
        q_real_LHF = q_real_LHF.to(device=device, dtype=torch.float32).contiguous()
        q_imag_LHF = q_imag_LHF.to(device=device, dtype=torch.float32).contiguous()
        mlr_coef_LHF = mlr_coef_LHF.to(device=device, dtype=torch.float32).contiguous()
        freq_scale_sq = freq_scale_sq.to(device=device, dtype=torch.float32).contiguous()
        omega = omega.to(device=device, dtype=torch.float32).contiguous()
        offsets = offsets.to(device=device, dtype=torch.float32).contiguous()
        _, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
        if num_q_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.num_freqs = head_dim // 2
        strides = tuple(int(value) for value in p0.stride())
        self.geometry_args = (
            num_q_heads,
            num_kv_heads,
            self.num_freqs,
            head_dim,
            tokens_per_block,
            kv_factor,
            int(offsets.numel()),
            *strides,
        )
        anchor = layer_pools[layer_indices[0]]
        pool_storage, storage_base = _flat_from_true_storage_base(anchor)
        element_size = anchor.element_size()
        layer_base_ptrs = torch.zeros(len(layer_pools), dtype=torch.int64, device=device)
        for layer in layer_indices:
            pool = layer_pools[layer]
            if (
                pool.untyped_storage().data_ptr() != anchor.untyped_storage().data_ptr()
                or tuple(pool.shape[1:]) != tuple(p0.shape[1:])
                or tuple(pool.stride()) != strides
                or pool.dtype != p0.dtype
            ):
                raise ValueError("fixed score layers must share one uniform storage")
            offset_bytes = int(pool.data_ptr()) - storage_base
            if offset_bytes < 0 or offset_bytes % element_size:
                raise ValueError("fixed score layer offset is invalid")
            layer_base_ptrs[layer] = offset_bytes // element_size

        num_segments = max_requests * self.num_layers
        segment = torch.arange(num_segments, device=device)
        page_ids_offsets = (
            torch.arange(max_requests + 1, dtype=torch.int64, device=device) * page_count
        )
        seg_req = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
            self.num_layers
        )
        seg_layer = torch.tensor(layer_indices, dtype=torch.int32, device=device).repeat(
            max_requests
        )
        seg_seq = torch.full((num_segments,), seq_len, dtype=torch.int32, device=device)
        seg_out_off = segment.to(torch.int64) * seq_len
        self.token_block = 64
        self.max_ntblk = (seq_len + self.token_block - 1) // self.token_block
        seg_ntblk = torch.full((num_segments,), self.max_ntblk, dtype=torch.int32, device=device)
        self.seg_offsets = (
            torch.arange(num_segments + 1, dtype=torch.int32, device=device) * seq_len
        )
        self.output = torch.empty(
            num_q_heads * num_segments * seq_len, dtype=torch.float32, device=device
        )
        if page_ids.shape != (max_requests, page_count) or not page_ids.is_contiguous():
            raise ValueError("page ids do not match fixed score geometry")
        self.pointer_prefix = (
            pool_storage,
            layer_base_ptrs,
            page_ids.view(-1),
            page_ids_offsets,
            seg_req,
            seg_layer,
            seg_seq,
        )
        self.pointer_middle = (
            seg_out_off,
            seg_ntblk,
            q_real_LHF.view(-1),
            q_imag_LHF.view(-1),
            mlr_coef_LHF.view(-1),
        )
        self.pointer_tail = (freq_scale_sq, omega, offsets)

    def launch(
        self,
        request_count: int,
        round_starts_device: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        score_aggregation: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Launch the unchanged score kernel using the persistent metadata."""
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        num_segments = request_count * self.num_layers
        sum_seq = num_segments * self.seq_len
        output = self.output[: self.num_q_heads * sum_seq]
        _launch_tri_score_perhead(
            (num_segments, self.max_ntblk),
            (
                *self.pointer_prefix,
                round_starts_device,
                *self.pointer_middle,
                mean_cos.view(-1),
                mean_sin.view(-1),
                *self.pointer_tail,
                output,
            ),
            (sum_seq, *self.geometry_args),
            score_aggregation=score_aggregation,
            token_block=self.token_block,
            num_freqs=self.num_freqs,
        )
        return output.view(self.num_q_heads, sum_seq), self.seg_offsets[: num_segments + 1]


def triton_tri_score_perhead(
    layer_pools,  # list[L_all] of HND views get_buffers(l,'HND').
    page_ids_list,  # list[K] of 1-D int tensors: each request's page ids.
    seq_lens,  # list[K] int: per-request committed seq_len (>top_B).
    round_starts,  # list[K] int/float: per-request round_start.
    q_real_LHF,  # [L_all,H,F] fp32 (stack of per-layer E_q.real).
    q_imag_LHF,  # [L_all,H,F] fp32.
    mlr_coef_LHF,  # [L_all,H,F] fp32 (E_q_norm - |E_q|).
    freq_scale_sq,  # [F] fp32.
    omega,  # [F] fp32.
    offsets,  # [O] fp32.
    num_q_heads,  # H.
    score_aggregation="mean",
    layer_indices=None,  # optional list of ABSOLUTE layer ids to score (default all).
) -> Tuple[torch.Tensor, torch.Tensor, List[SegMeta]]:
    """Per-query-head fused trig-score: returns un-reduced per-query-head
    scores (one row per query head) over (request x layer) segments.

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
    L_all = len(layer_pools)
    layer_ids = list(range(L_all)) if layer_indices is None else list(layer_indices)
    device = layer_pools[layer_ids[0]].device
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
        f"score kernel requires num_q_heads ({num_q_heads}) % num_kv_heads ({num_kv_heads}) == 0"
    )
    s_page, s_kv_factor, s_kv_head, s_slot, s_dim = (
        p0.stride(0),
        p0.stride(1),
        p0.stride(2),
        p0.stride(3),
        p0.stride(4),
    )
    for lid in layer_ids:
        pl = layer_pools[lid]
        assert tuple(pl.shape[1:]) == tuple(p0.shape[1:]), (
            "uniform (kv_factor,nkv,tpb,hd) required across scored layers; "
            "group by config otherwise."
        )
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
            "scored layer views do not share one storage; group segments by pool "
            "and call triton_tri_score_perhead per group."
        )
        off_bytes = int(pl.data_ptr()) - storage_base
        assert off_bytes >= 0 and off_bytes % elt_size == 0, (
            "layer base not within/aligned to chosen storage"
        )
        layer_base_ptrs[lid] = off_bytes // elt_size

    # ---- page ids: concat per request, request-indexed offsets ----
    page_ids_offsets = torch.zeros(K + 1, dtype=torch.int64, device=device)
    pid_parts = []
    for r in range(K):
        pid = torch.as_tensor(page_ids_list[r], device=device, dtype=torch.int64).reshape(-1)
        pid_parts.append(pid)
        page_ids_offsets[r + 1] = page_ids_offsets[r] + pid.numel()
    page_ids_flat = (
        torch.cat(pid_parts).contiguous()
        if pid_parts
        else torch.zeros(0, dtype=torch.int64, device=device)
    )

    # ---- segments: request-major then layer ----
    L = len(layer_ids)
    nseg = K * L
    seg_req = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_layer = torch.empty(nseg, dtype=torch.int32, device=device)
    seg_seq = torch.empty(nseg, dtype=torch.int32, device=device)
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
    T_BLOCK = 64  # verified token-tile width

    request_round_starts = torch.as_tensor(round_starts, dtype=torch.float32, device=device)
    mean_cos = torch.empty(K, num_freqs, dtype=torch.float32, device=device)
    mean_sin = torch.empty(K, num_freqs, dtype=torch.float32, device=device)
    omega_d = omega.to(device=device, dtype=torch.float32).contiguous()
    offsets_d = offsets.to(device=device, dtype=torch.float32).contiguous()

    run = 0
    max_ntblk = 1
    s = 0
    for r in range(K):
        sl = int(seq_lens[r])
        rs = float(round_starts[r])
        # offset-collapsed cos/sin: same for all layers of this request, built
        # EXACTLY as triton_tri_score does (mean over offsets of cos/sin(phase)).
        qp = (rs + offsets_d).view(-1, 1)  # [O,1]
        phase = qp * omega_d.view(1, -1)  # [O,F]
        mc = torch.cos(phase).mean(dim=0)  # [F]
        ms = torch.sin(phase).mean(dim=0)  # [F]
        mean_cos[r] = mc
        mean_sin[r] = ms
        ntblk = (sl + T_BLOCK - 1) // T_BLOCK
        for layer_slot, lid in enumerate(layer_ids):
            seg_req[s] = r
            seg_layer[s] = lid
            seg_seq[s] = sl
            seg_out_off[s] = run
            seg_ntblk[s] = ntblk
            seg_offsets[s + 1] = run + sl
            seg_meta.append(
                SegMeta(seg_index=s, request_index=r, layer_index=lid, seq_len=sl, round_start=rs)
            )
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
    # Production launch: the kv_head-loop scoring kernel with Triton's autochosen
    # schedule.
    _launch_tri_score_perhead(
        (nseg, max_ntblk),
        (
            pool_storage,
            layer_base_ptrs,
            page_ids_flat,
            page_ids_offsets,
            seg_req,
            seg_layer,
            seg_seq,
            request_round_starts,
            seg_out_off,
            seg_ntblk,
            q_real_f,
            q_imag_f,
            mlr_f,
            mean_cos.view(-1).contiguous(),
            mean_sin.view(-1).contiguous(),
            fss_d,
            omega_d,
            offsets_d,
            out,
        ),
        (
            sum_seq,
            num_q_heads,
            num_kv_heads,
            num_freqs,
            head_dim,
            tokens_per_block,
            kv_factor,
            int(offsets_d.numel()),
            s_page,
            s_kv_factor,
            s_kv_head,
            s_slot,
            s_dim,
        ),
        score_aggregation=score_aggregation,
        token_block=T_BLOCK,
        num_freqs=num_freqs,
    )
    return out.view(num_q_heads, sum_seq), seg_offsets, seg_meta


def flat_perhead_to_list(
    perhead_scores: torch.Tensor, seg_offsets: torch.Tensor
) -> List[torch.Tensor]:
    """Slice perhead_scores[H, sum_seq] into the per-segment list of [H, seq_i]
    views (one per request x layer segment), keeping the head dim. Each slice
    is a column-range VIEW (no copy); the only host work is ``nseg`` narrow calls.
    """
    off = seg_offsets.tolist()
    return [perhead_scores[:, off[s] : off[s + 1]] for s in range(len(off) - 1)]


def fixed_perhead_segment_views(
    perhead_scores: torch.Tensor,
    request_count: int,
    layer_count: int,
    seq_len: int,
) -> torch.Tensor:
    """View fixed score output as ``[H, R, L, S]`` without reading offsets.

    The fixed score launcher writes request-major, then layer-major segments of
    one exact sequence length.  Its geometry is already known by the caller, so
    materializing ``seg_offsets`` on the host would add a needless CUDA sync.
    """
    if min(request_count, layer_count, seq_len) <= 0:
        raise ValueError("fixed score segment geometry must be positive")
    expected_width = request_count * layer_count * seq_len
    if perhead_scores.ndim != 2 or int(perhead_scores.shape[1]) != expected_width:
        raise ValueError(
            "fixed score output width does not match request/layer/sequence geometry"
        )
    return perhead_scores.view(
        int(perhead_scores.shape[0]), request_count, layer_count, seq_len
    )


# --------------------------------------------------------------------------- #
# Standalone equivalence test                                                  #
# --------------------------------------------------------------------------- #
