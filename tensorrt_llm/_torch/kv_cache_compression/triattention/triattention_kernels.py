# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendored Triton kernels for the TriAttention KV-eviction pipeline.

The production eviction path is fully fused over (request x layer) x KV head: ONE score
launch spanning ALL dense layers (layers may live in distinct storages with distinct
block tables; the kernel takes per-layer absolute base addresses), torch.topk SELECT,
and one per-layer compaction.
The kernels here back that path:

  * Fused trig-score over (request x layer) segments, writing un-reduced
    per-query-head rows
    (``triton_tri_score_perhead`` / ``_tri_score_perhead_kernel``
  * In-place compaction over many requests sharing one layer pool
    (``triton_tri_compact`` / ``_tri_compact_kernel``), reusing a
    per-(dtype,device) scratch buffer (``_get_compact_scratch`` /
    ``_COMPACT_SCRATCH``).
  * Flat->list bridge (``flat_perhead_to_list``) and ``SegMeta``.

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


def cpp_sparse_compact_supported(pool: torch.Tensor) -> bool:
    """The upstream C++ single-pass compact kernel needs an interleaved-K/V
    HND pool (kv_factor == 2, contiguous) and the thop op compiled in."""
    return (
        pool.ndim == 5
        and pool.shape[1] == 2
        and pool.is_contiguous()
        and pool.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and hasattr(torch.ops.trtllm, "sparse_kv_cache_compact")
    )


def cpp_sparse_compact(pool, page_ids_list, keep_list, seq_len_list):
    """Union compaction via the upstream single-pass C++ kernel
    (updateSparseKvCacheAfterFmha through the sparse_kv_cache_compact
    thop op). One launch covers all requests of ONE layer pool; K and V move
    in the same pass (no global scratch: grid (1, nkv, batch) with an ordered
    in-block sweep, so it wins at high batch and loses at batch 1 -- see the
    compact bench). Falls back are the caller's responsibility via
    ``cpp_sparse_compact_supported``.
    """
    device = pool.device
    _, kv_factor, num_kv_heads, tokens_per_block, _ = pool.shape
    if kv_factor != 2:
        raise ValueError("cpp_sparse_compact requires an interleaved K/V pool (kv_factor == 2)")
    tables, keeps = [], []
    for page_ids, keep, seq_len in zip(page_ids_list, keep_list, seq_len_list):
        keep32 = keep.to(device=device, dtype=torch.int32)
        if keep32.numel() >= seq_len:
            continue  # nothing to drop for this request
        tables.append(page_ids.to(device=device, dtype=torch.int32).reshape(-1))
        keeps.append(keep32)
    if not keeps:
        return
    batch = len(keeps)
    max_pages = max(int(t.numel()) for t in tables)
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    for i, t in enumerate(tables):
        page_table[i, : t.numel()] = t
    offsets_host = [0]
    for k in keeps:
        offsets_host.append(offsets_host[-1] + int(k.numel()))
    offsets = torch.tensor(offsets_host, dtype=torch.int32, device=device)
    flat = torch.cat(keeps)
    # union: every KV head keeps the same token set -> replicate rows.
    indices = flat.unsqueeze(0).expand(num_kv_heads, -1).contiguous()
    torch.ops.trtllm.sparse_kv_cache_compact(pool, page_table, indices, offsets)


class SegMeta(NamedTuple):
    seg_index: int  # position in the request-major (req_slot*L + layer_slot) order
    request_index: int  # slot into page_ids_list / seq_lens / round_starts
    layer_index: int  # ABSOLUTE layer id (from layer_indices)
    seq_len: int  # seq_len_r
    round_start: float  # per-request round_start


@triton.jit
def _tri_score_perhead_kernel(
    pool_anchor_ptr,  # typed pool pointer; used ONLY to infer the element type
    #   for the int->pointer cast below (its data is never read through it).
    layer_base_addrs,  # [num_layers] int64: ABSOLUTE device address of each
    #   scored layer's HND base. Layers do NOT need to share one storage:
    #   each segment casts its own layer's address back to a typed pointer, so
    #   all address arithmetic stays inside that layer's own allocation.
    page_ids_ptr,  # [sum_page_tables] int64: page tables concatenated in
    #   SEGMENT order (V2 allocates pages per layer, so block tables are
    #   per-(request, layer), not per-request).
    seg_page_off,  # [nseg] int64: offset of this segment's page table into
    #   page_ids_ptr.
    # per-SEGMENT metadata (seg = req_slot*L_scored + layer_slot), idx by pid(0):
    seg_req_id,  # [nseg] int32: request slot (round_start / mean phase lookup)
    seg_layer_id,  # [nseg] int32: ABSOLUTE layer id (indexes layer_base_addrs + calib)
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
    # per-layer HND element strides (uniform across scored layers):
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
    # KV heads are grid-parallel (axis 2): iterations of the former kv_head
    # loop shared NO data (each KV head reads its own K and writes its own
    # output rows), so hoisting it onto the grid multiplies parallelism with
    # zero extra HBM traffic. The q-in-group loop below stays inside the
    # program because it REUSES this head's K from registers (GQA dedup).
    kv_head = tl.program_id(2)

    # tiles beyond this segment's own block count do nothing (ragged grid).
    n_tblk = tl.load(seg_n_tblk + seg)
    if t_blk >= n_tblk:
        return

    seq_len = tl.load(seg_seq_len + seg)
    layer_id = tl.load(seg_layer_id + seg)
    req_id = tl.load(seg_req_id + seg)
    rstart = tl.load(req_round_start + req_id)
    out_base = tl.load(seg_out_offset + seg)
    # This segment's layer base: an absolute address cast back to a pool-typed
    # pointer. TRT-LLM V2 exposes every layer as its own TensorWrapper storage,
    # so "element offset relative to one shared storage" does not exist; the
    # per-layer absolute address is the same device-pointer-array pattern the
    # C++ backends use (KVBlockArray / grouped-GEMM pointer arrays).
    layer_ptr = tl.load(layer_base_addrs + layer_id).to(
        tl.pointer_type(pool_anchor_ptr.dtype.element_ty)
    )
    page_off = tl.load(seg_page_off + seg)

    f = tl.arange(0, F_BLOCK)
    f_mask = f < num_freqs
    f64 = f.to(tl.int64)

    # ---- token tile of THIS segment ----
    t = t_blk * T_BLOCK + tl.arange(0, T_BLOCK)
    t_mask = t < seq_len
    blk_in_seq = t // tokens_per_block
    slot = (t % tokens_per_block).to(tl.int64)
    # page_off + blk_in_seq stays within this segment's page table for in-range
    # tokens (t_mask). For masked tail tokens the load is masked (other=0) so the
    # possibly-out-of-this-segment index is never dereferenced.
    phys_page = tl.load(page_ids_ptr + page_off + blk_in_seq, mask=t_mask, other=0).to(tl.int64)

    # element offset into THIS layer's pool for (page, KEY=0, *, slot).
    # KEY half is kv_factor index 0 -> its stride term is 0 (matches reference).
    tok_base = phys_page * s_page + slot * s_slot  # [T_BLOCK] int64

    # per-request 'mean'-path phase + shared freq scale.
    mcos = tl.load(mean_cos_ptr + req_id * num_freqs + f, mask=f_mask, other=0.0)
    msin = tl.load(mean_sin_ptr + req_id * num_freqs + f, mask=f_mask, other=0.0)
    fss = tl.load(freq_scale_sq_ptr + f, mask=f_mask, other=0.0)

    # ---- PER-HEAD (position + mlr), GQA-deduped, NO head reduction ----
    # This program scores ONE KV head's token tile for the group_size q-heads
    # that share it. K (and |K|) is loaded ONCE and reused across the group;
    # h = kv_head*group_size + qg keeps query-head order 0..num_q_heads-1, so
    # every head's math is bit-for-bit identical to the looped variant.
    group_size = num_q_heads // num_kv_heads
    load_mask = t_mask[:, None] & f_mask[None, :]
    off_re = f64[None, :] * s_dim
    off_im = (num_freqs + f64[None, :]) * s_dim

    base = tok_base + kv_head.to(tl.int64) * s_kv_head  # [T_BLOCK]
    # paged K loaded ONCE for this KV head (shared by group_size q-heads).
    k_re = tl.load(layer_ptr + base[:, None] + off_re, mask=load_mask, other=0.0).to(tl.float32)
    k_im = tl.load(layer_ptr + base[:, None] + off_im, mask=load_mask, other=0.0).to(tl.float32)
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
    """Persistent score metadata/output for one sequence bucket.

    Since the per-layer absolute-address ABI, ONE group can span dense layers
    living in DISTINCT storages with DISTINCT block tables: ``page_ids`` holds
    one page table per (page-table slot, request) and ``page_table_slots`` maps
    each scored layer to its slot.
    """

    def __init__(
        self,
        layer_pools: List[torch.Tensor],
        layer_indices: List[int],
        max_requests: int,
        page_count: int,
        seq_len: int,
        num_q_heads: int,
        page_ids: torch.Tensor,  # [num_slots, max_requests, page_count] int64
        page_table_slots: List[int],  # per scored layer: slot into page_ids
        q_real_LHF: torch.Tensor,
        q_imag_LHF: torch.Tensor,
        mlr_coef_LHF: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        omega: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        if not layer_indices or min(max_requests, page_count, seq_len) <= 0:
            raise ValueError("fixed score group requires non-empty positive geometry")
        if len(page_table_slots) != len(layer_indices):
            raise ValueError("page_table_slots must align with layer_indices")
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
        self.num_kv_heads = int(num_kv_heads)
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
        # Per-layer ABSOLUTE base addresses. Layers may live in distinct
        # storages (V2 TensorWrapper-per-layer); only geometry must be uniform.
        element_size = p0.element_size()
        layer_base_addrs = torch.zeros(len(layer_pools), dtype=torch.int64, device=device)
        for layer in layer_indices:
            pool = layer_pools[layer]
            if (
                tuple(pool.shape[1:]) != tuple(p0.shape[1:])
                or tuple(pool.stride()) != strides
                or pool.dtype != p0.dtype
            ):
                raise ValueError("fixed score layers must share one uniform geometry")
            address = int(pool.data_ptr())
            if address % element_size:
                raise ValueError("fixed score layer base is not element-aligned")
            layer_base_addrs[layer] = address
        # The anchor pool is passed as a typed kernel argument ONLY so the
        # kernel can recover the element type for the int->pointer cast.
        self.anchor_pool = p0

        num_segments = max_requests * self.num_layers
        segment = torch.arange(num_segments, device=device)
        seg_req = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
            self.num_layers
        )
        seg_layer = torch.tensor(layer_indices, dtype=torch.int32, device=device).repeat(
            max_requests
        )
        seg_seq = torch.full((num_segments,), seq_len, dtype=torch.int32, device=device)
        seg_out_off = segment.to(torch.int64) * seq_len
        # Per-segment page-table offsets into page_ids.view(-1): segment
        # (r, layer_slot) reads table [page_table_slots[layer_slot], r].
        if page_ids.ndim != 3 or tuple(page_ids.shape[1:]) != (max_requests, page_count):
            raise ValueError("page ids do not match fixed score geometry")
        if not page_ids.is_contiguous():
            raise ValueError("fixed score page ids must be contiguous")
        slots_t = torch.tensor(page_table_slots, dtype=torch.int64, device=device)
        if int(slots_t.max()) >= int(page_ids.shape[0]):
            raise ValueError("page table slot exceeds staged page-id planes")
        req_idx = torch.arange(max_requests, dtype=torch.int64, device=device).repeat_interleave(
            self.num_layers
        )
        slot_idx = slots_t.repeat(max_requests)
        seg_page_off = (slot_idx * max_requests + req_idx) * page_count
        self.token_block = 64
        self.max_ntblk = (seq_len + self.token_block - 1) // self.token_block
        seg_ntblk = torch.full((num_segments,), self.max_ntblk, dtype=torch.int32, device=device)
        self.seg_req = seg_req
        self.seg_seq = seg_seq
        self.seg_ntblk = seg_ntblk
        self.seg_offsets = (
            torch.arange(num_segments + 1, dtype=torch.int32, device=device) * seq_len
        )
        self.output = torch.empty(
            num_q_heads * num_segments * seq_len, dtype=torch.float32, device=device
        )
        self.pointer_prefix = (
            self.anchor_pool,
            layer_base_addrs,
            page_ids.view(-1),
            seg_page_off,
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

    def stage_lengths(self, valid_seq_lens: torch.Tensor, request_count: int) -> None:
        """Stage per-request valid lengths into this fixed upper-bound bucket."""
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        if valid_seq_lens.ndim != 1 or valid_seq_lens.numel() < request_count:
            raise ValueError("valid sequence lengths do not match the fixed score cohort")
        num_segments = request_count * self.num_layers
        torch.index_select(
            valid_seq_lens,
            0,
            self.seg_req[:num_segments],
            out=self.seg_seq[:num_segments],
        )
        torch.add(
            self.seg_seq[:num_segments],
            self.token_block - 1,
            out=self.seg_ntblk[:num_segments],
        )
        torch.div(
            self.seg_ntblk[:num_segments],
            self.token_block,
            rounding_mode="floor",
            out=self.seg_ntblk[:num_segments],
        )

    def launch(
        self,
        request_count: int,
        round_starts_device: torch.Tensor,
        mean_cos: torch.Tensor,
        mean_sin: torch.Tensor,
        score_aggregation: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Launch the fused score kernel using the persistent metadata."""
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        num_segments = request_count * self.num_layers
        sum_seq = num_segments * self.seq_len
        output = self.output[: self.num_q_heads * sum_seq]
        _launch_tri_score_perhead(
            (num_segments, self.max_ntblk, self.num_kv_heads),
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
    page_ids_list,  # list[K] of 1-D int tensors: per-request page ids (used for
    #   ALL scored layers when ``page_ids_per_layer`` is None).
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
    page_ids_per_layer=None,  # optional [request][layer_slot] page-id tensors for
    #   layers with DISTINCT block tables (V2 allocates pages per layer). When
    #   given, ``page_ids_list`` is ignored.
) -> Tuple[torch.Tensor, torch.Tensor, List[SegMeta]]:
    """Per-query-head fused trig-score over (request x layer) segments.

    Scored layers may live in DISTINCT storages (V2 TensorWrapper-per-layer):
    the kernel receives each layer's ABSOLUTE base address and casts it back to
    a typed pointer, so no shared-storage anchor is required. Geometry
    (shape[1:], strides, dtype) must still be uniform across scored layers.

    Returns (perhead_scores[H, sum_seq], seg_offsets[nseg+1] int32, seg_meta).
    seg_meta is REQUEST-MAJOR, then layer. To split into per-segment [H, seq_i]
    views use ``flat_perhead_to_list(perhead_scores, seg_offsets)``.
    """
    assert score_aggregation in ("mean", "max")
    L_all = len(layer_pools)
    layer_ids = list(range(L_all)) if layer_indices is None else list(layer_indices)
    device = layer_pools[layer_ids[0]].device
    K = len(page_ids_list) if page_ids_per_layer is None else len(page_ids_per_layer)
    assert len(seq_lens) == K and len(round_starts) == K
    if page_ids_per_layer is not None:
        assert all(len(per_layer) == len(layer_ids) for per_layer in page_ids_per_layer), (
            "page_ids_per_layer must provide one page table per scored layer"
        )

    # ---- HND geometry (uniform across scored layers; assert it) ----
    p0 = layer_pools[layer_ids[0]]
    _num_pages0, kv_factor, num_kv_heads, tokens_per_block, head_dim = p0.shape
    num_freqs = head_dim // 2
    # GQA-deduped score kernel: KV heads run on grid axis 2 and the group_size
    # = H // nkv q-heads sharing one K run in-program; exact divisibility keeps
    # the h order (and thus the per-head math) bit-for-bit.
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

    # ---- per-layer ABSOLUTE base addresses (no shared-storage anchor) ----
    elt_size = p0.element_size()
    layer_base_addrs = torch.zeros(L_all, dtype=torch.int64, device=device)
    for lid in layer_ids:
        address = int(layer_pools[lid].data_ptr())
        assert address % elt_size == 0, "layer base not element-aligned"
        layer_base_addrs[lid] = address

    # ---- page tables, concatenated in SEGMENT order ----
    # V2 allocates pages per layer, so a request's block table differs by
    # layer. With ``page_ids_per_layer`` each (request, layer) segment gets its
    # own table; otherwise all layers of a request share one table and their
    # segments simply point at the same slice (no duplication).
    L = len(layer_ids)
    nseg = K * L
    pid_parts = []
    seg_page_off = torch.empty(nseg, dtype=torch.int64, device=device)
    page_cursor = 0
    if page_ids_per_layer is None:
        req_table_off = []
        for r in range(K):
            pid = torch.as_tensor(page_ids_list[r], device=device, dtype=torch.int64).reshape(-1)
            pid_parts.append(pid)
            req_table_off.append(page_cursor)
            page_cursor += pid.numel()
        for r in range(K):
            seg_page_off[r * L : (r + 1) * L] = req_table_off[r]
    else:
        s = 0
        for r in range(K):
            for layer_slot in range(L):
                pid = torch.as_tensor(
                    page_ids_per_layer[r][layer_slot], device=device, dtype=torch.int64
                ).reshape(-1)
                pid_parts.append(pid)
                seg_page_off[s] = page_cursor
                page_cursor += pid.numel()
                s += 1
    page_ids_flat = (
        torch.cat(pid_parts).contiguous()
        if pid_parts
        else torch.zeros(0, dtype=torch.int64, device=device)
    )

    # ---- segments: request-major then layer ----
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
    # launch/index overhead changes).
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
        # EXACTLY as the reference does (mean over offsets of cos/sin(phase)).
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

    # grid: axis 0 = nseg (on the 2^31-1 axis); axis 1 = max token-blocks per
    # segment (small: cdiv(seq_len, T_BLOCK), <=1024 even at 64k tokens); axis
    # 2 = KV heads (data-independent across heads -> free grid parallelism).
    _launch_tri_score_perhead(
        (nseg, max_ntblk, num_kv_heads),
        (
            p0,
            layer_base_addrs,
            page_ids_flat,
            seg_page_off,
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
        raise ValueError("fixed score output width does not match request/layer/sequence geometry")
    return perhead_scores.view(int(perhead_scores.shape[0]), request_count, layer_count, seq_len)


# --------------------------------------------------------------------------- #
# Standalone equivalence test                                                  #
# --------------------------------------------------------------------------- #
