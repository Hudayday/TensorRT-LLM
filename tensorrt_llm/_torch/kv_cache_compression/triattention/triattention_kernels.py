# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU kernels for the TriAttention KV-eviction pipeline.

The production path uses one fixed-shape trig-score launch across all dense
layers, CuTE-DSL TopK selection, and grouped C++ compaction. This module owns
the score kernel and its persistent launcher; selection and compaction live in
their respective runtime modules.

House rules honored throughout:
  * fp32 math (loads up-cast to fp32, fp32 accumulators, fp32 score output).
  * int64 for every page/stride offset that can exceed 2^31 (paged-pool reads).
  * mask seq tails not divisible by ``tokens_per_block`` (and freq/dim tails).
  * the kernels are vendored in this module (no lazy-load hub).
"""

from __future__ import annotations

from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _canonicalize_topk_scores_kernel(
    scores,
    seq_lens,
    selected_indices,
    WIDTH: tl.constexpr,
    KEEP_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rewrite one score row so a second TopK resolves boundary ties by index."""
    row = tl.program_id(0)
    row_scores = scores + row * WIDTH
    row_selected = selected_indices + row * KEEP_COUNT

    threshold = float("inf")
    for start in tl.static_range(0, KEEP_COUNT, BLOCK):
        selected_offset = start + tl.arange(0, BLOCK)
        selected_mask = selected_offset < KEEP_COUNT
        token_index = tl.load(
            row_selected + selected_offset,
            mask=selected_mask,
            other=0,
        )
        selected_score = tl.load(
            row_scores + token_index,
            mask=selected_mask,
            other=float("inf"),
        ).to(tl.float32)
        threshold = tl.minimum(threshold, tl.min(selected_score, axis=0))

    seq_len = tl.load(seq_lens + row)
    for start in tl.static_range(0, WIDTH, BLOCK):
        token_index = start + tl.arange(0, BLOCK)
        in_width = token_index < WIDTH
        valid = in_width & (token_index < seq_len)
        score = tl.load(
            row_scores + token_index,
            mask=valid,
            other=float("-inf"),
        ).to(tl.float32)
        canonical_score = tl.where(
            score > threshold,
            1.0,
            tl.where(score == threshold, -token_index.to(tl.float32), float("-inf")),
        )
        tl.store(row_scores + token_index, canonical_score, mask=in_width)


def canonicalize_topk_scores(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    selected_indices: torch.Tensor,
    keep_count: int,
) -> None:
    """Canonicalize exact TopK boundary ties without host synchronization.

    ``selected_indices`` is a provisional CuTE-DSL TopK result. For each row,
    scores above its selected minimum become ``1`` and scores equal to that
    boundary become ``-token_index``. A second TopK therefore keeps every
    strictly-better token and fills the remaining slots with the lowest token
    indices from the exact tie. The caller must provide finite valid FP32
    scores and ``keep_count <= seq_lens[row] <= scores.shape[1]``.
    """
    keep_count = int(keep_count)
    if not scores.is_cuda:
        raise ValueError("deterministic TopK canonicalization requires CUDA scores")
    if scores.ndim != 2 or scores.dtype != torch.float32 or not scores.is_contiguous():
        raise ValueError("TopK canonicalization requires contiguous two-dimensional FP32 scores")
    rows, width = scores.shape
    if rows <= 0 or not 1 <= keep_count <= width or width > 2**24:
        raise ValueError("TopK canonicalization requires 1 <= keep_count <= width <= 2**24")
    if (
        seq_lens.shape != (rows,)
        or seq_lens.dtype != torch.int32
        or seq_lens.device != scores.device
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("TopK canonicalization sequence lengths do not match the score rows")
    if (
        selected_indices.shape != (rows, keep_count)
        or selected_indices.dtype != torch.int32
        or selected_indices.device != scores.device
        or not selected_indices.is_contiguous()
    ):
        raise ValueError("TopK canonicalization indices do not match the requested selection")
    _canonicalize_topk_scores_kernel[(rows,)](
        scores,
        seq_lens,
        selected_indices,
        WIDTH=width,
        KEEP_COUNT=keep_count,
        BLOCK=256,
        num_warps=4,
    )


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
    out_ptr,  # [request, layer, query_head, decode_token] fp32
    output_width,
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
    TOKEN_START: tl.constexpr,
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
    absolute_t = t + TOKEN_START
    t_mask = absolute_t < seq_len
    blk_in_seq = absolute_t // tokens_per_block
    slot = (absolute_t % tokens_per_block).to(tl.int64)
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
                per_f = fss[None, :] * (prod_real * cphase[None, :] - prod_imag * sphase[None, :])
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

        # Segments are request-major then layer-major. Write the decode-only
        # score directly in the selector's [request, layer, head, token] layout.
        out_offset = (seg.to(tl.int64) * num_q_heads + h) * output_width + t
        tl.store(out_ptr + out_offset, score + mlr, mask=t_mask)
        qg += 1


def _launch_tri_score_perhead(
    grid: tuple,
    pointer_args: tuple,
    geometry_args: tuple,
    *,
    score_aggregation: str,
    token_start: int,
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
        TOKEN_START=token_start,
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
        prompt_len: int = 0,
    ) -> None:
        if not layer_indices or min(max_requests, page_count, seq_len) <= 0:
            raise ValueError("fixed score group requires non-empty positive geometry")
        if prompt_len < 0 or prompt_len >= seq_len:
            raise ValueError("fixed score prompt length must leave a non-empty decode region")
        if len(page_table_slots) != len(layer_indices):
            raise ValueError("page_table_slots must align with layer_indices")
        self.max_requests = max_requests
        self.prompt_len = prompt_len
        self.output_width = seq_len - prompt_len
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
        seg_req = torch.arange(max_requests, dtype=torch.int32, device=device).repeat_interleave(
            self.num_layers
        )
        seg_layer = torch.tensor(layer_indices, dtype=torch.int32, device=device).repeat(
            max_requests
        )
        seg_seq = torch.full((num_segments,), seq_len, dtype=torch.int32, device=device)
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
        self.max_ntblk = (self.output_width + self.token_block - 1) // self.token_block
        seg_ntblk = torch.full((num_segments,), self.max_ntblk, dtype=torch.int32, device=device)
        self.seg_req = seg_req
        self.seg_seq = seg_seq
        self.seg_ntblk = seg_ntblk
        self.output = torch.empty(
            max_requests,
            self.num_layers,
            num_q_heads,
            self.output_width,
            dtype=torch.float32,
            device=device,
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
            self.token_block - 1 - self.prompt_len,
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
    ) -> torch.Tensor:
        """Return decode-only scores as ``[request, layer, head, token]``."""
        if request_count <= 0 or request_count > self.max_requests:
            raise ValueError("request count exceeds fixed score capacity")
        num_segments = request_count * self.num_layers
        output = self.output[:request_count]
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
            (self.output_width, *self.geometry_args),
            score_aggregation=score_aggregation,
            token_start=self.prompt_len,
            token_block=self.token_block,
            num_freqs=self.num_freqs,
        )
        return output
