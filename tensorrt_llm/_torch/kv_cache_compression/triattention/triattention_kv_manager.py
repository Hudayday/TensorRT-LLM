"""TriAttention-specific KVCacheManagerV2 subclass that RECLAIMS the blocks freed
by decode-time eviction (the paper's capacity gain), WITHOUT modifying any V2
manager code. Routed in via _util._get_model_kv_cache_manager_cls when the
sparse_attention_config is TriAttention.

Why a subclass (not a hook): V2's gen update_resources sets history = max_beam-1
(the FULL logical length) every step and requires capacity >= history, so a
post-hoc hook can never hold capacity below the logical length. The subclass
overrides update_resources to use the COMPACTED physical history
(max_beam - py_tri_evicted) for tri-evicting requests and shrinks capacity to
match, freeing the trailing blocks via the V2 resize() API.

Correctness (see the block-free investigation):
  - STREAM ORDERING: the page-reuse-gating finish_event is recorded on the
    manager stream at resize entry; it must dominate the compaction kernel. The
    compression manager records a CUDA event right after the compaction
    (request.py_tri_compaction_event); we wait it on the manager stream before
    the shrink so the freed page's reuse is ordered behind the compaction.
  - DENSE-LAYER UNLOCK: dense (window_size is None) life cycles' trailing blocks
    are never SWA-stale, so V2's resize shrink-assert (indices[new:]==BAD_PAGE_INDEX)
    fails. We release those _SharedPageLocks first (dropping the lock auto-sets
    BAD_PAGE_INDEX via its destructor); resize() then deletes the SeqBlocks and
    recycles the pages (no double-free; del stays resize's job).
  - MONOTONIC GUARD: history can't be lowered through resize(); bypass exactly as
    the existing reconcile does (write _history_length before resize).

Gated by `.tri_free_blocks` (TriAttention._TRI_FREE_BLOCKS). OFF => pure
pass-through to KVCacheManagerV2 (byte-identical; validated Stage 1).
"""
from __future__ import annotations

import os

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.logger import logger

_TRI_FREE_BLOCKS = os.path.exists("/scratch/triattn_e2e/.tri_free_blocks")
# correctness probe (remove before PR): logs per-eviction freed-block count so we can
# assert evicted blocks are physically returned to the pool (freed > 0 on shrink).
_TRI_CORRECTNESS = os.path.exists("/scratch/triattn_e2e/.tri_correctness")


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


class TriAttentionKVCacheManagerV2(KVCacheManagerV2):
    """V2 manager that returns eviction-freed blocks to the pool (stream-ordered,
    dense-layer-safe). Pure pass-through when block-free is disabled."""

    def update_resources(self, scheduled_batch, attn_metadata=None,
                         kv_cache_dtype_byte_size=None) -> None:
        if not _TRI_FREE_BLOCKS:
            return super().update_resources(scheduled_batch, attn_metadata,
                                            kv_cache_dtype_byte_size)
        # Handle context requests exactly as the base (no eviction there).
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
        if not self.is_draft:
            from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
                _update_kv_cache_draft_token_location)
            _update_kv_cache_draft_token_location(
                self, scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)
        for req in scheduled_batch.context_requests:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue
            if self.enable_block_reuse and not self.is_draft and not req.is_dummy_request:
                if req.context_current_position > kv_cache.num_committed_tokens:
                    tokens = self._augment_tokens_for_block_reuse(
                        req.get_tokens(0), req,
                        start=kv_cache.num_committed_tokens,
                        end=req.context_current_position)
                    kv_cache.commit(tokens)
                if req.context_remaining_length == 0:
                    kv_cache.stop_committing()
            else:
                kv_cache.resize(None, req.context_current_position)

        for req in scheduled_batch.generation_requests:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue  # overlap-suspended victim: skip; resumed next iter
            ev = int(getattr(req, "py_tri_evicted", 0) or 0)
            if ev <= 0 or req.state in (LlmRequestState.GENERATION_COMPLETE,
                                        LlmRequestState.CONTEXT_INIT):
                # Native V2 behavior for non-evicting / completing requests.
                new_capacity = None if req.state in (
                    LlmRequestState.GENERATION_COMPLETE,
                    LlmRequestState.CONTEXT_INIT) else kv_cache.capacity - req.py_rewind_len
                kv_cache.resize(new_capacity, req.max_beam_num_tokens - 1)
                continue
            # tri-evicting request: compacted physical history + shrunk capacity.
            target_hist = req.max_beam_num_tokens - ev          # = num_cached + 1
            tpb = kv_cache.tokens_per_block
            target_cap = _div_up(target_hist + 1, tpb) * tpb     # +1 for next-token grow
            self._tri_resize_with_reclaim(kv_cache, req, target_cap, target_hist)

    def _tri_resize_with_reclaim(self, kv_cache, req, target_cap, target_hist) -> None:
        # 1) order the manager stream behind the compaction so the reuse-gating
        #    finish_event (recorded at resize entry) dominates it.
        ev_obj = getattr(req, "py_tri_compaction_event", None)
        if ev_obj is not None:
            try:
                self._stream.cuda_stream.wait_event(ev_obj)
            except Exception:
                pass
        # 2) bypass the monotonic-decrease guard exactly like the reconcile.
        if target_hist < kv_cache.history_length:
            kv_cache._history_length = target_hist
        tpb = kv_cache.tokens_per_block
        committed = getattr(kv_cache, "_num_committed_blocks", 0)
        # never shrink below the committed prefix (block_reuse OFF -> committed=0)
        target_cap = max(target_cap, committed * tpb)
        new_num_blocks = _div_up(target_cap, tpb)
        _old_nb = int(kv_cache.num_blocks)   # correctness probe: blocks before resize
        try:
            # 3) only when SHRINKING blocks: release the trailing dense locks so
            #    resize()'s shrink precondition holds (V2 then deletes + recycles).
            if new_num_blocks < kv_cache.num_blocks:
                self._tri_unlock_trailing_dense_blocks(kv_cache, new_num_blocks)
            # one resize handles grow (alloc next-token block) and shrink (free).
            kv_cache.resize(target_cap, target_hist)
            if _TRI_CORRECTNESS:
                try:
                    import json as _cj
                    with open("/scratch/triattn_e2e/correctness_blockfree.jsonl", "a") as _cf:
                        _cf.write(_cj.dumps({"rid": int(req.py_request_id),
                            "blocks_before": _old_nb, "blocks_after": int(kv_cache.num_blocks),
                            "freed": _old_nb - int(kv_cache.num_blocks),
                            "target_cap_tokens": int(target_cap), "tpb": int(tpb),
                            "kept_kv_tokens": int(kv_cache.num_blocks) * int(tpb),
                            "history_len": int(kv_cache.history_length)}) + "\n")
                except Exception:
                    pass
        except Exception as e:  # defensive: degrade to safe no-free history set
            logger.warning(f"TriAttention reclaim fell back to no-free for "
                           f"{req.py_request_id}: {e}")
            if target_hist < kv_cache.history_length:
                kv_cache._history_length = target_hist
            kv_cache.resize(None, target_hist)

    def _tri_unlock_trailing_dense_blocks(self, kv_cache, new_num_blocks) -> None:
        """Release the trailing dense (window_size is None) blocks' page locks so
        resize()'s shrink precondition holds. Dropping a _SharedPageLock auto-sets
        its _base_page_indices entry to BAD_PAGE_INDEX (via __del__ -> unlock).
        resize() then deletes the SeqBlocks + recycles the pages (no double-free).

        CRITICAL: do NOT bind the lock to a local var. The drop must happen with no
        lingering Python reference so __del__ fires INSIDE this _record_event scope
        (where kv_cache.finish_event is armed); otherwise the destructor runs at GC
        time outside the scope and unlock() hits finish_event==None. (Mirrors
        _unlock_stale_blocks, which only ever holds `.holder`, never the lock.)
        We free unconditionally (no hold_for_commit): block_reuse is OFF for
        TriAttention reclaim, so trailing blocks are never committed/reusable."""
        from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import (
            AttnLifeCycle)
        manager = kv_cache.manager
        ssm_lc_id = manager._life_cycles.ssm_life_cycle_id
        num_blocks = kv_cache.num_blocks
        with kv_cache._record_event():
            for lc_idx, lc in manager._life_cycles.items():
                if lc_idx == ssm_lc_id:
                    continue
                if not (isinstance(lc, AttnLifeCycle) and lc.window_size is None):
                    continue  # SWA layers handled by V2 _unlock_stale_blocks
                for ordinal in range(new_num_blocks, num_blocks):
                    block = kv_cache._blocks[ordinal]
                    for beam_block in block.pages:
                        if beam_block[lc_idx] is not None:
                            # drop with no lingering ref -> __del__ in-scope -> BAD set
                            beam_block[lc_idx] = None
