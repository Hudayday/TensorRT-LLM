"""KVCacheManagerV2 subclass for TriAttention: it returns the physical blocks
freed by decode-time eviction back to the pool (the paper's capacity gain),
without touching any V2 manager code. Selected by
``_util._get_model_kv_cache_manager_cls`` when the kv-cache-compression config is
TriAttention.

A subclass (not a post-hoc hook) is required: V2's generation update_resources
resets history to the full logical length (``max_beam - 1``) every step and
requires ``capacity >= history``, so nothing running after it can hold capacity
below the logical length. This override instead uses the COMPACTED physical
history (``max_beam - evicted``) for evicting requests and shrinks capacity to
match, freeing the trailing blocks through the V2 ``resize()`` API.

Three things keep the shrink safe (see the block-reclaim investigation):
  * Stream order -- wait the compaction's CUDA event on the manager stream before
    the shrink, so a freed page is reused only after the compaction that read it.
  * Dense layers -- their trailing blocks are not SWA-stale, so resize()'s
    shrink-assert would reject them; release those page locks first (the lock's
    destructor sets BAD_PAGE_INDEX) and resize() then recycles the pages.
  * History is monotonic through resize(), so set ``_history_length`` directly
    first, exactly like the existing num_cached reconcile.

Reclaim is always on: this manager exists specifically to return eviction-freed
blocks to the pool, so there is no on/off gate.
"""

from __future__ import annotations

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.logger import logger


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


class TriAttentionKVCacheManagerV2(KVCacheManagerV2):
    """V2 manager that returns eviction-freed blocks to the pool (stream-ordered,
    dense-layer-safe)."""

    def update_resources(
        self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None
    ) -> None:
        # Reclaim is mandatory: this manager always returns eviction-freed blocks
        # to the pool (the capacity gain is the whole point of this manager).
        # Context requests: identical to the base manager (eviction is decode-only).
        from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

        if not self.is_draft:
            from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
                _update_kv_cache_draft_token_location,
            )

            _update_kv_cache_draft_token_location(
                self, scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
            )
        for req in scheduled_batch.context_requests:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue
            if self.enable_block_reuse and not self.is_draft and not req.is_dummy_request:
                if req.context_current_position > kv_cache.num_committed_tokens:
                    tokens = self._augment_tokens_for_block_reuse(
                        req.get_tokens(0),
                        req,
                        start=kv_cache.num_committed_tokens,
                        end=req.context_current_position,
                    )
                    kv_cache.commit(tokens)
                if req.context_remaining_length == 0:
                    kv_cache.stop_committing()
            else:
                kv_cache.resize(None, req.context_current_position)

        for req in scheduled_batch.generation_requests:
            kv_cache = self.kv_cache_map.get(req.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue  # overlap-suspended victim: skip; resumed next iter
            ev = int(getattr(req, "py_triattn_evicted", 0) or 0)
            if ev <= 0 or req.state in (
                LlmRequestState.GENERATION_COMPLETE,
                LlmRequestState.CONTEXT_INIT,
            ):
                # Non-evicting / completing requests: native V2 behavior.
                new_capacity = (
                    None
                    if req.state
                    in (LlmRequestState.GENERATION_COMPLETE, LlmRequestState.CONTEXT_INIT)
                    else kv_cache.capacity - req.py_rewind_len
                )
                kv_cache.resize(new_capacity, req.max_beam_num_tokens - 1)
                continue
            # Evicting request: compacted physical history + shrunk capacity.
            target_hist = req.max_beam_num_tokens - ev  # = num_cached + 1
            tpb = kv_cache.tokens_per_block
            target_cap = _div_up(target_hist + 1, tpb) * tpb  # +1 for next-token grow
            self._resize_with_reclaim(kv_cache, req, target_cap, target_hist)

    def _resize_with_reclaim(self, kv_cache, req, target_cap, target_hist) -> None:
        # 1) Order the manager stream behind the compaction, so the reuse-gating
        #    finish_event (recorded at resize entry) dominates the compaction.
        ev_obj = getattr(req, "py_triattn_compaction_event", None)
        if ev_obj is not None:
            try:
                self._stream.cuda_stream.wait_event(ev_obj)
            except Exception:
                pass
        # 2) Bypass the monotonic-decrease guard, exactly like the reconcile.
        if target_hist < kv_cache.history_length:
            kv_cache._history_length = target_hist
        tpb = kv_cache.tokens_per_block
        committed = getattr(kv_cache, "_num_committed_blocks", 0)
        # Never shrink below the committed prefix (block_reuse off => committed = 0).
        target_cap = max(target_cap, committed * tpb)
        new_num_blocks = _div_up(target_cap, tpb)
        try:
            # 3) When shrinking, release the trailing dense locks first so resize()'s
            #    shrink precondition holds (V2 then deletes + recycles the pages).
            if new_num_blocks < kv_cache.num_blocks:
                self._unlock_trailing_dense_blocks(kv_cache, new_num_blocks)
            # One resize handles both grow (next-token block) and shrink (free).
            kv_cache.resize(target_cap, target_hist)
        except Exception as e:  # defensive: fall back to a safe no-free history set
            logger.warning(
                f"TriAttention reclaim fell back to no-free for {req.py_request_id}: {e}"
            )
            if target_hist < kv_cache.history_length:
                kv_cache._history_length = target_hist
            kv_cache.resize(None, target_hist)

    def _unlock_trailing_dense_blocks(self, kv_cache, new_num_blocks) -> None:
        """Release the trailing dense-layer (window_size is None) page locks so
        resize()'s shrink precondition holds. Dropping a _SharedPageLock sets its
        _base_page_indices entry to BAD_PAGE_INDEX (via __del__ -> unlock); resize()
        then deletes the SeqBlocks and recycles the pages.

        The lock must be dropped with NO lingering Python reference, so __del__ runs
        inside this _record_event scope (where finish_event is armed) and not at GC
        time (when unlock() would hit finish_event == None). Mirrors
        _unlock_stale_blocks. Freed unconditionally: block_reuse is off here, so the
        trailing blocks are never committed/reusable."""
        from tensorrt_llm.runtime.kv_cache_manager_v2._life_cycle_registry import AttnLifeCycle

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
