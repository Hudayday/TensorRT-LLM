<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVFP4 Cold-Page Codec Design Record — 2026-08-17

## Record metadata

```text
Title: NVFP4 cold-page codec on the current KVCM2 cold-page contract
Record ID / revision: NVFP4-KVCM2-2026-08-17 / r7
Status: exact code-parent build/unit validation, wheel acceptance, functional E2E, and Nsight campaign complete
Date / owner: 2026-08-17 / KV-cache compression side
Repository / publication branch / validated code parent:
  Hudayday/TensorRT-LLM
  nvfp4-pr17512-latest-e2e-clean-20260817
  base 984183e480cd7e23f79d73005fb395d923803b28
  validated code parent 297061375c923b0a4d9f6c8b82af2f9c19dccad3
  documentation-only publication commit: the commit containing this record
Stacked-PR base: NVIDIA/TensorRT-LLM PR #17512 at 984183e480cd7e23f79d73005fb395d923803b28
Reviewers or approval source: explicit user decision on 2026-08-17
```

The approval covers the existing NVFP4 algorithm and a minimal high-level Python construction adapter. It does not
approve a change to native KVCM2 or StorageManager ownership, migration, tier mapping, stream/event behavior, the
native KVCM2 binding, the pure-runtime implementation, or the public cold-page ABI.

## Scope

### Goal

Provide an optional NVFP4 representation for KVCM2 cold pages while using the current PR #17512 codec contract
unchanged. The active hot representation remains the model-selected FP16, BF16, or FP8 KV layout. Every cold level
stores one fixed-size opaque page blob and KVCM2 remains the sole owner of page selection, storage, migration,
publication, rollback, and events.

The exact port must:

- preserve FP16/BF16/FP8-to-NVFP4 and NVFP4-to-FP16/BF16/FP8 conversion;
- work for Host cold storage and for Disk storage through KVCM2's existing pinned staging path;
- retain raw cold-to-cold copies, including Host-to-Disk and Disk-to-Host;
- add no new representation-specific KVCM2 path and no model-, tier-, or geometry-specific kernel family beyond the
  three general boundary kernels;
- keep the final diff under native `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`, pure-runtime
  `tensorrt_llm/runtime/kv_cache_manager_v2/`, and the native KVCM2 binding empty;
- limit KVCM-side adaptation to final-target provider plumbing in high-level PyExecutor construction; and
- document, rather than patch, any native KVCM2 or StorageManager incompatibility discovered during the port.

### Non-goals

- Changing KVCM2 allocation, lifecycle-to-pool-group mapping, tier quotas, eviction, migration routing, or events.
- Changing `IKvCacheColdPageCodec`, its nanobind ownership contract, or the default lossless codec.
- Adding a codec callback from C++ into Python or a late codec setter.
- Compressing hot GPU pages in place or changing attention metadata and attention kernels.
- Adding token eviction, logical-length changes, or an iteration-driven resource-manager hook.
- Adding an MLA-specific, SSM-specific, model-specific, or geometry-specific CUDA kernel.
- Fixing known KVCM2 side-path bugs that are not required by the NVFP4 main path.
- Claiming support for a model/layout that does not expose conventional K and V buffers matching the configured
  geometry.
- Making a performance claim in this port. Native byte/layout correctness plus E2E activation and storage-route
  evidence are required; lossy NVFP4 token agreement remains diagnostic rather than a general accuracy gate.

### Current and intended flow

```text
TorchLlmArgs.kv_cache_compression_config
  -> KvCacheCreator selects boundary quantization only for the final target KVCM2
  -> QuantizationCompression loads ModelOpt scalar K/V calibration
  -> high-level KVCacheManagerV2 receives cold_page_codec_provider
  -> _create_kv_cache_manager_v2_impl creates a fresh native codec
  -> KVCacheManagerPy consumes cold_page_codec during construction
  -> existing StorageManager calls codec.configure(all hot PoolGroupDesc)
  -> codec derives immutable per-lifecycle plans and fixed cold-page sizes
  -> existing KVCM2 builds its cold lifecycle <-> pool-group mapping
  -> existing migration engine invokes encode/decode only across hot/cold boundaries
  -> attention consumes the restored hot representation with no metadata change
```

`QuantizationCompression` is a construction-only provider for algorithm selection, calibration, and layer metadata.
It need not survive native construction. The native codec owns the immutable lowered plans after construction. KVCM2
owns all storage and migration state.

The direct high-level Python handoff is the accepted integration boundary. Base `984183e` does not expose this
construction seam, so the consumer adapter supplies a fresh codec before native manager construction. It does not
modify native KVCM2, StorageManager, migration, the codec ABI, the native binding, or the pure-runtime implementation.
Codec-enabled construction fails closed instead of silently retrying with a GPU-only/lossless manager. A late `impl`
replacement remains unsafe because codec configuration, cold sizes, staging buffers, and cold pools are fixed during
native construction.

## PR #17512 lineage and module map

The old and force-pushed feature heads are siblings, not a linear before/after range. Use these ranges when reviewing
the owner change:

| View | Range | Size |
|---|---|---|
| old cold-page feature | `48df89d..01b1ee56` | 24 files, +3024/-551 |
| current prerequisite rename | `e8965de..56d2672` | hot-cache-level rename |
| current cold-page feature only | `56d2672..984183e` | 52 files, +4858/-934 |
| current whole PR | `e8965de..984183e` | 55 files, +4910/-985 |

Do not interpret raw `01b1ee56..984183e` as the feature delta: their merge base is `48df89d`, so that comparison also
contains unrelated mainline churn. The current PR is organized as follows:

| Module | Current responsibility |
|---|---|
| codec ABI/default codec | fixed-size opaque cold blob; configure/query/encode/decode; batching representative and page-index location |
| construction/ownership | native manager consumes one codec `unique_ptr`; `None` creates the default lossless concat codec |
| hot/cold tier mapping | immutable bidirectional `LifeCyclePoolGroupMapping` for hot and cold representations; ratio/stat projection |
| migration transaction | level-specific source/destination pool-group routing, lifecycle-keyed queues, raw same-representation copies, codec boundary calls, rollback and event fences |
| staging/copy | pinned page ring, device-index ring, and Disk's pinned Host bridge |
| physical storage lifetime | StorageManager-owned shared GPU physical allocator; levels borrow it and are destroyed first |
| control plane/stats | nanobind mirror, high-level wrapper, pure-Python structural mirror, metrics and rebalancing consumers |
| owner verification | default-codec, cold-page, staging, allocator, stats, and documentation suites |

Tier divergence itself was not first introduced by the latest force-push. Pre-feature base `48df89d` used one shared
`mLifeCycleGrouping`; old feature head `01b1ee56` already introduced `mLifeCycleGroupings[level]` and one common cold
grouping. The latest feature replaces those forward-only vectors with bidirectional mappings and makes every consumer
level-aware. Host and Disk still deliberately share `mColdPoolGroupMapping` because they store the same encoded blob;
there is no Host-specific or Disk-specific NVFP4 pool-group mapping. Grouping equal-size cold lifecycles into one
physical pool is an allocator optimization, not a format requirement.

## Evidence state

| ID | Classification | Claim | Source or experiment | Status |
|---|---|---|---|---|
| E-001 | `FACT` | PR #17512 exposes one fixed-size opaque cold page per lifecycle through `IKvCacheColdPageCodec`. | `coldPageCodec.h`; `kv-cache-cold-page-codec.md` at base `984183e` | confirmed |
| E-002 | `FACT` | Hot and cold lifecycle-to-pool-group mappings are independent; all cold levels share the cold representation. | PR #17512 storage design and source at base `984183e` | confirmed |
| E-003 | `FACT` | Hot-to-Disk and Disk-to-hot already use KVCM2-owned pinned staging; Host-to-Disk and Disk-to-Host are raw blob copies. | PR #17512 migration-path documentation and source | confirmed |
| E-004 | `FACT` | One NVFP4 block scale represents 16 adjacent head-dimension values. | existing NVFP4 boundary kernel arithmetic and layout | confirmed |
| E-005 | `FACT` | `headDim % 16 == 0` makes each row contain complete scale groups and also makes the packed element count integral. | `Nvfp4BoundaryKernelParams` and codec size calculation | confirmed |
| E-006 | `FACT` | No `tokensPerPage % 4`, scale-count `% 4`, or K/V scale-placement divisibility is part of the logical format. | existing compact-layout helpers and byte-tail paths | confirmed |
| E-007 | `FACT` | `tokensPerPage` is copied from KVCM2 `tokens_per_block`; the codec does not choose page granularity. | Python layer-config builder | confirmed |
| E-008 | `MEASUREMENT` | Exact code parent `2970613` passes 80 native and 41 Python tests on B200 after an incremental `100-real` build. | jobs 3719753 and 3720301 | confirmed |
| E-009 | `MEASUREMENT` | The immutable r19 wheel bytes have four-way native and three-way Python provenance, isolated installed-wheel imports, and unchanged source/build postflight. | B200 job 3721144 | confirmed |
| E-010 | `FACT` | Base `984183e` has no high-level codec construction seam: its wrapper directly invokes `KVCacheManagerPy` twice, while the native codec is consumed before cold sizing/allocation. The consumer-side Python adapter now supplies the codec before native construction without changing native/storage/runtime paths. | wrapper lines 1073-1086; binding lines 2155-2196; StorageManager lines 355-459 | resolved by the authorized Python adapter |
| E-011 | `MEASUREMENT` | Exact-wheel Qwen3 FP8 completes all seven functional arms across GPU-only, raw/NVFP4 Host-only, Host+Disk, and Disk-only. | B200 job 3721270 | confirmed for activation, route, reuse, and storage mechanism |
| E-012 | `MEASUREMENT` | Exact-wheel Qwen3.5 BF16 completes all seven functional arms while Attention is compressed and SSM/conv remains lossless. | B200 job 3721272 | confirmed for hybrid activation, route, reuse, and storage mechanism |
| E-013 | `FACT` | One fixed 2,048-half-group tile covers every admitted geometry by linear half-group indexing. It uses at most 8 KiB of packed staging plus 1 KiB of scale staging and may cross a token/head row without splitting a 16-value scale group. | general boundary-kernel source and arithmetic | confirmed; cross-row/large-`headDim` representative cases pass in B200 job 3720301 |
| E-014 | `MEASUREMENT` | Exact-wheel Qwen3 FP8 and Qwen3.5 BF16 traces associate every boundary launch with the matching codec range; the four Disk-enabled traces contain `pwrite`/`pread` in forced offload/onboard ranges. | B200 jobs 3721683 and 3721681 | all six topology traces and their target-process trace-health, whole-page, and cold-migration proofs PASS |

## Ownership and lifetime

| State or operation | Canonical owner | Lifetime | Producer | Consumer | Mutation/reuse edge |
|---|---|---|---|---|---|
| Compression config | `TorchLlmArgs` | executor construction | user/YAML | `KvCacheCreator` | immutable after validation |
| ModelOpt K/V scales | `QuantizationCompression` | native construction | checkpoint loader | native layer configs | read once; finite positive scalars |
| Hot pool layout and addresses | KVCM2 | manager | KVCM2 storage setup | codec `configure()` | authoritative; never duplicated by Python |
| Lowered layer plan | `Nvfp4ColdPageCodec` | native manager | codec `configure()` | boundary kernels | immutable after one successful configure |
| Cold page size | codec, consumed by KVCM2 | manager | `queryColdPageBytes()` | cold pool construction/staging | fixed per lifecycle |
| Hot/cold slots and page mappings | KVCM2 | page/migration | KVCM2 | codec call and attention | publication only after fenced submission |
| Page-index pairs | KVCM2 | one codec call/stream work | migration batching | codec | host pairs consumed before return |
| CUDA stream and completion event | KVCM2 | migration transaction | migration engine | codec and slot/page reuse | normal success path only enqueues; KVCM2 fences; synchronous launch failure may drain already-submitted codec work before throwing |
| Cold Host/Disk blob | KVCM2 | cold slot | codec or raw copy | raw copy or codec | opaque to every KVCM2 consumer |
| Attention hot KV | KVCM2/model runtime | hot slot | model write or codec decode | attention | unchanged runtime dtype/layout |

## Invariants and support matrix

### Invariants

| ID | Invariant | Authoritative owner | Failure mode | Verification |
|---|---|---|---|---|
| I-001 | No final diff under native C++ KVCM2/StorageManager, the native KVCM2 binding, or pure-runtime KVCM2. High-level final-target construction plumbing is the sole permitted KVCM-side adaptation. | port boundary | review failure | path-filtered diff |
| I-002 | One cold `SlotId` selects one physical cold pool and one opaque logical blob at every cold level. | KVCM2 PR #17512 | incompatible construction | existing KVCM2 tests plus E2E |
| I-003 | Host and Disk use byte-identical cold payloads; cold-to-cold migration does not invoke the codec. | KVCM2 | corruption or needless requantization | Disk E2E and logs/counters |
| I-004 | For layer element count `N = H * P * D`, the payload is `N/2 + N/2 + N/16 + N/16 = 9N/8` bytes. | codec | wrong slot size/out-of-bounds access | host size tests and kernel byte tests |
| I-005 | A layer payload is `[K packed | V packed | K block scales | V block scales]`; K and V scales remain independent. | codec/kernel | wrong dequantization | deterministic K/V scale tests |
| I-006 | `headDim` is positive and divisible by 16; `numKvHeads` and `tokensPerPage` are positive. No other model-geometry divisibility is required. | codec admission | incomplete scale group | negative and odd-page-size tests |
| I-007 | Every configured attention layer maps to exactly one K and one V hot buffer whose byte size matches its geometry/runtime dtype. | codec `configure()` | wrong address/stride | native codec tests |
| I-008 | FP8 runtime KV remains supported; FP8 source scales and NVFP4 destination scales are independent. FP8 source scales are required only for an FP8 runtime and impose no unused admission condition on FP16/BF16. | compression manager/kernel | scale-domain corruption or false rejection | dtype-specific scale-admission and FP8 round-trip tests |
| I-009 | On success, encode/decode only enqueue on the supplied stream and never publish, synchronize, release, or retry pages themselves. If a later chunk has a synchronous launch failure after earlier chunks were submitted, the codec drains that already-submitted work before throwing; a drain failure is fail-stop. | codec/KVCM2 boundary | race/use-after-free or unsafe rollback | native tests and current KVCM2 contract |
| I-010 | Unsupported non-attention lifecycles use PR #17512's default lossless codec, without format-specific branches. | codec composition | hybrid-state corruption | lossless fallback test |
| I-011 | Each layer starts at a 16-byte-aligned relative cold offset. The staging base may have arbitrary byte alignment; the same kernel uses 16-byte, 8-byte, or byte transfers as appropriate. Padding is only inter-record/trailing padding, at most 15 bytes per layer, is zero-filled by encode, and is never interpreted by decode. | codec layout/kernel | wrong vector access or stale bytes persisted to Disk | layout, deterministic full-Slot, and base+1 round-trip tests |
| I-012 | `tokensPerPage` follows KVCM2 `tokens_per_block` exactly. | KVCM2/config adapter | incompatible indexing | config/unit test and E2E manifest |
| I-013 | One native launch contains at most 128 local Attention layers. This is a descriptor-ABI bound, not a model-geometry divisibility rule. | boundary kernel plan | launch-argument overflow | admission test and model-scope audit |
| I-014 | Boundary transfer tiles are linear in half-groups and bounded at 2,048 half-groups. No row-count, GCD, or shared-memory admission condition is part of the format. | boundary kernel | hidden geometry limit | cross-row and 2-half-group-tail tests |
| I-015 | Codec batching remains lifecycle-keyed even when KVCM maps equal-size cold lifecycles into one physical PoolGroup. Physical allocation sharing does not imply transform equivalence. | codec | applying one lifecycle plan to another lifecycle's buffers/scales | identity representative tests and hybrid E2E |

The logical layout contains no padding inside the four payload segments. Alignment padding is outside a layer's
payload and does not create another KVCM2 pool, slot, or buffer. It is copied as part of the opaque cold page but is
never consumed by decode. Encode zero-fills it in the existing general offload kernel, after the V-scale writer has
finished the final payload segment. The complete `coldPageBytes` image is therefore deterministic and safe for raw
Host-to-Disk serialization even when a mapped staging Slot was reused from another Page or lifecycle.

For one Attention layer, let `N = H * P * D`. The exact payload is `9N/8` bytes: `N/2` packed K, `N/2` packed V,
`N/16` K scales, and `N/16` V scales. Relative to two FP8 hot buffers (`2N` bytes), this is 43.75% smaller; relative
to two FP16/BF16 hot buffers (`4N` bytes), it is 71.875% smaller, before bounded record padding. Legal payload sizes
are even, so although the generic alignment bound is 15 bytes, the actual maximum padding for this geometry is 14
bytes. An observed 8-byte tail only means `9N/8 mod 16 == 8`; it does not impose another divisibility condition.

### Supported combinations

| Dimension | Supported | Rejected or deferred | Admission owner | Evidence |
|---|---|---|---|---|
| Backend / architecture | C++ KVCM2 on the admitted SM100 family | pure-Python KVCM2; any non-SM100-family target, including pre-SM100 and SM120 | Python/native factory | exact unit validation on one B200; no broader architecture claim |
| Runtime KV dtype | FP16, BF16, FP8 E4M3 | other types | compression manager/codec | all three native paths pass exact B200 tests; final E2E covers FP8 and BF16 |
| Cold levels | Host, Disk, or both through the common cold representation | custom tier with no GPU-accessible codec staging | KVCM2 contract | final campaign covers Host-only, Host+Disk, and Disk-only |
| Attention layout | conventional per-layer K and V buffers, `headDim % 16 == 0`, at most 128 local Attention layers per lifecycle | unrecognized or incomplete K/V; MLA layouts not matching this contract; larger per-rank layer counts | codec `configure()` / kernel plan | positive, negative, odd-page, and tail tests confirmed |
| Full/SWA/hybrid | full Attention and conventional-K/V Attention inside a hybrid lifecycle; non-attention lifecycles lossless | compressing SSM/conv state; independent SWA E2E not yet claimed | KVCM2 lifecycle + codec | native hybrid test plus single-rank Qwen3.5 campaign |
| TP/PP/attention-DP | local geometry is validated at construction | distributed runtime combinations are not claimed by this record | executor adapter | exact campaign is single-rank |
| Speculative target/draft | none in the initial support surface | speculative decoding is rejected by the config compatibility check | LLM args | Python unit tests |
| Overlap / CUDA Graph | no representation-specific path was added | runtime support or performance is not independently claimed | KVCM2 | outside this campaign |
| Block reuse | admitted by the compression config; logical token/page identity is unchanged | general accuracy or lossy-byte-equality claim | compression config | only a short same logical `(block, lifecycle)` activation round trip is exercised |

### Audited model scope

The following classification is based on the checked-in runtime routes and the model configs available on
2026-08-17. A layer-count or `headDim % 16` match alone is insufficient: the codec also requires distinct `key` and
`value` buffers in the lifecycle.

| Family/config evidence | Current status | Reason |
|---|---|---|
| Qwen3 (`Qwen3-8B`: 36 layers, head dimension 128) | in scope | conventional K/V Attention layout and valid NVFP4 block geometry |
| Qwen3.5 dense/MoE (`Qwen3.5-4B`, `35B-A3B`, `397B-A17B`: head dimension 256) | in scope for full-Attention layers | full-Attention K/V is compressed; linear-Attention/conv state is delegated byte-exactly to the default codec |
| DeepSeek-V4 Flash/Pro (`deepseek_v4`) | out of this initial codec | runtime uses MLA and the `SELFKONLY` latent-cache route, not two conventional K/V buffers |
| GLM-5.2 (`glm_moe_dsa`, `kv_lora_rank=512`) | out of this initial codec | DeepSeek-style MLA/DSA latent cache and sparse side-buffer layout do not satisfy the two-buffer contract |

Qwen3.5 Host and Disk migration are part of the intended hybrid path. The codec accepts an arbitrarily byte-aligned
staging base, so mixed-lifecycle suballocation does not depend on KVCM2 requesting a stronger alignment. Job 3718654
also forces the concrete SSM/conv plus Attention allocation through Host+Disk and Disk-only routes.
DeepSeek-V4 and GLM-5.2 can use NVFP4 for other independently supported tensors, but this does not turn their
single-latent MLA KV record into the four-segment K/V cold-page format defined here.

### Minimal configuration contract

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: auto                   # must resolve to FP16, BF16, or FP8 for the serving checkpoint
  host_cache_size: 17179869184  # optional; use 0 for Disk-only
  disk_cache_size: 68719476736  # optional
  disk_cache_path: /real/local/disk/kv-cache

kv_cache_compression_config:
  algorithm: quantization_for_boundary
  quant: nvfp4
  scale_checkpoint_path: /separate/modelopt/nvfp4-kv-scales
```

`scale_checkpoint_path` supplies calibration only; the validated harness uses a physically separate calibration
checkpoint. The public KV dtype must resolve to FP16, BF16, or FP8. If the serving checkpoint itself declares native
NVFP4 KV, `auto` may resolve to hot NVFP4; that is not a source representation for this boundary codec and construction
fails closed. Qwen3 explicitly requests FP8; Qwen3.5 requests `auto` and the constructed manager is required to report
BF16.
`tokensPerPage` is never configured here; it follows KVCM2's `tokens_per_block`.

## Decisions

### D-001: Treat the native PR #17512 storage contract as immutable

```text
Status: accepted
Choice: Port the NVFP4 consumer and its minimal high-level Python construction adapter. Do not transplant old native
  KVCM2 fixes or edit native KVCM2/StorageManager, the native KVCM2 binding, or pure-runtime KVCM2.
Rationale and evidence: The force-pushed PR already redesigned cold pages, tier mapping, routing, staging, and failure
  fencing. Mixing the old branch's fixes would create a third contract and obscure ownership.
Rejected alternatives and why: Rebase all six old commits (contains obsolete KVCM2 changes); patch remaining side-path
  bugs opportunistically (outside the requested boundary).
Owners/lifetimes affected: high-level Python creates the codec; native KVCM2 ownership remains unchanged.
Public interface or failure policy: unchanged `IKvCacheColdPageCodec` contract.
Performance-sensitive contract: existing batching and stream/event topology remain unchanged.
Intended source symbols: high-level `_create_kv_cache_manager` and `_create_kv_cache_manager_v2_impl`; no native or
  pure-runtime KVCM2 symbol.
Required tests/artifacts: path-filtered diff, build, Host/Disk E2E.
Compatibility/migration: unresolved provider bugs go to the compatibility ledger below.
```

### D-002: Keep one construction-time native codec handoff

```text
Status: accepted
Choice: A construction-only Python provider loads calibration and creates one native codec before the final target
  KVCM2 constructor consumes it. The provider is not retained afterward.
Rationale and evidence: Cold slot sizes are needed during KVCM2 construction and the migration path must not call Python.
Rejected alternatives and why: late setter/unregister (unsafe lifetime); Python callbacks (hot-path/GIL ownership); global
  registry (cross-manager lifetime ambiguity).
Owners/lifetimes affected: Python owns calibration metadata; native KVCM2 owns the transferred unique_ptr.
Public interface or failure policy: construction rejects unsupported backends, devices, layouts, or calibration.
Performance-sensitive contract: no Python on migration path.
Intended source symbols: `KvCacheCreator`, `_create_kv_cache_manager`, `KVCacheManagerV2.__init__`,
  `_create_kv_cache_manager_v2_impl`, `QuantizationCompression.create_cold_page_codec`, and
  `create_nvfp4_cold_page_codec`.
Required tests/artifacts: unchanged construction without compression; pure-Python fail-closed; only the final target
  receives the provider; codec-enabled failure does not retry GPU-only/lossless; every native construction receives a
  fresh codec before the manager consumes it.
Compatibility/migration: factory returns a fresh codec for each construction attempt.
```

### D-003: Use one common compact blob for every cold level

```text
Status: accepted
Choice: Encode only on hot-to-cold and decode only on cold-to-hot. Copy Host/Disk blobs raw between cold levels.
Rationale and evidence: This is the current PR #17512 representation and migration contract.
Rejected alternatives and why: per-tier formats (requires new mappings/conversions); Disk-only codec (duplicates policy);
  four physical cold pools (breaks the one-blob contract).
Owners/lifetimes affected: KVCM2 owns each cold slot; codec interprets bytes only at hot boundaries.
Public interface or failure policy: one fixed `queryColdPageBytes()` per lifecycle.
Performance-sensitive contract: no requantization during Host/Disk movement.
Intended source symbols: `Nvfp4ColdPageCodec::encode/decode/queryColdPageBytes`.
Required tests/artifacts: Host round trip and Disk migration with activation evidence.
Compatibility/migration: no Host-only admission restriction is added by the compression side.
```

### D-004: Keep only format-necessary geometry restrictions

```text
Status: accepted
Choice: Require positive geometry and `headDim % 16 == 0`; do not require token-count or scale-count multiples of four.
Rationale and evidence: NVFP4 produces one E4M3 block scale for each 16 adjacent dimension values. Existing byte-tail
  paths handle legal 8-byte packed tails and arbitrary scale-byte tails.
Rejected alternatives and why: `tokensPerPage % 4 == 0` and V-scale `% 4 == 0` are transfer-shape assumptions, not
  logical format constraints; padding inside segments wastes space and complicates decoding.
Owners/lifetimes affected: codec admission and immutable kernel plan only.
Public interface or failure policy: invalid geometry fails construction with a specific error.
Performance-sensitive contract: vector bodies plus existing exact tail paths; no new kernel.
Intended source symbols: codec size helpers and `prepareNvfp4BoundaryPlan` validation.
Required tests/artifacts: non-multiple token pages and non-vector scale tails.
Compatibility/migration: K/V scale arrays retain independent offsets.
```

### D-005: Retain bounded inter-layer alignment padding

```text
Status: accepted
Choice: Start each layer record at a 16-byte boundary and keep all four segments tightly packed inside the record.
Rationale and evidence: It preserves aligned bulk paths while bounding waste to at most 15 bytes per layer. The user
  explicitly allowed this padding.
Rejected alternatives and why: padding every segment (unnecessary waste); forcing all geometry to align every segment
  (unnecessary admission restriction); removing all record alignment in this port (changes an already tested kernel
  performance contract without need).
Owners/lifetimes affected: codec's fixed cold-page stride only.
Public interface or failure policy: padding has no semantic meaning; encode writes zero and decode ignores it.
Performance-sensitive contract: relative layer offsets remain 16-byte aligned. An aligned staging base uses the
  vector path; an unaligned staging base uses the same kernel's 8-byte or byte fallback.
Intended source symbols: `compactLayerBytes`, layer cold-offset construction, and `clearCompactRecordPadding`.
Required tests/artifacts: exact offsets, zero inter-layer/trailing gaps, deterministic full-Slot re-encode, and no
  out-of-bounds writes.
Compatibility/migration: cold raw copies include deterministic zero padding as opaque bytes.
```

### D-006: Compose the default lossless codec for non-attention lifecycles

```text
Status: accepted
Choice: Delegate complete unsupported lifecycles to PR #17512's default codec.
Rationale and evidence: Hybrid recurrent state remains byte-exact without teaching NVFP4 code its semantics.
Rejected alternatives and why: reject every hybrid model (needlessly narrow); add SSM kernels (scope and correctness
  risk); duplicate lossless copy code (maintenance and staging-boundary risk).
Owners/lifetimes affected: NVFP4 codec owns one composed default codec.
Public interface or failure policy: a lifecycle cannot partially mix configured K/V with unknown side buffers.
Performance-sensitive contract: default codec retains its upstream lossless copy implementation; outer codec batching
  remains lifecycle-keyed under D-009.
Intended source symbols: `Nvfp4ColdPageCodec::configure/encode/decode`.
Required tests/artifacts: lossless fallback byte test.
Compatibility/migration: future lifecycle kinds are safe but uncompressed until explicitly supported.
```

### D-007: Preserve the existing two-way FP8 conversion

```text
Status: accepted
Choice: Keep FP8 E4M3 as a hot runtime type and apply its source-domain scale separately from calibrated NVFP4 K/V
  scales in both directions.
Rationale and evidence: NVFP4 <=> FP8 is an explicit required path; deleting FP8 would regress the accepted branch.
Rejected alternatives and why: FP16/BF16-only port (feature regression); reuse one K/V/global scale for both formats
  (wrong domains).
Owners/lifetimes affected: calibration config and immutable kernel parameters.
Public interface or failure policy: native FP8 layer configs require finite positive K and V source scales. The current
  PyTorch provider supplies unit source scales for the FP8 runtime; FP16/BF16 admission does not validate unused FP8
  fields.
Performance-sensitive contract: existing fused boundary kernels only.
Intended source symbols: runtime-type dispatch and FP8 quantize/dequantize helpers.
Required tests/artifacts: FP8 K/V independent-scale round trip.
Compatibility/migration: none.
```

### D-008: Use one fixed linear transfer tile

```text
Status: accepted and validated on the exact code parent in B200 job 3720301
Choice: Process at most 2,048 consecutive eight-value half-groups per tile in the same three logical kernels.
Rationale and evidence: `headDim % 16 == 0` makes the total half-group count even. The fixed tile and its final tail are
  therefore also even and never split the one-scale-per-16-values unit. Packed values and scales are both linear in HND
  order, so crossing a token/head row needs no alternate layout or kernel.
Rejected alternatives and why: choose a whole-number-of-rows tile using GCD/round-up (introduces a row-shape constraint);
  reject rows whose compact staging exceeds 36 KiB (not a format limit); retain shape-specific direct kernels (duplicate
  code and launch specializations).
Owners/lifetimes affected: CTA-local shared staging only; the cold record and public task ABI are unchanged.
Public interface or failure policy: no new admission guard; the existing 32-bit compact-offset bound remains.
Performance-sensitive contract: at most 9,216 bytes dynamic shared memory per CTA; aligned, 8-byte, and byte transfers
  remain paths inside the same kernel.
Intended source symbols: `compressedTransferHalfGroups`, the three tiled boundary kernels.
Required tests/artifacts: tile boundary inside a row, very large legal `headDim` with a two-half-group tail, arbitrary
  cold-base alignment, and full FP16/BF16/FP8 B200 regression.
Compatibility/migration: none.
```

### D-009: Keep codec batching lifecycle-keyed

```text
Status: accepted and validated on the exact code parent in B200 job 3720301
Choice: Return each known lifecycle as its own codec batching representative; return the failure sentinel for an unknown
  lifecycle.
Rationale and evidence: KVCM's cold PoolGroup merge is a physical-allocation optimization. The codec already receives a
  lifecycle key, and correctness needs no cross-lifecycle batch. Identity representatives are the interface default and
  cover Attention plus generic lossless lifecycles uniformly.
Rejected alternatives and why: compare GPU base addresses, strides, every scale, dtype, offsets, and payload size to
  discover transform-equivalent lifecycles (about 60 lines of special equivalence logic for an optional batching
  optimization with no functional requirement).
Owners/lifetimes affected: migration queue batching only; cold PoolGroup construction and sizes are unchanged.
Public interface or failure policy: known lifecycle -> itself; unknown lifecycle -> `-1`.
Performance-sensitive contract: possible cross-lifecycle coalescing is deferred until measurements justify it.
Intended source symbols: `Nvfp4ColdPageCodec::getBatchingLayerGroupId`.
Required tests/artifacts: distinct known identities, unknown sentinel, Attention plus SSM/conv Host/Disk E2E.
Compatibility/migration: KVCM still groups equal-size cold lifecycles physically and routes every migration by lifecycle.
```

## Implemented batches

| Batch | Decision IDs | Intended symbols | Explicit non-changes | Verification |
|---|---|---|---|---|
| 1 | D-001..D-009 | NVFP4 kernel/codec/factory/binding and high-level final-target Python adapter | native C++ KVCM2/StorageManager, native KVCM2 binding, pure-runtime KVCM2, migration, mapping, events | completed; path/static audit PASS |
| 2 | D-002, D-004..D-009 | focused Python/C++/CUDA tests | upstream KVCM2 implementation | completed; exact B200 80 native + 41 Python PASS |
| 3 | D-003 | exact Qwen runner bound to the validated wheel | model/attention path | completed; four B200 jobs cover functional and Nsight Host/Disk matrices |
| 4 | all | this record, validation manifests, compatibility ledger | unrelated docs | source conformance complete; publication SHA/clean-tree match is verified after push and reported in the handoff |

## KVCM2 compatibility ledger

These entries are observations for alignment with the KVCM2 owner. They authorize no native KVCM2, StorageManager,
native-binding, or pure-runtime code change; the high-level consumer adapter is explicitly in scope.

### Disposition of the old PR #17512 bug ledger

| ID | Current `984183e` disposition | Consumer action |
|---|---|---|
| C0 | fixed: `storageManager.cpp` includes `cudaDriverWrapper.h`, so batched `TLLM_CU_CHECK(cuMemcpyBatchAsync)` compiles | do not port old include fix; B200 native build covers it |
| C1 | fixed on the batched main path: `FuncGuard` fences source/destination finish events and eviction rollback reschedules the source | use upstream transaction path; do not add consumer fencing |
| C2 | fixed by redesign: raw same-representation migration is selected before codec lookup; source and destination pool groups are resolved independently per level | use raw Host/Disk copies; Qwen3 `0->1->2->0` and Disk-only `0->1->0` are route evidence |
| C3 | generically open on single-page `_copyPageToTreeBlock`: the ready event is attached only after `copySlotData` returns | document for owner; no patch because this is not the batched main path and NVFP4 validates before launch, then drains already-submitted chunks on synchronous failure |
| C4 | fixed structurally: StorageManager owns the shared GPU physical allocator, levels borrow it, levels are destroyed before allocator reset | do not port old lifetime fix; current cold-page/allocator owner suites cover construction and teardown sequence |

### Consumer-specific compatibility items

| ID | Observation | NVFP4 main-path impact | Action in this branch |
|---|---|---|---|
| K-001 | C3's single-page false-return event window remains an owner side-path issue. | not on the normal batched migration path; current codec's failure behavior does not leave queued work unfenced | document only; do not patch KVCM2 |
| K-002 | Upstream has no separately named Host-to-Disk bypass test, although the raw branch is structurally ordered correctly. | no implementation change required | retain consumer-side Host+Disk route evidence |
| K-003 | Base has no high-level codec injection seam. | Late replacement is unsafe because cold sizes, staging, and pools are fixed during native construction. | resolved in the authorized consumer Python adapter: the final target passes a fresh codec into native construction; native/storage/runtime code remains unchanged, and codec-enabled failure is not silently retried losslessly |
| K-004 | Disk codec staging requests alignment `1`, so a codec cannot assume the cold base itself is 16-byte aligned. | resolved on the consumer side: the general tiled kernel retains aligned fast paths and adds an exact byte fallback; no KVCM2 alignment contract is required | base+1 BF16/FP8 unit test plus Qwen3.5 mixed-lifecycle Disk E2E; no owner change |

The adapter changes only final-target high-level construction. It creates a fresh codec immediately before each native
manager construction attempt and passes it as the existing third native argument. Plain lossless construction is
unchanged. Estimation, draft, and cross managers receive no provider. Pure-Python construction rejects compression,
and a failed codec-enabled native construction is not retried without the codec.

## Decision log

| Revision/date | Evidence or request | Old decision | Approved replacement | Affected code/tests/docs | Approval |
|---|---|---|---|---|---|
| r1 / 2026-08-17 | User requested a direct port to latest #17512 and prohibited KVCM2 edits. | old branch included provider-side bug fixes | latest #17512 is immutable; consumer-side adaptation only | whole port | user |
| r1 / 2026-08-17 | User requested a general design with no new special kernel. | possible geometry-specific restrictions | keep only format constraints and existing general tail paths | codec/kernel tests | user |
| r1 / 2026-08-17 | User explicitly retained FP8 and allowed bounded padding. | possible FP8 deletion or fully tight records | retain FP8; retain only per-layer 16-byte alignment | codec/kernel/docs | user |
| r2 / 2026-08-17 | Final boundary clarified to include the high-level KVCM2 wrapper. | temporary local provider bridge | remove bridge; require the owner factory seam in K-003 | Python integration/docs | user |
| r2 / 2026-08-17 | Code-slimming audit found shape-specific direct/tiled dispatch and four descriptor-capacity instantiations. | six logical paths and 48 instantiations | validate one bounded tiled path and fixed 256-descriptor chunks in an isolated B200 experiment | CUDA kernel/tests | user request for general design |
| r3 / 2026-08-17 | Disk staging has no base-alignment guarantee. | require a 16-byte cold base | retain relative record alignment but support arbitrary staging-base alignment in the same tiled kernel | CUDA kernel/tests/docs | compression-side adaptation boundary |
| r4 / 2026-08-17 | Row-shaped transfer sizing imposed a 36 KiB shared-memory guard unrelated to the compact format. | select a whole-row tile using GCD/round-up | use one 2,048-half-group linear tile; retain only `headDim % 16` and the 32-bit ABI bound | CUDA kernel/tests/docs | user request for only format-necessary restrictions |
| r4 / 2026-08-17 | Cross-lifecycle codec equivalence was optional and duplicated plan comparison logic. | merge calls when every physical field and scale matches | keep codec batching lifecycle-keyed; leave physical cold PoolGroup merging to KVCM | codec/tests/docs | general minimal design |
| r5 / 2026-08-17 | User clarified that the no-KVCM2-change boundary applies to native C++/StorageManager/runtime internals; manual high-level Python wiring is allowed. | wait for an owner hook and remove the bridge | retain the direct final-target provider handoff; require prohibited native/runtime paths to stay zero diff | Python adapter/tests/docs | user |
| r6 / 2026-08-17 | Raw Host-to-Disk copies serialize the complete aligned cold Slot, while staging and physical pools may be reused. | leave padding uninitialized because decode ignores it | zero only the bounded record tail inside the existing general encode kernel; keep KVCM2/StorageManager and the compact four-segment payload unchanged | kernel/tests/docs | data-hygiene audit |
| r6 / 2026-08-17 | FP16/BF16 transforms never consume source-FP8 calibration. | validate all four scale arrays for every runtime dtype | validate NVFP4 scales for every dtype and FP8 scales only for FP8 runtime KV | codec/test/docs | source-slimming audit |
| r7 / 2026-08-17 | Exact-wheel provenance and E2E evidence supersede preliminary-tree validation. | mix preliminary and authoritative results | bind final claims to code parent `2970613` and immutable wheel SHA, keep lossy token agreement diagnostic, and retain old runs only as historical evidence | validation/docs only | conformance audit |

## Conformance audit

### Design to implementation

| Decision/invariant | Implemented symbol | Verification | Result |
|---|---|---|---|
| D-001 / I-001 | no native/storage/runtime KVCM2 implementation symbol | path-filtered final diff | prohibited native/binding/pure-runtime paths pass; authorized high-level adapter only |
| D-002 | codec factory and final-target construction adapter | Python/native ownership tests | direct provider handoff and fail-closed tests PASS on exact code parent |
| D-003 / I-002 / I-003 | existing KVCM2 migration contract + codec boundary calls | Host/Disk E2E | functional jobs 3721270/3721272 and Nsight jobs 3721683/3721681 PASS |
| D-004 / D-008 / I-004..I-007 / I-012 / I-014 | compact layout, fixed linear tile, and plan validation | byte/layout/cross-row/large-dimension tests | exact B200 native suite PASS |
| D-005 / I-011 | layer offset construction and padding zero-fill | offset, canary, and deterministic full-Slot tests | exact B200 native suite PASS |
| D-006 / I-010 | composed default codec | lossless fallback test | exact B200 native suite PASS; Qwen3.5 Attention/SSM allocation signature remains compressed/lossless respectively |
| D-009 / I-015 | lifecycle-keyed codec batching | identity/sentinel tests | exact B200 native suite PASS |
| D-007 / I-008 | FP8 runtime dispatch and dtype-specific scales | FP16/BF16 admission and FP8 round-trip tests | exact B200 native suite PASS |
| I-009 | encode/decode success-path enqueue implementation | native tests plus final Nsight attribution | native suite and all six trace proof sets PASS; synchronous launch-failure drain remains the documented exception |

### Implementation to design

The exit criterion is zero unmapped material guards, fallbacks, persistent state fields, native calls, or support
restrictions.

| Implementation item | Authorizing decision | Necessary? | Result |
|---|---|---|---|
| native codec and layer config | D-002, D-003 | yes | present and tested |
| three general NVFP4 boundary kernels | D-004, D-005, D-007, D-008 | yes | direct/shape-specific and row-shaped paths removed; exact B200 suite PASS |
| default lossless codec member | D-006 | yes | present and tested |
| CMake target and source wiring | D-002, D-004 | yes | one codec target and one general kernel source family |
| compression nanobind factory and registration | D-002 | yes | existing codec ownership ABI; wheel binding verified |
| public config union and API/telemetry golden | D-002 | yes | discriminator/config tests and source-only golden regeneration PASS |
| final-target provider selection and consumption | D-001, D-002 | yes | `_util.py` selects only the target; high-level wrapper consumes before native construction |
| codec-enabled fail-closed construction | D-002 | yes | no pure-Python or silent lossless/GPU-only retry |
| record-tail zero-fill | D-005 / I-011 | yes | existing V-role encode CTA writes bounded inter/trailing padding; deterministic Slot tests PASS |
| 128-layer descriptor cap and 32-bit compact offsets | I-013, D-008 | yes | explicit ABI/admission guards with native negative coverage |
| focused fixtures and regression tests | D-002, D-004..D-009 | yes | 80 native and 41 Python tests on exact code parent |
| cross-lifecycle codec-equivalence map | none | no | removed; physical cold PoolGroup sharing remains upstream-owned |
| SM100 and dtype admission | support matrix, D-007 | yes | present and tested |
| Python control-plane retention after native handoff | D-002 | no | absent |
| any Host-only or token/scale divisibility guard | none | no | absent by source search and odd-page tests |
| any native/storage/runtime KVCM2 change | none | no | absent by path-filtered diff; high-level adapter is authorized by D-001/D-002 |

## Deviations

The high-level Python `cold_page_codec_provider` handoff is an accepted consumer adapter, not a deviation. Prohibited
native C++ KVCM2/StorageManager, native-binding, and pure-runtime paths remain unchanged. K-004 is resolved entirely in
the compression-owned transfer helpers and does not require an owner change.

## Validation and handoff

### Authoritative exact code-parent build and unit validation

The runtime evidence is bound to clean code parent
`297061375c923b0a4d9f6c8b82af2f9c19dccad3`, not to the documentation-only publication commit.

| Evidence | Result |
|---|---|
| Exact source contract | 10,714 tracked entries: 6,610 non-LFS files, 4,102 materialized LFS files, and two symlinks |
| Incremental B200 build | job 3719753 on `umb-b200-237`; Release, `100-real`, `BUILD_PYT=ON`, `BUILD_TESTS=ON`; 640 initial Ninja actions plus 28 canonical staging actions completed with no compiler/linker error |
| Build-harness disposition | r18b later failed with exit 134 because the harness loaded staged and build-tree `libth_common.so` copies into one process and registered the same Torch op twice; this is not recorded as a build/test PASS |
| Immutable test continuation | job 3720301 on `umb-b200-237`, PASS, exit 0 at `2026-08-17T16:40:58Z`; it reused the completed build and did not invoke CMake, Ninja, or wheel packaging |
| Native tests | 80/80: 38 boundary kernels, 13 NVFP4 codec, five default codec, 11 cold-page, 11 staging, and two slot-allocator tests |
| Python tests | 41/41: 20 boundary/control-plane, two compression-config, and 19 telemetry/golden-manifest tests |
| Immutability and imports | 30,963 build-tree file hashes identical before/after; exact source postflight, fresh source import/native maps/schema smoke, and generated-versus-committed telemetry manifest all PASS |
| Build artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-build-20260817/runs/20260817T160000Z-2970613-b200-exact-r18b` |
| Test artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-test-continuation-20260817/runs/20260817T164000Z-2970613-b200-test-only-r18d` |

### Final wheel provenance

The exact wheel is packaged from the staged native products above. Acceptance requires four-way equality for
`libtensorrt_llm.so`, `libth_common.so`, and the Python binding across build, staged source, wheel ZIP, and fresh
installation; three-way equality for the five changed Python modules; an isolated installed-wheel import with all
TensorRT-LLM native mappings under that installation; and an unchanged build/source tree.

The changed `tensorrt_llm/usage/llm_args_golden_manifest.json` is intentionally source-only. Its source content differs
between base and head, while the unchanged `setup.py` package-data rule in both revisions packages only
`usage/schemas/*.json`; runtime telemetry builds its capture manifest from the validated config types. The r18d
telemetry suite proves the generated manifest matches the committed head source. Adding this docs/privacy golden to
`setup.py` would be an unrelated packaging-contract change.

| Evidence | Result |
|---|---|
| Accepted wheel job | job 3721144 on `umb-b200-239`, PASS, exit 0 at `2026-08-17T17:38:35Z`; r22b accepted the unchanged wheel bytes produced by r19 |
| Wheel | `tensorrt_llm-1.3.0rc25-cp312-cp312-linux_x86_64.whl`, 704,361,052 bytes, SHA-256 `ac7f11d7c5166af28f58c355a40aaee3a810c59999951b39e640efce390cd142` |
| Native provenance | `libtensorrt_llm.so`, `libth_common.so`, and the Python binding have identical hash and size in build, staged source, wheel ZIP, and fresh install |
| Python provenance | the five changed packaged Python modules have identical hash and size in staged source, wheel ZIP, and fresh install |
| Golden contract | source SHA-256 `bb8ed6c3427e29794f165653ff3202f1011b783edda0f0d8ef3098be3ca273fd`; absent from wheel/install by the unchanged base package-data rule; r18d regeneration test PASS |
| Installed import | `python -I` loaded the package, bindings, and NVFP4 factory types from `install-r22-2970613`; every TensorRT-LLM native mapping came from that install |
| Immutability | seven selected build products, the wheel inode/mtime/hash/size, and complete staged/candidate source manifests were identical before and after validation |
| Wheel artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-wheel-continuation-20260817/runs/20260817T175000Z-2970613-b200-wheel-continuation-r22b` |
| Wheel directory | `/home/scratch.tianruih_coreai/formal-wheel-kvcc-pr17512-latest-20260817/20260817T102500Z-c37cf05-b200-incremental-r1/dist-r19-2970613` |

### Final exact-wheel functional validation

Both models ran from fresh subprocesses against code parent `2970613` and immutable wheel SHA-256
`ac7f11d7c5166af28f58c355a40aaee3a810c59999951b39e640efce390cd142` on one B200 each. The common workload used the
C++ backend, overlap scheduling, 2,048 hot-KV tokens, 32 tokens per block, a 535-token prompt, 32 output tokens, batch
size one, and seed 888. The functional event ledger proves logical KVCM level transitions; Disk syscall attribution is
reserved for the Nsight evidence below.

| Model | Job / terminal UTC | Runtime and matrix | Immutable postflight | Artifact basename |
|---|---|---|---|---|
| Qwen3-8B | 3721270 / `2026-08-17T17:54:20.800149Z`, PASS, exit 0 | FP8; 7/7 arms; matrix SHA-256 `28fe20671bf2974b805a6c79a3c2059b24bf2adf98b4b1a63ca6490eab090e81` | source HEAD/status, sealed suite `a61c21ac...`, and wheel binding identical before/after | `20260817T174051Z-qwen3-functional-slurm3721270` |
| Qwen3.5-4B | 3721272 / `2026-08-17T17:56:48.376387Z`, PASS, exit 0 | `auto` -> BF16; 7/7 arms; matrix SHA-256 `f5d5b757e5a1a4f7ce4d91b3c03ca715d394c95c2d324561ccfd775c7af25ad0` | same source/suite/wheel checks; Attention NVFP4 and SSM/conv lossless | `20260817T174051Z-qwen35-functional-slurm3721272` |

The route strings below are contract expectations. The fill/replay transition counts are separately observed events;
the Disk-only topology maps logical level 1 directly to Disk and sets Host quota to zero.

| Model / topology | Contract route and observed fill/replay transitions | Logical round trip and partial page | Raw -> NVFP4 cold Slot | Comparison evidence |
|---|---|---|---|---|
| Qwen3 FP8 / Host | `0->1->0`; fill `0->1 x478`; replay `0->1 x18, 1->0 x17` | 17 Attention `(block,lifecycle)` identities; partial ordinal 16, stored 23/matched 22 tokens | 2,359,296 -> 1,327,104 B (-43.75%) | GPU-vs-raw 32/32 exact gate; raw-vs-NVFP4 32/32 diagnostic |
| Qwen3 FP8 / Host+Disk | `0->1->2->0`; fill `0->1 x478, 1->2 x276`; replay `0->1 x18, 1->2 x18, 2->0 x17` | same 17 Attention identities and partial-page evidence | both Host and Disk: 2,359,296 -> 1,327,104 B (-43.75%) | same gate/diagnostic result |
| Qwen3 FP8 / Disk-only | `0->1->0`; fill `0->1 x478`; replay `0->1 x18, 1->0 x17` | same 17 Attention identities and partial-page evidence | Disk: 2,359,296 -> 1,327,104 B (-43.75%) | same gate/diagnostic result |
| Qwen3.5 BF16 / Host | `0->1->0`; fill `0->1 x594` (lifecycle 0/1: 46/548); replay `0->1 x20, 1->0 x18` (2+18, 1+17) | 18 identities total, of which 17 are Attention; Attention partial ordinal 16, stored/matched 23 tokens | Attention 1,048,576 -> 294,912 B (-71.875%); SSM/conv 26,345,472 B unchanged | GPU-vs-raw 32/32 exact gate; raw-vs-NVFP4 32/32 diagnostic |
| Qwen3.5 BF16 / Host+Disk | `0->1->2->0`; fill `0->1 x594, 1->2 x412` (46+548, 38+374); replay `0->1 x20, 1->2 x20, 2->0 x18` (2+18, 2+18, 1+17) | same 18 identities and 17 Attention round trips; same partial page | both tiers: Attention 1,048,576 -> 294,912 B; SSM/conv 26,345,472 B unchanged | same gate/diagnostic result |
| Qwen3.5 BF16 / Disk-only | `0->1->0`; fill `0->1 x594` (46+548); replay `0->1 x20, 1->0 x18` (2+18, 1+17) | same 18 identities and 17 Attention round trips; same partial page | Disk: Attention 1,048,576 -> 294,912 B; SSM/conv 26,345,472 B unchanged | same gate/diagnostic result |

The matched-arm PASS gate proves identical workloads and lossless raw preservation. The observed raw-versus-NVFP4
32/32 agreement is a diagnostic for this short deterministic smoke only; it is neither an activation gate nor a
general quality claim. For Qwen3.5, the cold-size signature confirms that only the Attention cold PoolGroup changes
representation across levels. Separately, the run observes one SSM snapshot hit and 535 reused tokens. Public events
do not expose the SSM snapshot-byte transition, so the lifecycle-0 logical record must not be described as proof that
an SSM page traversed Host or Disk. Both Disk runs used ext4 backed by `/dev/md0[/tmp/computelab-tmp]`, not tmpfs.

### Final exact-wheel Nsight validation

The profiler campaign used the same code parent, wheel, models, and activation workload as the functional campaign.
It makes no timing or speedup claim. Instead, it proves that the general NVFP4 boundary kernels execute under the
matching codec and KVCM migration ranges, and that the existing Disk path performs writes during forced offload and
reads during forced onboard.

| Model | Job / terminal UTC | Runtime and matrix | Immutable postflight | Artifact basename |
|---|---|---|---|---|
| Qwen3-8B | 3721683 / `2026-08-17T18:09:27.096502Z`, PASS, exit 0 | FP8; 3/3 NVFP4 arms; matrix SHA-256 `7bc940cf8c4acd707adae1518c57392965c4457a26504c931059f34ded53f432` | source HEAD/status, sealed suite, wheel binding, and full inner/artifact manifests PASS | `20260817T175814Z-qwen3-nsys-slurm3721683` |
| Qwen3.5-4B | 3721681 / `2026-08-17T18:13:38.744136Z`, PASS, exit 0 | BF16; 3/3 NVFP4 arms; matrix SHA-256 `0d1b7a7da77785a4540776d0a2c766fe247064c2b506e1deec233f86e7d05db2` | same checks; 14,855 inner and 14,865 final artifact entries verified | `20260817T175814Z-qwen35-nsys-slurm3721681` |

In the launch column, each histogram maps pages per launch to launch count. Codec-associated totals cover every boundary
launch; primary counts include only launches nested in the explicit forced phase. Each whole-arm Qwen3 trace contains
two offload codec launches outside that phase, and each Qwen3.5 trace contains three; no cause is attributed here. Each
arm contains one forced offload and one forced onboard phase.

| Model / arm | Observed route / logical identities | Boundary launches | Codec-associated total / inside primary | Disk syscall evidence | Proof result |
|---|---|---|---|---|---|
| Qwen3 FP8 / Host | GPU->Host->GPU / 17 Attention | `offloadFromFp8TiledKernel`: 60 `{1:31,16:28,17:1}`; `onboardTiledKernel<__nv_fp8_e4m3>`: 1 `{17:1}`; 36 layers, grid.y 72 | offload `60 total; 58 primary`; onboard `1; 1` | N/A: topology has no Disk tier | all three proofs PASS |
| Qwen3 FP8 / Host+Disk | GPU->Host->Disk->GPU / 17 Attention | same FP8 families: offload 60 `{1:31,16:28,17:1}`; onboard 1 `{17:1}`; 36 layers | `60; 58`; `1; 1` | `pwrite=276`, `pread=17` in matching forced ranges | all three proofs PASS |
| Qwen3 FP8 / Disk-only | GPU->Disk->GPU / 17 Attention | same FP8 families: offload 69 `{1:31,2:1,3:1,4:1,5:1,6:2,7:2,8:2,9:2,10:2,11:1,12:1,13:1,14:1,16:19,17:1}`; onboard 2 `{5:1,12:1}`; 36 layers | `69; 67`; `2; 2` | `pwrite=478`, `pread=17` in matching forced ranges | all three proofs PASS |
| Qwen3.5 BF16 / Host | GPU->Host->GPU / 17 Attention identities; 18 total with one lifecycle-0 logical record, no snapshot-byte tier attribution | `offloadFrom16BitTiledKernel`: 102 `{1:68,2:1,15:32,16:1}`; `onboardTiledKernel<__nv_bfloat16>`: 1 `{17:1}`; 8 layers, grid.y 16 | `102 total; 99 primary`; `1; 1` | N/A: topology has no Disk tier | all three proofs PASS |
| Qwen3.5 BF16 / Host+Disk | GPU->Host->Disk->GPU / same 17 Attention and 18 total logical records | same BF16 families, 102/1 launches and histograms | `102; 99`; `1; 1` | target-process totals `pwrite=412`, `pread=18`; no per-lifecycle byte attribution | all three proofs PASS |
| Qwen3.5 BF16 / Disk-only | GPU->Disk->GPU / same 17 Attention and 18 total logical records | same BF16 families, 102/1 launches and histograms | `102; 99`; `1; 1` | target-process totals `pwrite=594`, `pread=18`; no per-lifecycle byte attribution | all three proofs PASS |

Every boundary launch is associated with its codec NVTX range, and every direction has at least one launch in its
primary forced phase. SQLite integrity, target-process trace-health, partial-page evidence, whole-page attribution,
and cold-migration checks pass for all six traces. The Qwen3.5 boundary launches cover the eight Attention layers
only; the target-process, forced-phase Disk syscall totals are not attributed to Attention, SSM, or codec bytes
individually. Ignored
non-target Nsight diagnostics are not a global trace-completeness claim.

| Model / arm | `.nsys-rep` SHA-256 | SQLite SHA-256 |
|---|---|---|
| Qwen3 / Host | `1be670f1bd1d3111c49cf9b0309f05faf626e426a57eba1b6896c2c7e8a65aed` | `3c5db741477c18d7e675aaec11b2d1f9ee070f817394df5d30aefa262046b073` |
| Qwen3 / Host+Disk | `060f57d00df88e716bc84b5e6530b4240f624d011016f8e84f468184b6c48cc2` | `a9777adb6e33b5fd5f50d9aaeb453b63e97823b361eef90e23d9408e3a120f53` |
| Qwen3 / Disk-only | `68525da05bf07340bbe134de1279784d506364ed298985fabbfd0ef2f9f302d0` | `781b5262bd451b9584f6ed980d6d3b9f6a121e44f0005b13356a67285eb27234` |
| Qwen3.5 / Host | `ef13bc0689f08b1cb8d06e82043a6c856a0901db8afb5c5eb1c1f2dbab11b276` | `83407349d23b6704c865b476ac213eb669c38a9cd8ba364123701f48b7462b2d` |
| Qwen3.5 / Host+Disk | `c9b1d8b232bffe67e4f21945ab2a7156f1ef6b5a9f0f0b6991644a7f7ad0acfa` | `d48bcdc1426d57025a755367c437279b7cfc776397823bd1301a7b5d4dfe74f7` |
| Qwen3.5 / Disk-only | `3a032cdaf2197d21f55ab2050e15e7b4160405a21a38b3ab9ae28280a54d0e59` | `ce6dd71fcc5206c7e29d71b060e445bc43bab47e7e58de627a22354b5c65a3d9` |

### Historical mechanism and preliminary E2E evidence (`dd97da0`)

| Evidence | Result |
|---|---|
| B200 incremental validation | job 3718073, PASS; Release, `100-real`, `BUILD_PYT=ON`, `BUILD_TESTS=ON`, NVTX enabled |
| Native tests | 84/84: 40 boundary, 3 partial-tail, 12 NVFP4 codec, 5 default codec, 11 cold-page owner, 11 staging, 2 allocator |
| Python tests | 41/41: 20 boundary control-plane, 2 config, 19 telemetry/manifest |
| Validation artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-build-20260817/runs/20260817T124100Z-dd97da0-b200-validation-r14` |
| Installed wheel | `tensorrt_llm-1.3.0rc25-cp312-cp312-linux_x86_64.whl`, SHA-256 `25c6b260e6a49ee6ab340c74869d07d2beab9b5f43e47ec8bf6f0c47c4a3bd7f` |
| Qwen3 functional | job 3718256, seven of seven arms PASS: GPU-only; raw/NVFP4 Host-only, Host+Disk, and Disk-only |
| Qwen3 Disk routes | Host+Disk `0->1->2->0`; Disk-only `0->1->0`; 17 Attention pages completed same-block/lifecycle round trips |
| Qwen3 storage | raw FP8 slot 2,359,296 bytes; NVFP4 slot 1,327,104 bytes, exactly the expected 43.75% reduction |
| Qwen3 replay | GPU vs raw 32/32 tokens exact; raw vs NVFP4 32/32 exact on this short deterministic activation workload only |
| Functional artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-e2e-results-20260817/20260817T130242Z-qwen3-functional-slurm3718256` |
| Qwen3.5 functional | job 3718654, seven of seven arms PASS; three matched-arm artifacts pass the raw-preservation and workload-identity gate; raw-versus-NVFP4 token agreement is recorded only as a diagnostic; `auto` resolved to BF16 |
| Qwen3.5 mixed lifecycle | SSM/conv allocation stayed lossless; 17 Attention identities and one lifecycle-0 logical record completed the round trip, while SSM snapshot reuse was observed separately; this does not attribute SSM snapshot bytes to a tier transition |
| Qwen3.5 Disk routes | Host+Disk `0->1->2->0`; Disk-only `0->1->0`; real `/dev/md0` ext4 backing |
| Qwen3.5 storage | raw BF16 Attention slot 1,048,576 bytes; NVFP4 slot 294,912 bytes, the expected 71.875% reduction; SSM/conv slot unchanged |
| Qwen3.5 artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-e2e-results-20260817/20260817T134840Z-qwen35-functional-slurm3718654` |
| Qwen3 Nsight | job 3718655 completed the workload and captured 60 FP8 offload launches plus one 17-page FP8 onboard launch in the correct nested NVTX ranges; the overall attempt is verifier-only FAIL because its name filter expected a nonexistent symbol |

This is a preliminary validation-only incremental build namespace, not the authoritative exact-wheel campaign. Its
Qwen3 result demonstrated activation, route correctness, reuse, and Disk residency on a real ext4 block device. The
short same logical `(block, lifecycle)` replay is diagnostic and is not a general accuracy claim. The old functional
run also does not by itself prove kernel/NVTX ranges or `pwrite`/`pread` nesting.

### Incomplete/failed harness attempts

| Attempt | Disposition |
|---|---|
| job 3718172 | recursive Git-alternate preflight failure before any arm; harness-only |
| job 3718260 | Qwen3.5 explicit `bfloat16` rejected by its loader before model/KVCM construction; config-contract finding, not codec failure |
| job 3718357 | source-dirty preflight failure before profiler launch; harness-only |
| jobs 3718478/3718479 | sealed r6 normalized nanobind dtype enum values to strings `7`/`6`; failed the evidence gate after manager construction and before any usable functional/profile workload |
| job 3718655 | r7 expected an invented `onboardToFp8` symbol; the trace contains `onboardKernel<32, __nv_fp8_e4m3>` with the correct task ABI, grid, codec range, and outer onboard range. This is a verifier-only failure; rerun uses task-ABI and template-dtype matching. |
| job 3719753 | exact compile and canonical native staging completed, then the r18b import harness double-loaded `libth_common.so` and duplicated Torch-op registration; immutable r18d reran the tests correctly |
| job 3720154 | r18c passed all native tests and two Python groups; telemetry collection stopped because the base image lacked the dev-only `mako` dependency; r18d added the existing dependency root and reran the entire suite |

r19 produced the immutable wheel, but its verifier misclassified a source-only telemetry golden as package data.
Append-only r20/r21/r22 attempts repaired verifier and provenance-mounting logic only; they did not rebuild or change
the wheel. r22b job 3721144 accepted the unchanged r19 bytes.

The corrected Nsight acceptance requires all three NVFP4 storage arms, fused boundary kernels inside the matching
offload/onboard NVTX ranges, and Disk `pwrite` during forced offload plus `pread` during forced onboard. Kernel
classification is based on `Nvfp4BoundaryOffloadPageTask` / `Nvfp4BoundaryOnboardPageTask`; dtype evidence comes from
the offload family and onboard template argument rather than a shape-specific symbol name.

### Evidence limitations

- Exact E2E evidence is single-rank and one B200 per invocation. It does not claim TP, PP, attention-DP, independent
  SWA, speculative decoding, or CUDA Graph/overlap performance coverage.
- Qwen3 exercises FP8 hot KV and Qwen3.5 exercises BF16 hot KV. FP16 is covered by native boundary/codec tests, not a
  model E2E in this campaign.
- The activation workload validates construction, route transitions, logical reuse, partial pages, raw preservation,
  codec kernel attribution, and storage syscalls. It is not a general model-accuracy or quality evaluation.
- No latency, throughput, or compression-speed claim is made. The build is an exact incremental B200 build with
  immutable artifact provenance, not a clean full rebuild.
- MLA/`SELFKONLY` layouts, including the audited DeepSeek-V4 and GLM-5.2 configs, remain outside the two-buffer codec
  contract.

### Publication audit and exit criteria

| Item | Result |
|---|---|
| Stacked base | remote PR #17512 head rechecked as `984183e480cd7e23f79d73005fb395d923803b28` on 2026-08-17 |
| Validated code parent | `297061375c923b0a4d9f6c8b82af2f9c19dccad3`; 26 files, +5,534/-44 against the stacked base |
| Publication diff | documentation-only child of the validated parent; 26 files, +5,704/-44 against the stacked base |
| Kernel surface | exactly three `__global__` boundary kernels: 16-bit offload, FP8 offload, and common onboard |
| Owner boundary | zero diff in native C++ KVCM2/StorageManager, native KVCM2 binding, and pure-runtime KVCM2 paths |
| Removed restrictions | no token/page `%4`, scale `%4`, V-scale placement `%4`, half-group `%8`, Host-only, row-GCD, or 36-KiB geometry admission |
| Retained limits | positive geometry, `headDim % 16 == 0`, finite positive scales used by the runtime dtype, SM100-family, 128 local Attention layers, and 32-bit compact-offset ABI |

- The authorized final-target Python provider bridge is retained and covered by focused tests; native C++
  KVCM2/StorageManager, the native KVCM2 binding, and pure-runtime KVCM2 remain zero diff.
- Arbitrary cold-base alignment and mixed-lifecycle Disk support are covered by exact-tree unit and Qwen3.5 E2E.
- The general-kernel implementation passes the authoritative 80-test native suite, 41 Python tests, exact wheel
  acceptance, two functional matrices, and two Nsight matrices.
- The publication commit may differ from the validated parent only in this record. Any code change requires a fresh
  exact-tree build/run. After push, the branch SHA, clean tree, and compare base are verified in the handoff.
