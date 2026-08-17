<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVFP4 Cold-Page Codec Design Record — 2026-08-17

## Record metadata

```text
Title: NVFP4 cold-page codec on the current KVCM2 cold-page contract
Record ID / revision: NVFP4-KVCM2-2026-08-17 / r4
Status: mechanism validated on B200; zero-KVCM publish integration blocked on K-003
Date / owner: 2026-08-17 / KV-cache compression side
Repository / branch / HEAD / dirty-diff hash:
  Hudayday/TensorRT-LLM
  nvfp4-pr17512-latest-e2e-clean-20260817
  base 984183e480cd7e23f79d73005fb395d923803b28
  mechanism-evidence HEAD dd97da02f881824d78baee119fbb5b2e8dfa424b
Stacked-PR base: NVIDIA/TensorRT-LLM PR #17512 at 984183e480cd7e23f79d73005fb395d923803b28
Reviewers or approval source: explicit user decision on 2026-08-17
```

The approval covers the scope boundary and the existing NVFP4 algorithm. It does not approve a change to KVCM2
ownership, migration, tier mapping, stream/event behavior, or public cold-page ABI.

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
- add no new representation-specific KVCM2 path and no additional CUDA kernel family;
- keep the final diff under native `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`, pure-runtime
  `tensorrt_llm/runtime/kv_cache_manager_v2/`, and high-level
  `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` empty; and
- document, rather than patch, any KVCM2 incompatibility discovered during the port.

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
- Making a performance claim in this port. Correctness, activation, and storage-path evidence are required.

### Current and intended flow

```text
TorchLlmArgs.kv_cache_compression_config
  -> KvCacheCreator selects boundary quantization only for the final target KVCM2
  -> QuantizationCompression loads ModelOpt scalar K/V calibration
  -> compression-side mixin overrides an owner-provided _create_backend_impl(config) hook
  -> native factory creates unique_ptr<Nvfp4ColdPageCodec>
  -> existing KVCM2 Python binding consumes the codec in KVCacheManager construction
  -> existing StorageManager calls codec.configure(all hot PoolGroupDesc)
  -> codec derives immutable per-lifecycle plans and fixed cold-page sizes
  -> existing KVCM2 builds its cold lifecycle <-> pool-group mapping
  -> existing migration engine invokes encode/decode only across hot/cold boundaries
  -> attention consumes the restored hot representation with no metadata change
```

`QuantizationCompression` is a construction-only provider for algorithm selection, calibration, and layer metadata.
It need not survive native construction. The native codec owns the immutable lowered plans after construction. KVCM2
owns all storage and migration state.

The `_create_backend_impl(config)` step is the only missing owner seam in base `984183e`. The validated
`dd97da0` mechanism tree used a temporary 57-line high-level wrapper bridge to prove the remainder of the pipeline.
That bridge is not eligible for the final branch under the user-requested ownership boundary. No global monkeypatch,
double-KVCM bootstrap, copied constructor, late `impl` replacement, or native registry is an accepted substitute.

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
| E-008 | `MEASUREMENT` | The `dd97da0` mechanism tree builds for `100-real` and passes 84 native plus 41 Python tests. | B200 job 3718073; validation artifact below | confirmed for the proof bridge tree |
| E-009 | `MEASUREMENT` | Qwen3 reaches Host, Host-to-Disk, direct Disk, and back to GPU with the expected same-page lifecycle routes. | B200 job 3718256; seven-arm artifact below | confirmed for mechanism/route activation |
| E-010 | `FACT` | Base `984183e` has no pre-construction backend factory: its high-level wrapper directly invokes `KVCacheManagerPy` twice, while the native codec is consumed before cold sizing/allocation. | wrapper lines 1073-1086; binding lines 2155-2196; StorageManager lines 355-459 | K-003 publish blocker |
| E-011 | `MEASUREMENT` | Qwen3.5 keeps SSM/conv lossless while 17 Attention pages complete Host, Host+Disk, and Disk-only round trips. | B200 job 3718654; seven-arm artifact below | confirmed for the mechanism tree |
| E-012 | `MEASUREMENT` | Qwen3 FP8 offload and onboard kernels are both present inside their codec and outer migration NVTX ranges. | B200 job 3718655; SQLite trace evidence below | trace captured; r7 post-processor name filter failed |
| E-013 | `FACT` | One fixed 2,048-half-group tile covers every admitted geometry. It uses at most 8 KiB of packed staging plus 1 KiB of scale staging and may cross a token/head row without splitting a 16-value scale group. | general boundary-kernel source and cross-row/large-`headDim` tests | static review complete; exact-tree B200 validation pending |

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
| CUDA stream and completion event | KVCM2 | migration transaction | migration engine | codec and slot/page reuse | codec enqueues only; KVCM2 fences |
| Cold Host/Disk blob | KVCM2 | cold slot | codec or raw copy | raw copy or codec | opaque to every KVCM2 consumer |
| Attention hot KV | KVCM2/model runtime | hot slot | model write or codec decode | attention | unchanged runtime dtype/layout |

## Invariants and support matrix

### Invariants

| ID | Invariant | Authoritative owner | Failure mode | Verification |
|---|---|---|---|---|
| I-001 | No final diff under native, pure-runtime, or high-level PyExecutor KVCM2 paths. | port boundary | review failure | path-filtered diff |
| I-002 | One cold `SlotId` selects one physical cold pool and one opaque logical blob at every cold level. | KVCM2 PR #17512 | incompatible construction | existing KVCM2 tests plus E2E |
| I-003 | Host and Disk use byte-identical cold payloads; cold-to-cold migration does not invoke the codec. | KVCM2 | corruption or needless requantization | Disk E2E and logs/counters |
| I-004 | For layer element count `N = H * P * D`, the payload is `N/2 + N/2 + N/16 + N/16 = 9N/8` bytes. | codec | wrong slot size/out-of-bounds access | host size tests and kernel byte tests |
| I-005 | A layer payload is `[K packed | V packed | K block scales | V block scales]`; K and V scales remain independent. | codec/kernel | wrong dequantization | deterministic K/V scale tests |
| I-006 | `headDim` is positive and divisible by 16; `numKvHeads` and `tokensPerPage` are positive. No other model-geometry divisibility is required. | codec admission | incomplete scale group | negative and odd-page-size tests |
| I-007 | Every configured attention layer maps to exactly one K and one V hot buffer whose byte size matches its geometry/runtime dtype. | codec `configure()` | wrong address/stride | native codec tests |
| I-008 | FP8 runtime KV remains supported; FP8 source scales and NVFP4 destination scales are independent. | compression manager/kernel | scale-domain corruption | FP8 round-trip tests |
| I-009 | Encode/decode enqueue on the supplied stream and never publish, synchronize, release, or retry pages themselves. | codec/KVCM2 boundary | race/use-after-free | native tests and current KVCM2 contract |
| I-010 | Unsupported non-attention lifecycles use PR #17512's default lossless codec, without format-specific branches. | codec composition | hybrid-state corruption | lossless fallback test |
| I-011 | Each layer starts at a 16-byte-aligned relative cold offset. The staging base may have arbitrary byte alignment; the same kernel uses 16-byte, 8-byte, or byte transfers as appropriate. Padding is only inter-record/trailing padding, at most 15 bytes per layer, and is never interpreted. | codec layout/kernel | wrong vector access | layout and base+1 round-trip tests |
| I-012 | `tokensPerPage` follows KVCM2 `tokens_per_block` exactly. | KVCM2/config adapter | incompatible indexing | config/unit test and E2E manifest |
| I-013 | One native launch contains at most 128 local Attention layers. This is a descriptor-ABI bound, not a model-geometry divisibility rule. | boundary kernel plan | launch-argument overflow | admission test and model-scope audit |
| I-014 | Boundary transfer tiles are linear in half-groups and bounded at 2,048 half-groups. No row-count, GCD, or shared-memory admission condition is part of the format. | boundary kernel | hidden geometry limit | cross-row and 2-half-group-tail tests |
| I-015 | Codec batching remains lifecycle-keyed even when KVCM maps equal-size cold lifecycles into one physical PoolGroup. Physical allocation sharing does not imply transform equivalence. | codec | applying one lifecycle plan to another lifecycle's buffers/scales | identity representative tests and hybrid E2E |

The logical layout contains no padding inside the four payload segments. Alignment padding is outside a layer's
payload and does not create another KVCM2 pool, slot, or buffer. It is copied as part of the opaque cold page but is
never consumed by decode. The current encoder does not initialize those padding bytes; they are therefore outside the
serialized payload contract and must not be hashed or compared as deterministic data.

For one Attention layer, let `N = H * P * D`. The exact payload is `9N/8` bytes: `N/2` packed K, `N/2` packed V,
`N/16` K scales, and `N/16` V scales. Relative to two FP8 hot buffers (`2N` bytes), this is 43.75% smaller; relative
to two FP16/BF16 hot buffers (`4N` bytes), it is 71.875% smaller, before bounded record padding. Legal payload sizes
are even, so although the generic alignment bound is 15 bytes, the actual maximum padding for this geometry is 14
bytes. An observed 8-byte tail only means `9N/8 mod 16 == 8`; it does not impose another divisibility condition.

### Supported combinations

| Dimension | Supported | Rejected or deferred | Admission owner | Evidence |
|---|---|---|---|---|
| Backend / architecture | C++ KVCM2 on SM100-family; B200 validation target | pure-Python KVCM2; pre-SM100 | Python/native factory | B200 unit and Qwen3 E2E confirmed |
| Runtime KV dtype | FP16, BF16, FP8 E4M3 | other types | compression manager/codec | all three native paths confirmed; Qwen3 FP8 E2E confirmed |
| Cold levels | Host, Disk, or both through the common cold representation | custom tier with no GPU-accessible codec staging | KVCM2 contract | Qwen3 Host/Host+Disk/Disk-only confirmed |
| Attention layout | conventional per-layer K and V buffers, `headDim % 16 == 0`, at most 128 local Attention layers per lifecycle | unrecognized or incomplete K/V; MLA layouts not matching this contract; larger per-rank layer counts | codec `configure()` / kernel plan | positive, negative, odd-page, and tail tests confirmed |
| Full/SWA/hybrid | full and SWA attention; non-attention lifecycles lossless | compressing SSM/conv state | KVCM2 lifecycle + codec | codec hybrid test and Qwen3.5 full E2E confirmed |
| TP/PP/attention-DP | local K/V geometry; target manager only | unverified distributed combinations are not claimed | executor adapter | single-rank E2E first |
| Speculative target/draft | none in the initial support surface | speculative decoding is rejected by the config compatibility check | LLM args | Python unit tests |
| Overlap / CUDA Graph | no attention-path change; migration stays stream-ordered | independent performance claim | KVCM2 | E2E correctness only |
| Block reuse | supported; logical tokens and page identity do not change | lossy-byte equality claim | compression config | Qwen3 same-page lifecycle replay confirmed |

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

`scale_checkpoint_path` supplies calibration only and must be physically separate from the serving model. The public
KV dtype request must also satisfy that checkpoint loader's validator: for example, the audited Qwen3 runner requests
`fp8`, while Qwen3.5 accepts `auto` and resolves to BF16 but rejects an explicit `bfloat16` request. If the serving
checkpoint itself declares native NVFP4 KV, `auto` may resolve to hot NVFP4; that is not a source representation for
this boundary codec and construction fails closed. The accepted constructed hot dtype is FP16, BF16, or FP8.
`tokensPerPage` is never configured here; it follows KVCM2's `tokens_per_block`.

## Decisions

### D-001: Treat current PR #17512 as an immutable provider contract

```text
Status: accepted
Choice: Port only the NVFP4 consumer. Do not transplant old KVCM2 fixes or edit KVCM2 sources/tests.
Rationale and evidence: The force-pushed PR already redesigned cold pages, tier mapping, routing, staging, and failure
  fencing. Mixing the old branch's fixes would create a third contract and obscure ownership.
Rejected alternatives and why: Rebase all six old commits (contains obsolete KVCM2 changes); patch remaining side-path
  bugs opportunistically (outside the requested boundary).
Owners/lifetimes affected: none in KVCM2.
Public interface or failure policy: unchanged `IKvCacheColdPageCodec` contract.
Performance-sensitive contract: existing batching and stream/event topology remain unchanged.
Intended source symbols: no KVCM2 symbol.
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
Intended source symbols: `QuantizationCompression`, `create_nvfp4_cold_page_codec`.
Required tests/artifacts: ownership-consumption and repeated-construction tests.
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
Public interface or failure policy: padding has no public meaning and decode ignores it.
Performance-sensitive contract: relative layer offsets remain 16-byte aligned. An aligned staging base uses the
  vector path; an unaligned staging base uses the same kernel's 8-byte or byte fallback.
Intended source symbols: `compactLayerBytes`, layer cold-offset construction.
Required tests/artifacts: exact offsets and no out-of-bounds writes.
Compatibility/migration: cold raw copies include padding as opaque bytes.
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
Performance-sensitive contract: default codec retains its upstream batching/copy behavior.
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
Public interface or failure policy: FP8 requires explicit finite positive K and V source scales.
Performance-sensitive contract: existing fused boundary kernels only.
Intended source symbols: runtime-type dispatch and FP8 quantize/dequantize helpers.
Required tests/artifacts: FP8 K/V independent-scale round trip.
Compatibility/migration: none.
```

### D-008: Use one fixed linear transfer tile

```text
Status: accepted, exact-tree B200 validation pending
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
Status: accepted, exact-tree B200 validation pending
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

## Implementation plan

| Batch | Decision IDs | Intended symbols | Explicit non-changes | Verification |
|---|---|---|---|---|
| 1 | D-001..D-009 | NVFP4 kernel/codec/factory/binding and Python config adapter | KVCM2 source, migration, mapping, events | diff audit + format/static checks |
| 2 | D-002, D-004..D-007 | focused Python/C++/CUDA tests | upstream KVCM2 tests | unit/native tests on B200 |
| 3 | D-003 | exact prior Qwen runner adapted to new wheel | model/attention path | Host E2E + Disk E2E with activation evidence |
| 4 | all | this record, validation manifest, compatibility ledger | unrelated docs | conformance audit and remote ref verification |

## KVCM2 compatibility ledger

These entries are observations for alignment with the KVCM2 owner. They authorize no provider-side code change.

### Disposition of the old PR #17512 bug ledger

| ID | Current `984183e` disposition | Consumer action |
|---|---|---|
| C0 | fixed: `storageManager.cpp` includes `cudaDriverWrapper.h`, so batched `TLLM_CU_CHECK(cuMemcpyBatchAsync)` compiles | do not port old include fix; B200 native build covers it |
| C1 | fixed on the batched main path: `FuncGuard` fences source/destination finish events and eviction rollback reschedules the source | use upstream transaction path; do not add consumer fencing |
| C2 | fixed by redesign: raw same-representation migration is selected before codec lookup; source and destination pool groups are resolved independently per level | use raw Host/ Disk copies; Qwen3 `0->1->2->0` and Disk-only `0->1->0` are route evidence |
| C3 | generically open on single-page `_copyPageToTreeBlock`: the ready event is attached only after `copySlotData` returns | document for owner; no patch because this is not the batched main path and NVFP4 validates before launch, then drains already-submitted chunks on synchronous failure |
| C4 | fixed structurally: StorageManager owns the shared GPU physical allocator, levels borrow it, levels are destroyed before allocator reset | do not port old lifetime fix; current cold-page/allocator owner suites cover construction and teardown sequence |

### Consumer-specific compatibility items

| ID | Observation | NVFP4 main-path impact | Action in this branch |
|---|---|---|---|
| K-001 | C3's single-page false-return event window remains an owner side-path issue. | not on the normal batched migration path; current codec's failure behavior does not leave queued work unfenced | document only; do not patch KVCM2 |
| K-002 | Upstream has no separately named Host-to-Disk bypass test, although the raw branch is structurally ordered correctly. | no implementation change required | retain consumer-side Host+Disk route evidence |
| K-003 | The native binding consumes `cold_page_codec`, but the high-level PyExecutor adapter has no construction-time backend factory hook. Both old feature head `01b1ee56` and current head `984183e` directly construct `KVCacheManagerPy` in two places; the old NVFP4 branch's bridge was consumer-added commit `5195b62`, not an owner hook. | without a hook, LLMAPI constructs the default lossless codec; a post-construction swap is unsafe because cold sizes, pools, events, and page-table pointers are already fixed | blocking owner prerequisite; remove the proof bridge and wait for Yao Yao's generic hook before publishing |
| K-004 | Disk codec staging requests alignment `1`, so a codec cannot assume the cold base itself is 16-byte aligned. | resolved on the consumer side: the general tiled kernel retains aligned fast paths and adds an exact byte fallback; no KVCM2 alignment contract is required | base+1 BF16/FP8 unit test plus Qwen3.5 mixed-lifecycle Disk E2E; no owner change |

The minimal K-003 owner seam contains no compression policy:

```python
def _create_backend_impl(self, config):
    return KVCacheManagerPy(config, event_manager=self.event_manager)
```

The two direct constructor calls at base lines 1074 and 1086 then call this method. A compression-side mixin binds the
provider before `super().__init__`, overrides the hook to construct a fresh codec and pass the third native argument,
and wraps CUDA/OOM construction failure so the base cannot silently retry as GPU-only/lossless. Plain V2 and resolved
hybrid V2 classes share the same mixin; estimation, draft, and cross managers remain unwrapped.

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

## Conformance audit

### Design to implementation

| Decision/invariant | Implemented symbol | Verification | Result |
|---|---|---|---|
| D-001 / I-001 | no KVCM2 implementation symbol | path-filtered final diff | native/pure-runtime pass; high-level wrapper blocked by K-003 |
| D-002 | codec factory and final-target construction adapter | Python/native ownership tests | proof bridge passes; final mixin waits for owner hook |
| D-003 / I-002 / I-003 | existing KVCM2 migration contract + codec boundary calls | Host/Disk E2E | Qwen3 Host/Host+Disk/Disk-only pass |
| D-004 / D-008 / I-004..I-007 / I-012 / I-014 | compact layout, fixed linear tile, and plan validation | byte/layout/cross-row/large-dimension tests | previous layout passes on B200; exact slimmed tree pending |
| D-005 / I-011 | layer offset construction | exact-offset tests | pass on B200 |
| D-006 / I-010 | composed default codec | lossless fallback test | pass on B200; Qwen3.5 hybrid E2E pass |
| D-009 / I-015 | lifecycle-keyed codec batching | identity/sentinel tests | static source check complete; B200 validation pending |
| D-007 / I-008 | FP8 runtime dispatch and scales | FP8 round-trip test | pass on B200 and Qwen3 FP8 runtime |
| I-009 | encode/decode enqueue-only implementation | native stream/event tests | pass on B200; Nsight evidence pending |

### Implementation to design

This table must be completed after the port. The exit criterion is zero unmapped material guards, fallbacks, persistent
state fields, native calls, or support restrictions.

| Implementation item | Authorizing decision | Necessary? | Result |
|---|---|---|---|
| native codec and layer config | D-002, D-003 | yes | present and tested |
| three general NVFP4 boundary kernels | D-004, D-005, D-007, D-008 | yes | direct/shape-specific and row-shaped paths removed; B200 validation pending |
| default lossless codec member | D-006 | yes | present and tested |
| cross-lifecycle codec-equivalence map | none | no | removed; physical cold PoolGroup sharing remains upstream-owned |
| SM100 and dtype admission | support matrix, D-007 | yes | present and tested |
| Python control-plane retention after native handoff | D-002 | no | absent |
| any Host-only or token/scale divisibility guard | none | no | absent by source search and odd-page tests |
| any provider-side KVCM2 change | none | no | K-003 proof bridge must be removed before publish |

## Deviations

No KVCM2 deviation is accepted for the final publish branch. Mechanism-evidence HEAD `dd97da0` temporarily added a
generic `cold_page_codec_provider` handoff to the high-level wrapper; native and pure-runtime KVCM2 remained unchanged,
but that still violates the clarified boundary. Its B200 results may be cited only to prove the codec/storage pipeline.
The bridge and its owner-path tests must be removed after K-003 lands, then the exact final tree must be rebuilt and
rerun. K-004 is resolved entirely in the compression-owned transfer helpers and does not require an owner change.

## Validation and handoff

### Completed mechanism evidence (`dd97da0`)

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
| Qwen3.5 functional | job 3718654, seven of seven arms and three of three raw/NVFP4 comparisons PASS; `auto` resolved to BF16 |
| Qwen3.5 mixed lifecycle | SSM/conv lifecycle stayed lossless; 17 Attention pages and one SSM page completed the forced round trip |
| Qwen3.5 Disk routes | Host+Disk `0->1->2->0`; Disk-only `0->1->0`; real `/dev/md0` ext4 backing |
| Qwen3.5 storage | raw BF16 Attention slot 1,048,576 bytes; NVFP4 slot 294,912 bytes, the expected 71.875% reduction; SSM/conv slot unchanged |
| Qwen3.5 artifact | `/home/scratch.tianruih_coreai/artifacts/nvfp4-pr17512-latest-e2e-results-20260817/20260817T134840Z-qwen35-functional-slurm3718654` |
| Qwen3 Nsight | job 3718655 captured 60 FP8 offload launches and one 17-page FP8 onboard launch in the correct nested NVTX ranges; product workload PASS, verifier name filter failed |

This is a validation-only incremental build namespace, not a clean full rebuild. The Qwen3 result proves activation,
route correctness, reuse, and Disk residency on a real ext4 block device. Its short exact replay is diagnostic and is
not a general accuracy claim. It also does not by itself prove kernel/NVTX ranges or `pwrite`/`pread` nesting.

### Incomplete/failed harness attempts

| Attempt | Disposition |
|---|---|
| job 3718172 | recursive Git-alternate preflight failure before any arm; harness-only |
| job 3718260 | Qwen3.5 explicit `bfloat16` rejected by its loader before model/KVCM construction; config-contract finding, not codec failure |
| job 3718357 | source-dirty preflight failure before profiler launch; harness-only |
| jobs 3718478/3718479 | sealed r6 normalized nanobind dtype enum values to strings `7`/`6`; failed the evidence gate after manager construction and before any usable functional/profile workload |
| job 3718655 | r7 expected an invented `onboardToFp8` symbol; the trace contains `onboardKernel<32, __nv_fp8_e4m3>` with the correct task ABI, grid, codec range, and outer onboard range. This is a verifier-only failure; rerun uses task-ABI and template-dtype matching. |

The corrected Nsight acceptance requires all three NVFP4 storage arms, fused boundary kernels inside the matching
offload/onboard NVTX ranges, and Disk `pwrite` during forced offload plus `pread` during forced onboard. Kernel
classification is based on `Nvfp4BoundaryOffloadPageTask` / `Nvfp4BoundaryOnboardPageTask`; dtype evidence comes from
the offload family and onboard template argument rather than a shape-specific symbol name.

### Final publish exit criteria

- K-003 owner hook present in the stacked base; proof bridge removed; all KVCM2 paths have zero final diff.
- Arbitrary cold-base alignment and mixed-lifecycle Disk support remain covered by exact-tree unit and Qwen3.5 E2E.
- General-kernel slimming, if retained, passes the same native suite and exact-tree B200 E2E.
- Final source is clean, latest PR head is rechecked, remote SHA matches, and the compare link is against that exact PR
  head. A docs-only commit may cite code-parent evidence, but code changes require a fresh exact-tree build/run.
