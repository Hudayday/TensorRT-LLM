<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVFP4 Cold-Page Codec Design Record — 2026-08-17

## Record metadata

```text
Title: NVFP4 cold-page codec on the current KVCM2 cold-page contract
Record ID / revision: NVFP4-KVCM2-2026-08-17 / r1
Status: accepted for implementation; validation evidence pending
Date / owner: 2026-08-17 / KV-cache compression side
Repository / branch / HEAD / dirty-diff hash:
  Hudayday/TensorRT-LLM
  nvfp4-pr17512-latest-e2e-clean-20260817
  base 984183e480cd7e23f79d73005fb395d923803b28
  clean before this record
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
- keep the final diff under native `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/` and pure-runtime
  `tensorrt_llm/runtime/kv_cache_manager_v2/` empty; and
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
| E-008 | `HYPOTHESIS` | The exact port compiles against `984183e` without a KVCM2 source change. | incremental B200 build | open |
| E-009 | `HYPOTHESIS` | Direct Disk offload reaches encode, raw Disk storage, and decode through the current staging path. | activation-proven B200 Disk E2E | open |

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
| I-001 | No implementation diff under `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`. | port boundary | review failure | path-filtered diff |
| I-002 | One cold `SlotId` selects one physical cold pool and one opaque logical blob at every cold level. | KVCM2 PR #17512 | incompatible construction | existing KVCM2 tests plus E2E |
| I-003 | Host and Disk use byte-identical cold payloads; cold-to-cold migration does not invoke the codec. | KVCM2 | corruption or needless requantization | Disk E2E and logs/counters |
| I-004 | For layer element count `N = H * P * D`, the payload is `N/2 + N/2 + N/16 + N/16 = 9N/8` bytes. | codec | wrong slot size/out-of-bounds access | host size tests and kernel byte tests |
| I-005 | A layer payload is `[K packed | V packed | K block scales | V block scales]`; K and V scales remain independent. | codec/kernel | wrong dequantization | deterministic K/V scale tests |
| I-006 | `headDim` is positive and divisible by 16; `numKvHeads` and `tokensPerPage` are positive. No other model-geometry divisibility is required. | codec admission | incomplete scale group | negative and odd-page-size tests |
| I-007 | Every configured attention layer maps to exactly one K and one V hot buffer whose byte size matches its geometry/runtime dtype. | codec `configure()` | wrong address/stride | native codec tests |
| I-008 | FP8 runtime KV remains supported; FP8 source scales and NVFP4 destination scales are independent. | compression manager/kernel | scale-domain corruption | FP8 round-trip tests |
| I-009 | Encode/decode enqueue on the supplied stream and never publish, synchronize, release, or retry pages themselves. | codec/KVCM2 boundary | race/use-after-free | native tests and current KVCM2 contract |
| I-010 | Unsupported non-attention lifecycles use PR #17512's default lossless codec, without format-specific branches. | codec composition | hybrid-state corruption | lossless fallback test |
| I-011 | Each layer starts at a 16-byte-aligned cold offset. Padding is only inter-record/trailing padding, at most 15 bytes per layer, and is never interpreted. | codec layout | misaligned vector path | layout tests |
| I-012 | `tokensPerPage` follows KVCM2 `tokens_per_block` exactly. | KVCM2/config adapter | incompatible indexing | config/unit test and E2E manifest |

The logical layout contains no padding inside the four payload segments. Alignment padding is outside a layer's
payload and does not create another KVCM2 pool, slot, or buffer. It is copied as part of the opaque cold page but is
never consumed by decode.

### Supported combinations

| Dimension | Supported | Rejected or deferred | Admission owner | Evidence |
|---|---|---|---|---|
| Backend / architecture | C++ KVCM2 on SM100-family; B200 validation target | pure-Python KVCM2; pre-SM100 | Python/native factory | unit + B200 E2E pending |
| Runtime KV dtype | FP16, BF16, FP8 E4M3 | other types | compression manager/codec | native tests pending |
| Cold levels | Host, Disk, or both through the common cold representation | custom tier with no GPU-accessible codec staging | KVCM2 contract | Disk E2E pending |
| Attention layout | conventional per-layer K and V buffers, `headDim % 16 == 0` | unrecognized or incomplete K/V; MLA layouts not matching this contract | codec `configure()` | negative tests pending |
| Full/SWA/hybrid | full and SWA attention; non-attention lifecycles lossless | compressing SSM/conv state | KVCM2 lifecycle + codec | focused tests pending |
| TP/PP/attention-DP | local K/V geometry; target manager only | unverified distributed combinations are not claimed | executor adapter | single-rank E2E first |
| Speculative target/draft | none in the initial support surface | speculative decoding is rejected by the config compatibility check | LLM args | Python unit tests |
| Overlap / CUDA Graph | no attention-path change; migration stays stream-ordered | independent performance claim | KVCM2 | E2E correctness only |
| Block reuse | supported; logical tokens and page identity do not change | lossy-byte equality claim | compression config | replay E2E pending |

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
Performance-sensitive contract: layer base remains 16-byte aligned; existing 8-byte/byte tails remain general.
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

## Implementation plan

| Batch | Decision IDs | Intended symbols | Explicit non-changes | Verification |
|---|---|---|---|---|
| 1 | D-001..D-007 | NVFP4 kernel/codec/factory/binding and Python config adapter | KVCM2 source, migration, mapping, events | diff audit + format/static checks |
| 2 | D-002, D-004..D-007 | focused Python/C++/CUDA tests | upstream KVCM2 tests | unit/native tests on B200 |
| 3 | D-003 | exact prior Qwen runner adapted to new wheel | model/attention path | Host E2E + Disk E2E with activation evidence |
| 4 | all | this record, validation manifest, compatibility ledger | unrelated docs | conformance audit and remote ref verification |

## KVCM2 compatibility ledger

These entries are observations for alignment with the KVCM2 owner. They authorize no provider-side code change.

| ID | Observation | NVFP4 main-path impact | Action in this branch |
|---|---|---|---|
| K-001 | The previously recorded single-page `_copyPageToTreeBlock` false-return fence/release concern requires a fresh audit on `984183e`. | not currently demonstrated on the main codec migration path | document only; do not patch KVCM2 |
| K-002 | Exact same-representation Host-to-Disk coverage in upstream unit tests requires confirmation. | Disk correctness will be covered by this branch's E2E evidence | document/test from consumer side only |
| K-003 | The native binding consumes `cold_page_codec`, but the high-level PyExecutor adapter has no construction-time codec-provider hook. | without a hook, LLMAPI E2E silently uses the default lossless codec | add only a generic provider handoff in the PyExecutor adapter; align the missing upstream hook with the KVCM2 owner |

## Decision log

| Revision/date | Evidence or request | Old decision | Approved replacement | Affected code/tests/docs | Approval |
|---|---|---|---|---|---|
| r1 / 2026-08-17 | User requested a direct port to latest #17512 and prohibited KVCM2 edits. | old branch included provider-side bug fixes | latest #17512 is immutable; consumer-side adaptation only | whole port | user |
| r1 / 2026-08-17 | User requested a general design with no new special kernel. | possible geometry-specific restrictions | keep only format constraints and existing general tail paths | codec/kernel tests | user |
| r1 / 2026-08-17 | User explicitly retained FP8 and allowed bounded padding. | possible FP8 deletion or fully tight records | retain FP8; retain only per-layer 16-byte alignment | codec/kernel/docs | user |

## Conformance audit

### Design to implementation

| Decision/invariant | Implemented symbol | Verification | Result |
|---|---|---|---|
| D-001 / I-001 | no KVCM2 implementation symbol | path-filtered final diff | pending |
| D-002 | codec factory and final-target construction adapter | Python/native ownership tests | pending |
| D-003 / I-002 / I-003 | existing KVCM2 migration contract + codec boundary calls | Host/Disk E2E | pending |
| D-004 / I-004..I-007 / I-012 | compact layout and plan validation | byte/layout tests | pending |
| D-005 / I-011 | layer offset construction | exact-offset tests | pending |
| D-006 / I-010 | composed default codec | lossless fallback test | pending |
| D-007 / I-008 | FP8 runtime dispatch and scales | FP8 round-trip test | pending |
| I-009 | encode/decode enqueue-only implementation | native stream/event tests | pending |

### Implementation to design

This table must be completed after the port. The exit criterion is zero unmapped material guards, fallbacks, persistent
state fields, native calls, or support restrictions.

| Implementation item | Authorizing decision | Necessary? | Result |
|---|---|---|---|
| native codec and layer config | D-002, D-003 | yes | pending |
| existing NVFP4 boundary kernel pair | D-004, D-005, D-007 | yes | pending |
| default lossless codec member | D-006 | yes | pending |
| SM100 and dtype admission | support matrix, D-007 | yes | pending |
| Python control-plane retention after native handoff | D-002 | no | deleted |
| any Host-only or token/scale divisibility guard | none | no | must be absent |
| any provider-side KVCM2 change | none | no | must be absent |

## Deviations

One temporary integration deviation is recorded as K-003. This branch adds a generic construction-time
`cold_page_codec_provider` handoff in the high-level PyExecutor adapter because the latest PR exposes only the native
ownership ABI. It contains no NVFP4 layout, migration, mapping, or event logic. The KVCM2 owner is Yao Yao; the local
hook should be removed when the upstream high-level adapter exposes an equivalent provider/factory parameter. The
construction-order and final-target scoping tests identify when replacement is safe. No native or pure-runtime KVCM2
source is changed.

## Validation and handoff

Evidence for the exact final tree will be recorded here after execution:

- base/HEAD/dirty state and path-filtered diff;
- orphan/deletion-closure and stale-symbol searches;
- formatting, generated LLM-args manifest, API stability, and focused Python tests;
- incremental SM100/B200 native build and codec/kernel tests;
- Host E2E with codec activation evidence;
- Disk E2E proving cold-page residency/migration and decode activation;
- source, wheel, container, model, calibration, runner, job, and artifact identities; and
- pushed remote SHA plus a GitHub compare link against `984183e`.
