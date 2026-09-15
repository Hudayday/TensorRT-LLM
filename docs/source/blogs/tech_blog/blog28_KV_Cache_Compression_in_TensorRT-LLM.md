# KV Cache Compression in TensorRT LLM

**Table of Contents**
- [Introduction and Motivation](#introduction-and-motivation)
- [Overview of KV Cache Compression in TensorRT LLM](#overview-of-kv-cache-compression-in-tensorrt-llm)
- [KV Cache Compression Framework Design](#kv-cache-compression-framework-design)
  - [Design Philosophy](#design-philosophy)
  - [Architecture Overview](#architecture-overview)
  - [Iteration-Driven Lifecycle](#iteration-driven-lifecycle)
  - [Storage-Bound Lifecycle and the Cold-Page Codec Contract](#storage-bound-lifecycle-and-the-cold-page-codec-contract)
  - [Configuration and Ownership](#configuration-and-ownership)
- [Algorithm Implementations](#algorithm-implementations)
  - [NVFP4 Cold-Page Quantization](#nvfp4-cold-page-quantization)
  - [TriAttention](#triattention)
- [Evaluation](#evaluation)
  - [NVFP4 Cold-Page Quantization](#nvfp4-cold-page-quantization-1)
  - [TriAttention](#triattention-1)
- [Summary and Future Work](#summary-and-future-work)
  - [Current State](#current-state)
  - [Future Work](#future-work)

## Introduction and Motivation

The tasks handed to Large Language Models (LLMs) keep getting more complex and more expensive to serve: models grow, the text they read and write grows with them, and nowhere is this more visible than in agentic workflows, where a model works through a task in many steps instead of one reply. An agent job is a chain of model calls separated by tool calls, and each tool result is appended to one growing conversation. Input length grows turn by turn, tool calls repeat dozens of times per job, and most of every prompt is text the system has already seen. For example, in the [InferenceX](https://inferencex.semianalysis.com/) AgentX coding traces the median input length per request is 14.4k tokens and over 96% of the prompt tokens are reusable prefix.

The KV cache hit rate therefore becomes the dominant factor in agentic serving: it decides how much prefill is redundant, and losing cached KV is paid for on almost every turn. Serving such workloads well means keeping as much of that KV as possible for as long as it is useful, at a cost the deployment can afford. This calls for an advanced compression method, and a framework that lets it compress the KV cache efficiently and take the burden off GPU memory, host memory, and the links between them.

Three things follow for such a method:

- **KV cache volume.** Each request carries tens of thousands of tokens of KV, a job produces dozens of requests, and the same prefix pages are requested again and again across the workload. The KV worth keeping is far larger than any single request.
- **Capacity and data movement.** Whether a prefix is still present at the next turn depends on how many pages the GPU, host, and disk tiers can retain, and every page kept in a tier or moved across a tier boundary costs bytes. Capacity and movement are one budget.
- **A method for every model.** Agentic workloads are memory-heavy for every model that runs them, so the compression method must apply across attention layouts and cache structures without model-specific kernels.

A wide range of KV cache compression methods have been proposed to address these needs. They can be broadly classified along two dimensions: **what** is compressed (which tokens are retained, as in eviction, versus how the retained values are stored, as in quantization or another compact encoding) and **when and where** the compression runs (inside the forward iteration, so that the attention kernel consumes the compressed state directly, versus at a storage boundary, when pages move between the GPU and a colder tier). TensorRT LLM already covers the first class through active KV cache quantization and the [sparse attention framework](blog17_Sparse_Attention_in_TensorRT-LLM.md); this blog addresses the second, which the serving lifecycle supports at natural, stable points, between forward steps and at hot/cold page transitions, without any change to the attention kernel.

To bring these lifecycle-level methods into production, TensorRT LLM introduces a **unified KV cache compression framework** with two integration models, iteration-driven and storage-bound, built on `KVCacheManagerV2` (KVCM V2). On this framework, TensorRT LLM currently ships two methods that occupy distinct points of the design space:

- **NVFP4 cold-page quantization** (storage-bound): compresses attention KV pages only when they leave the GPU for the host or disk tier, fused with the transfer itself, and restores the runtime precision when they return.
- **[TriAttention](https://arxiv.org/abs/2604.04921)** (iteration-driven): a training-free method that periodically scores and evicts generation tokens between decode iterations once a sequence exceeds a token budget.

In the following sections, we first provide an overview of the KV cache compression capabilities in TensorRT LLM, then describe the framework design that makes them possible, walk through how each method is implemented on top of it, and finally present evaluation results.

## Overview of KV Cache Compression in TensorRT LLM

A key challenge in deploying KV cache compression at scale is the diversity of existing methods: they differ in **lifecycle** (inside the model iteration vs. at a storage transition), in **what they change** (the set of retained tokens vs. the representation of the retained values), and in the **cache structures** they can handle (conventional MHA/GQA pages, MLA latent pages, the mixed attention-plus-state layouts of hybrid models). To handle this diversity without method-specific branches in the executor or the cache manager, TensorRT LLM introduces a **unified, extensible KV cache compression framework**: one configuration and factory path, one manager base class, and two standardized contracts (lifecycle hooks for iteration-driven methods and a cold-page codec interface for storage-bound methods). Page ownership, migration, reuse, and mapping publication stay inside KVCM V2, so a compression method never allocates or publishes pages itself.

To demonstrate the framework's generality, we have integrated two methods that exercise the two contracts:

*   **NVFP4 cold-page quantization**: a storage-bound method that re-encodes attention KV pages into packed NVFP4 while they reside in the host or disk tier and decodes them on the way back; the GPU cache and the attention kernels keep the runtime KV type.
*   **TriAttention**: an iteration-driven method that runs between decode steps, scores the generation region of the cache with calibrated per-head statistics, and compacts the cache to a fixed budget while preserving the prompt.

The following tables summarize the current coverage:

<div align="center">

| Method | Lifecycle | What It Changes | Cache Structures |
| :--- | :--- | :--- | :--- |
| **NVFP4 cold-page quantization** | Storage-bound (GPU <-> Host/Disk migration) | Stored representation of cold attention KV | MHA / MQA / GQA; key-only MLA latent pages; hybrid attention + GDN/SSM models with non-attention state kept lossless |
| **TriAttention** | Iteration-driven (generation phase) | Set of KV tokens retained | MHA / MQA / GQA dense attention over `KVCacheManagerV2` pools |

</div>

<div align="center">

| Cache Tier / Phase | NVFP4 Cold-Page Quantization | TriAttention |
| :--- | :--- | :--- |
| **GPU (active KV)** | Unchanged (FP16 / BF16 / FP8 runtime type) | Generation tokens evicted periodically; prompt preserved |
| **Host tier** | Packed NVFP4 (E2M1) data + E4M3 block scales | n/a |
| **Disk tier** | Same compressed blob as the host tier, no re-quantization | n/a |

</div>

**Note**: Today, both methods require the PyTorch backend, KVCM V2, and an NVIDIA GPU with compute capability SM100 or SM103. The results in this blog were measured on GB300 systems; earlier functional validation of the cold-page path was done on B200.

This blog focuses on the **framework-level** design that is common across methods and on the NVFP4 cold-page implementation as the main worked example. For the C++ codec ABI, staging, and migration transaction, please refer to the [KVCacheManagerV2 Cold-Page Codec Design](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-cold-page-codec.md); for the extension APIs, please refer to the [KV Cache Compression Development Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-compression-development.md).

## KV Cache Compression Framework Design

The KV cache compression framework in TensorRT LLM is designed to provide a common runway for different methods while hiding most of the complexity from end users. Our goal is to make it straightforward for developers to integrate a new compression method by implementing one of two contracts (lifecycle hooks or a cold-page codec) without modifying the attention kernels, the cache manager's allocation and migration logic, or the serving infrastructure.

### Design Philosophy

KV cache compression methods and their algorithm-specific policies are selected via `KvCacheCompressionConfig`, which is deliberately orthogonal to `KvCacheConfig` (capacity, cache tiers, block reuse, active KV dtype) and `SparseAttentionConfig` (how attention computes). Only one compression method can be enabled per LLM instance, and the method must either handle every cache layout it is given or preserve the layouts it does not understand losslessly.

Lifecycle-level compression methods follow a common pattern:

- **Observe** the KV state at a stable boundary: after a forward step, or when the cache manager moves a page across tiers.
- **Transform** that state: evict tokens and compact the cache, or re-encode a page into a compact representation.
- **Reconcile** with the runtime: publish the new physical KV length, or hand the encoded bytes back to the cache manager so that the next consumer sees a valid page.

The framework abstracts this pattern into two standardized contracts:

- **Lifecycle hooks** for iteration-driven methods: `on_request_init`, `on_context_step_end`, `on_generation_step_begin`, `on_generation_step_end`, and `on_request_finish`, invoked by the executor around the existing request cycle.
- **The cold-page codec contract** for storage-bound methods: an encode/decode interface over batches of `{dst, src}` page-index records, invoked by KVCM V2 whenever pages cross the hot/cold boundary.

Two rules hold across both contracts. First, compression runs **outside the attention kernel**: the attention backend consumes the published active GPU representation and never sees the compressed form. Second, **KVCM V2 remains the authority** for pages, slots, cache levels, allocation, migration, reuse, completion ordering, and mapping publication. A method decides *what* to transform and *how*; it does not decide *where* bytes live.

### Architecture Overview

At a system level, the KV cache compression framework is built around three key components:

- A **lifecycle adapter** (`KVCacheCompressionManager`) that turns the executor's iteration hooks or the cache manager's migration requests into method-specific actions.
- A **transform operator** (a selection and compaction kernel for iteration-driven methods, or a codec encode/decode kernel for storage-bound methods) that performs the batched transformation on the supplied CUDA stream.
- **KVCM V2 ownership of pages and migration**, which allocates slots, orders events, invokes the codec, and publishes mappings, so that compressed state is consumed through the same paths as uncompressed state.

From a user perspective, all of this is controlled by a high-level `KvCacheCompressionConfig`. When such a config is provided, a factory validates the requested combination and constructs the concrete manager before the model runs or any page migrates.

<!-- TODO: draw figure -->
<div align="center">
<figure>
  <img src="../media/tech_blog28_framework.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 1: KV cache compression framework in TensorRT LLM. The executor invokes the lifecycle hooks of an iteration-driven manager around each forward step; KVCacheManagerV2 invokes the cold-page codec of a storage-bound manager whenever pages migrate between the GPU and the host or disk tier. Both paths share one configuration, factory, and manager base class.</em></sub></p>

Figure 1 summarizes how these components work together along the request path. An iteration-driven manager sets `uses_iteration_lifecycle = True` and is registered as a resource manager with the executor; `bind_kv_cache_managers()` gives it access to stable cache geometry once KVCM V2 exists. A storage-bound manager sets `provides_cold_page_codec = True` and is constructed *before* KVCM V2, because the cache manager needs the codec to size and lay out its cold tiers. A method may use either path or both, but the two are kept separate: migration policy does not belong in an iteration hook, and a cold-page provider does not allocate or publish pages.

### Iteration-Driven Lifecycle

During a prefill or decode forward pass, attention consumes a stable view of the KV cache; once that step completes and before the next begins, the framework can update the physical KV state that subsequent execution will use.

`KVCacheCompressionManager` exposes five semantic hooks at these lifecycle points. Methods override only the hooks they need; all five default to no-ops.

<div align="center">

| Hook | Exact Trigger | Appropriate Work |
| :--- | :--- | :--- |
| `on_request_init(request)` | Before a request's first prefill chunk | Initialize request-local compression state |
| `on_context_step_end(requests)` | After a request's final prefill chunk | Compress the completed context before generation, when needed |
| `on_generation_step_begin(scheduled_batch)` | Before each scheduled forward iteration | Prepare an iteration-level compression action, when needed |
| `on_generation_step_end(scheduled_batch)` | After each scheduled forward iteration and KV cache update | Compress the updated KV state before the next iteration, when needed |
| `on_request_finish(request)` | When a request completes or aborts | Release request-local compression state |

</div>

These hooks reuse the executor's existing request cycle; the framework handles registration and callback wiring. Methods that fit this lifecycle change the *set* of retained tokens or their *arrangement* in the paged pool. Two obligations come with the contract: selection policy stays separate from the generic compaction kernel, and a method must publish completion of its GPU work before it resizes or releases KVCM-owned capacity. TriAttention, described below, is the shipped example: it uses the generation-end hook and leaves scheduling, prompt-prefix block reuse, and the dense attention kernel unchanged.

### Storage-Bound Lifecycle and the Cold-Page Codec Contract

Storage-bound methods run when KV pages move across cache tiers. During offloading, a hot GPU page is encoded into a compressed representation as it moves to host or disk storage; during onboarding, the cold page is transferred back and decoded into the runtime GPU representation before it is reused. Figure 2 shows the data path.

<!-- TODO: draw figure -->
<div align="center">
<figure>
  <img src="../media/tech_blog28_cold_page_data_path.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 2: Hot-to-cold-to-hot data path for a storage-bound method. Encoding is fused with the GPU-to-host transfer, the host and disk tiers share one compressed blob, and decoding is fused with the host-to-GPU transfer. The GPU page and the attention kernel never see the compressed representation.</em></sub></p>

The contract that makes this possible is a small C++ interface in KVCM V2, `IKvCacheColdPageCodec`. A hot page may span several kernel-facing pools (K and V buffers, an MLA latent buffer plus an indexer key buffer, or the mixed pools of a hybrid model) while a cold page is **one fixed-size opaque blob**. The codec transforms between the two representations, and its contract has four load-bearing properties:

- **Fixed cold-page size per lifecycle.** `queryColdPageBytes()` returns one payload size per layer group. Variable-length pages are not supported, so every tier addresses blobs as `coldBase + slot * coldPageBytes` through a simple slot allocator.
- **Batched, descriptor-driven calls.** KVCM V2 hands the codec an array of `PageIndexPair {int32 dst, int32 src}` records (one 8-byte record per page copy) with a batching layer-group identifier, the cold base address, and a CUDA stream. Lifecycles in the same codec-equivalence class are concatenated into one call.
- **Enqueue-only execution.** `encode()` and `decode()` enqueue work on the supplied stream and do not synchronize it; the boolean result reports submission validity, not completion. KVCM V2 owns the event and ownership transaction around every transfer, so a codec inherits rollback and lifetime guarantees.
- **Opaque cold-to-cold copies.** Host-to-disk and disk-to-host migrations are raw blob copies; only hot/cold conversions invoke the codec, and disk addresses are never exposed to it.

Compression is optional at this layer. The default codec concatenates all hot-pool data into the cold blob without reducing its size; compressing codecs produce smaller blobs through the same paths and can declare spans lossless: non-attention state such as GDN, SSM, and convolution buffers, and attention side buffers such as a DSA indexer key, pass through unchanged inside the same page.

On the Python side, a storage-bound manager implements five provider APIs: `create_cold_page_codec()`, `build_codec_state()` (format state and handled layers), `build_lifecycle_metadata()` (the cold-page layout per lifecycle), and `encode_cold_pages()` / `decode_cold_pages()` (the batched transforms). A native adapter, `NativeColdPageCodec`, resolves the KVCM layout, routes each lifecycle to the provider or to the lossless fallback, and bridges Python and native lifetimes. It makes exactly **one** C++-to-Python callback per KVCM page batch; the launcher then chunks the batch in **256-page** groups, so a 4,096-page migration costs one callback and sixteen kernel launches. A new format adds a kernel and a policy; tier routing, staging, chunking, and event ordering are shared.

### Configuration and Ownership

Because the framework spans the executor, the cache manager, and native kernels, its most important design decision is who owns what. The table below summarizes the boundary.

<div align="center">

| Component | Owns | Does Not Own |
| :--- | :--- | :--- |
| `KvCacheCompressionConfig` and the factory | Method selection and admission | Pages, kernels, request mappings |
| `KVCacheCompressionManager` | Method cadence, request state, format metadata, kernel launches | KVCM allocation policy, attention runtime state |
| `NativeColdPageCodec` | Layout resolution, provider/fallback routing, Python/native lifetimes | Format-specific quantization policy |
| `KVCacheManagerV2` | Pages, slots, pools, mappings, migration, events, publication, rollback, cold storage | Method scores, quantization decisions |
| `AttentionBackend` | The published active GPU representation | Cold storage, migration |

</div>

Compatibility checks are admission predicates, not runtime fallbacks: an unsupported combination is rejected before construction, and a method that has begun moving bytes follows the completion and failure contract rather than silently falling back. Configurations declare three independent capabilities: whether the method changes the physical KV length, whether block reuse remains valid, and whether speculative decoding is supported.

Configuration is opt-in and small. Cold-page quantization is enabled by setting `algorithm: quantization_for_cold_page` and `quant: nvfp4` in `kv_cache_compression_config`, together with `use_kv_cache_manager_v2: true` and a nonzero `host_cache_size` (optionally `disk_cache_size` with `disk_cache_path`) in `kv_cache_config`; `kv_cache_config.dtype` stays at the runtime KV type. TriAttention is enabled by setting `algorithm: triattention` with `budget`, `beta`, `eviction_mode`, and `calibration_path`. The field names are identical in the Python API (`ColdPageQuantizationCompressionConfig`, `TriAttentionKvCacheCompressionConfig`), `trtllm-serve` YAML, and `trtllm-bench`.

Calibration and other offline artifacts are inputs, not work: a method must not calibrate in the inference critical path; the runtime accepts an artifact path, validates it during initialization, and fails fast if it is missing or incompatible. For the full option table and enablement checklist, please refer to the [KV Cache Compression feature documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md).

## Algorithm Implementations

In this section, we walk through the two KV cache compression methods currently implemented in TensorRT LLM, focusing on how each method works and how it integrates with the framework. For a quick-start guide and runnable configurations, please refer to the [KV cache compression examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/kv_cache_compression).

### NVFP4 Cold-Page Quantization

#### Introduction

NVFP4 is a 4-bit floating-point format (E2M1) in which every group of 16 values shares one FP8 (E4M3) block scale, optionally combined with a per-tensor global scale. It is the format TensorRT LLM already uses for NVFP4 weights and for the active NVFP4 KV cache, so its rounding behavior, scale contract, and conversion primitives are established. Cold-page quantization applies it at a different point in the system: instead of storing the *active* KV cache in NVFP4 and asking the attention kernel to consume it, it stores attention KV in NVFP4 **only while its pages reside in the host or disk tier**. The GPU cache and the attention kernels keep the model's runtime KV type (FP16, BF16, or FP8).

The cold tier is the right place for this transform for three reasons. In a reuse-heavy deployment the binding quantities are the bytes *stored* in the cold tiers and the bytes *moved* across the PCIe and storage links, so a smaller cold page buys more retained pages per gigabyte and fewer bytes per migration. Because the active KV is never quantized, no per-step quantization error accumulates, and a page that never leaves the GPU is never touched. And the transform is transparent to everything above it: page identity, token identity, block reuse, scheduling, and the attention-visible GPU layout are unchanged.

The layout arithmetic is straightforward. Packed E2M1 data plus one E4M3 scale per 16 values costs 0.5625 bytes per value, against 2 bytes for FP16/BF16 and 1 byte for FP8. After the per-span alignment the layout imposes, the measured cold-page sizes are:

- **FP8-hot models**: 43.75% fewer cold-tier bytes per attention page (Qwen3.5-397B-A17B 1,048,576 -> 589,824 B; Qwen3-8B with FP8 KV 2,359,296 -> 1,327,104 B), i.e. 1.78x more pages per cold-tier byte.
- **MLA latent pages (GLM-5.2)**: 41.1% fewer bytes (3,098,112 -> 1,824,000 B), i.e. 1.70x; the ratio is lower because the indexer key spans in the same page are kept lossless.
- **BF16-hot models**: 71.9% fewer bytes (Qwen3-8B with BF16 KV 4,718,592 -> 1,327,104 B), i.e. 3.56x.

These are attention-lifecycle figures; non-attention lifecycles of hybrid models (GDN, SSM, and convolution state) are stored losslessly and do not shrink.

#### How It Works in TensorRT LLM

Within TensorRT LLM, NVFP4 cold-page quantization is integrated as a storage-bound compression manager that provides a cold-page codec to KVCM V2. Below we highlight the key design choices in provider construction, kernel design, scale handling, and activation evidence.

**Codec provider.** `ColdPageQuantizationCompressionConfig(quant="nvfp4")` selects a manager with `provides_cold_page_codec = True` and `uses_iteration_lifecycle = False`, constructed before KVCM V2. During `configure()`, the provider receives every hot pool-group descriptor and base address in one call, resolves the runtime KV dtype and per-layer shapes, and builds a per-lifecycle **layout table** exactly once: for MHA/MQA/GQA pages the K and V buffers become NVFP4 data plus block scales; for key-only MLA the latent attention key is encoded; auxiliary roles in the same attention lifecycle, such as a DSA indexer key, are appended losslessly; and non-attention lifecycles are routed to the default lossless codec. For DeepSeek-V4, the NoPE prefix of the compressed sparse-attention history is encoded as NVFP4 while its RoPE suffix and the remaining specialized cache state are preserved in the same cold page. Figure 3 shows the hot and cold layouts side by side.

<!-- TODO: draw figure -->
<div align="center">
<figure>
  <img src="../media/tech_blog28_cold_page_layout.png" width="800">
</figure>
</div>
<p align="center"><sub><em>Figure 3: Hot page layout versus compressed cold page layout. A hot page spans one or more kernel-facing pools in the runtime KV type; the cold page is one fixed-size blob containing packed NVFP4 spans with E4M3 block scales for the attention KV and byte-exact lossless spans for buffers that are not quantized.</em></sub></p>

**Fused encode/decode kernels.** Encoding and transfer are fused in one descriptor-driven CUDA kernel family: the offload kernel reads the hot page from GPU memory, quantizes it, and writes the packed cold page directly into the mapped host slot; the onboard kernel reads the cold page from host memory, dequantizes it, and writes the runtime-type page into the GPU pool. Each launch consumes the resolved lifecycle metadata, a chunk of `PageIndexPair` records, the cold base pointer, and the stream KVCM V2 supplies, with no per-page host work. There is **no model-specific kernel**: the same launcher covers MHA, GQA, MQA, MLA latent, and the DeepSeek-V4 layout, driven entirely by the layout table; extending the codec to DeepSeek-V4 required a Python layout policy and its CUDA operator, and changed neither KVCM nor the generic codec adapter. The host and disk tiers share the encoded blob, so disk migration never re-quantizes.

**Scales and calibration.** By default, the codec uses identity K/V global scales, so a regular FP16, BF16, or FP8 checkpoint requires no calibration step; the dynamic E4M3 block scale for each group of 16 values is computed during encode and stored in the cold page. A compatible NVIDIA ModelOpt NVFP4 checkpoint can optionally supply per-layer K/V global-scale metadata through `scale_checkpoint_path`; this applies to the two-buffer K/V layout, while key-only MLA and draft-model cold pages use identity global scales. Weight quantization and KV compression are independent: a checkpoint with NVFP4 weights and FP8 active KV uses this feature normally, and a model whose active KV is already NVFP4 is migrated losslessly. Target and independent draft caches are both covered, so one-model MTP-EAGLE and EAGLE3 speculative decoding work with the codec enabled.

**Activation proof.** A parsed configuration is not evidence that any page was compressed, because a short request may never leave the GPU. The feature therefore exposes two counters on the `/metrics` endpoint, `trtllm_kv_cache_offload_bytes_total` and `trtllm_kv_cache_onboard_bytes_total`, which become nonzero only when pages cross a tier boundary. Because the counters cover all migrations, including lossless-fallback lifecycles, route-level evidence comes from a profiler trace: on a Qwen3-8B page-lineage probe, all 49 cold hits had a prior encode and a matching decode, and Nsight Systems showed the fused offload and onboard kernels on the compressed arm and none on the uncompressed arm. This rule ("configured is not proof") gates the accuracy runs and the GLM-5.2 serving wheel reported below. Live validation covers Qwen3 MHA/GQA on the host and disk paths, Qwen3.5 one-model MTP with target and draft caches, one-model EAGLE3 on host and disk, and an MLA model with FP8 hot KV under pipeline parallelism.

The concrete implementation can be found in `tensorrt_llm/_torch/kv_cache_compression/` and `cpp/tensorrt_llm/batch_manager/kv_cache_compression/`.

### TriAttention

#### Introduction

[TriAttention](https://arxiv.org/abs/2604.04921) (ICML 2026) is a training-free, decode-time KV cache eviction method for long-generation inference. During generation it periodically scores the cached tokens with a trigonometric importance measure derived from offline per-head query statistics, keeps the most important `budget` tokens, and physically compacts the cache, so that more sequences fit on a GPU at once. The prompt is always preserved; only the generation region is evicted. For technical details, please refer to the paper and to the official implementation at [github.com/WeianMao/triattention](https://github.com/WeianMao/triattention).

#### How It Works in TensorRT LLM

TriAttention is implemented as an iteration-driven `KVCacheCompressionManager` that overrides the generation-end hook. Every `beta` confirmed generation tokens, once a sequence exceeds its budget, the manager scores the evictable decode region with CuTe DSL and Triton kernels, selects the `budget` tokens to keep according to `eviction_mode` (`union`, `per_head`, or `per_layer_perhead`), and physically compacts the paged KV cache with a native CUDA kernel. A speculative iteration may confirm several tokens at once; crossing more than one eviction period in one update is coalesced into one eviction. The calibration file (each head's mean and magnitude of the pre-RoPE query, produced by the official tool) is loaded and converted once when the manager is created.

The method declares `changes_physical_kv_length = True` but still reports block reuse as supported, because it compacts only the generation suffix and preserves the committed prompt prefix that KVCM V2 reuses. There is no sparse-attention configuration and no custom attention backend; decode runs the model's standard attention kernel over the compacted cache. For calibration, configuration parameters, and current requirements, please refer to the [TriAttention example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md).

## Evaluation

This section consolidates accuracy and performance results for the KV cache compression methods supported in TensorRT LLM. All comparisons are against TensorRT LLM's own uncompressed configuration on the same hardware, wheel, and recipe; only the compression setting changes between arms.

### NVFP4 Cold-Page Quantization

Unless otherwise specified, the experiments below use GB300 GPUs, the PyTorch backend, native C++ KVCM V2, and block reuse enabled. "NVFP4" denotes the arm with cold-page NVFP4 enabled; the uncompressed arm's KV type and host tier are stated in each subsection. The accuracy runs are gated on nonzero offload and onboard page counts inside the scored window. The serving campaigns export no per-run migration counters; codec activation for the GLM-5.2 serving wheel is verified by a separate profiler probe on the same wheel, and the Qwen3.5-397B-A17B serving wheel has not been probed.

#### Accuracy

We evaluate accuracy on AIME25 with 16 seeds per arm, using each model's official sampling configuration (Qwen3.5-397B-A17B: 64,000 output tokens, temperature 0.6, top-p 0.95, top-k 20; Qwen3-8B: 38,912 output tokens). The protocol forces a controlled share of the reused prompt KV through the codec before scoring: prompts are first served to populate the cache, the GPU tier is flushed so that the pages are offloaded and compressed, and the scoring pass replays the prompts, onboarding and decoding the compressed pages. Decode KV is never compressed. "Medium" and "high" pressure correspond to roughly 28% and 61% of the reused prompt KV pages being NVFP4-compressed for Qwen3.5-397B-A17B, and approximately 30% and 60% for Qwen3-8B. Qwen3.5-397B-A17B runs NVFP4 weights with FP8 KV on four GB300 (TP=4, one engine); Qwen3-8B runs BF16 KV on one GB300. The baseline is the same model with all KV on the GPU in its runtime KV type (FP8 for Qwen3.5-397B-A17B, BF16 for Qwen3-8B) and no host tier.

| Model | Dataset | Uncompressed Baseline (runtime KV type) | Cold-Page NVFP4 (medium) | Cold-Page NVFP4 (high) |
| ------------------ | ------------------ | --------------------------------------- | ------------------------ | ---------------------- |
| Qwen3.5-397B-A17B | AIME25 (16 seeds) | 90.0 | 90.0 | 90.2 |
| Qwen3-8B | AIME25 (16 seeds) | 68.5 | 67.7 | 68.8 |

<div align="center">
<figure>
  <img src="../media/tech_blog28_accuracy_aime25.svg" width="900">
</figure>
</div>
<p align="center"><sub><em>Figure 4: AIME25 accuracy over 16 seeds for Qwen3-8B (left) and Qwen3.5-397B-A17B (right): uncompressed baseline, cold-page NVFP4 at medium and high pressure, and an every-step NVFP4 control that quantizes the live KV after each forward step. The band is the baseline mean plus or minus one standard deviation. The series labeled "All-NVFP4 KV cache" is the emulated every-step control described in the text, not the active NVFP4 KV cache feature; the figure footnote rounds both models' pressure shares to 30% and 60%.</em></sub></p>

Compared with the uncompressed baseline, no significant degradation is observed in the tested scope. The paired deltas are +0.00 pp (medium) and +0.21 pp (high) for Qwen3.5-397B-A17B, with standard errors of 0.68 and 0.57 pp, and -0.83 pp (medium) and +0.21 pp (high) for Qwen3-8B, with a standard error of 1.39 pp; all four are within one standard error of zero. Figure 4 also includes a directional control that quantizes the *live* KV to NVFP4 after every forward step: its deltas are -2.08 pp for Qwen3-8B and -0.42 pp for Qwen3.5-397B-A17B, inside seed noise at n=16 but consistently below the cold-page arms. Because no native FP4-KV decode kernel exists for these head sizes on the tested release, this control is a Torch quantize-dequantize emulation rather than a native kernel result, and we report it as directional only.

#### Performance

We benchmark NVFP4 cold pages against the uncompressed host tier on two models: GLM-5.2 (756B, MLA attention) and Qwen3.5-397B-A17B (hybrid attention plus GDN, NVFP4 weights). The workload is a 3,600-second replay of the InferenceX-derived AgentX 256k agentic trace, which has heavy prefix reuse across turns; these runs reuse the public trace and recipes but are not an official InferenceX submission. "Raw" denotes the uncompressed arm, whose host tier stores pages in FP8, the runtime KV type of both models. We sweep the published InferenceX recipes and a large set of derived configurations (concurrency, prefill/decode split, and GPU count) so that both arms are measured on the same grid: 54 matched configurations and 218 accepted runs (two repeats per arm) for GLM-5.2 on 8 to 48 GB300, and 131 matched configurations and 330 accepted runs (mostly single-repeat) for Qwen3.5-397B-A17B on 3 to 60 GB300. The host tier is 128 GiB per prefill rank in every run. We report **total token throughput per reserved GPU** against **P90 end-to-end normalized interactivity** (tokens per second per user, higher is better), the Pareto view in Figure 5.

<div align="center">
<figure>
  <img src="../media/tech_blog28_pareto.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 5: Throughput per reserved GB300 versus P90 interactivity for the uncompressed (Raw FP8) and NVFP4 host caches on GLM-5.2 (left) and Qwen3.5-397B-A17B (right). Each point is one configuration averaged over repeats; outlined points are the official InferenceX recipes; lines are the per-arm Pareto frontiers.</em></sub></p>

Figure 5 shows the throughput–interactivity Pareto frontiers for the two arms; curves further to the upper-right indicate better throughput at equivalent interactivity. On GLM-5.2 the NVFP4 frontier lies on or above the Raw frontier: it gains in the 30.7-51 tok/s/user band (median +12.1% over the 27 frontier vertices in the 30.7-133 overlap, maximum +28.5% at 36.1 tok/s/user) and coincides with Raw above roughly 51 tok/s/user, where both frontiers are formed by the same low-concurrency cells. Below 30.7 tok/s/user only NVFP4 has points, at 55,000 to 59,500 tok/s per GPU, and Raw has no tested point above 46,461 tok/s per GPU. On Qwen3.5-397B-A17B the two frontiers largely coincide, and NVFP4 separates from Raw only at the throughput-bound, low-interactivity end. We summarize the results using four metrics:

| Model | Workload | Peak Throughput/GPU Gain | Median Same-Config Gain | Cache-Read Hit Delta | TTFT p90 (median) |
| ------------------------------- | -------------------------- | ------------------------ | ----------------------- | -------------------------------------------------- | ----------------- |
| GLM-5.2 (MLA), 8-48 GB300 | AgentX 256k replay, 3600 s | +28.0% | +3.5% (54 configs) | +1.7 pp median (54 configs) | -32.5% (50 configs) |
| Qwen3.5-397B-A17B, 3-60 GB300 | AgentX 256k replay, 3600 s | n/a (single-repeat points; see text) | +0.9% (131 configs) | +0.3 pp median (131 configs) | -6.1% (131 configs) |

The +28.0% for GLM-5.2 is a frontier statement: the best NVFP4 point (24 GB300) against the best Raw point anywhere on the grid (also 24 GB300, at a different concurrency); the same +28.0% holds as the SLA-bounded best at P90 interactivity of at least 5 or 10 tok/s/user, and +25.8% at 25 tok/s/user. Across the 54 matched GLM-5.2 configurations the same-configuration medians are +3.5% in throughput per GPU and -32.5% in TTFT p90 (over the 50 configurations with TTFT p90 in both arms), with 25 of 54 configurations gaining more than 10%. For Qwen3.5-397B-A17B, the peak comparison (+6.0%, single-repeat points on 36 versus 28 GB300) and the SLA-bounded best (+5.4% at 10 tok/s/user) sit at the edge of the single-repeat noise band, so we report them as observations; the medians over 131 matched configurations (+0.9% throughput per GPU and -6.1% TTFT p90) are the representative figures.

To understand where the benefit comes from, it helps to look at the cache-read hit rate. The gain tracks how far the Raw arm's hit rate falls below the trace's prefix-reuse ceiling: where the uncompressed host tier holds the working set, both arms already reach cache-read hit rates of 93.6% to 97.9%, within about 3 pp of the ceiling (95.5% to 98.0%), and NVFP4 has nothing to recover; where the uncompressed tier thrashes, NVFP4 keeps more of the reusable prefix resident and throughput follows. On GLM-5.2 the correlation between the hit-rate delta and the per-GPU throughput gain over the 54 configurations is 0.959. The largest hit-rate delta, +34.2 pp (52.6% to 86.7%), comes from an under-provisioned 8-GPU cell where the Raw arm collapsed; it is one of 9 of the 54 configurations where Raw falls below half of the NVFP4 throughput, which we treat as robustness under pressure rather than as gains. At all 7 official InferenceX recipes for GLM-5.2 and all 6 for Qwen3.5-397B-A17B, NVFP4 is **neutral** (GLM-5.2 -0.27% to +0.84%, Qwen3.5 -0.48% to +1.13% throughput per GPU, within repeat noise), because those recipes are tuned so that the Raw host tier already holds the working set. Qwen3.5-397B-A17B stores its GDN state losslessly and its host tier is rarely the bottleneck at these recipes, which is why its medians are near zero.

Two qualifiers apply to every number above. The host tier is fixed at 128 GiB per prefill rank in all runs, so these campaigns measure what NVFP4 buys at a *given* host budget and make no claim about host memory saved at equal hit rate. And GPU counts differ across configurations, so peak comparisons mix compression and topology effects; the same-configuration medians do not.

### TriAttention

For TriAttention configuration, calibration workflow, validated modes, and evaluation, please refer to the [TriAttention example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md) and to [PR #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957).

## Summary and Future Work

### Current State

TensorRT LLM now provides a **unified KV cache compression framework** that supports two methods across two complementary lifecycle levels:

- **Framework-level**: One configuration and factory path (`KvCacheCompressionConfig`), one manager base class (`KVCacheCompressionManager`), and two standardized contracts: five lifecycle hooks for iteration-driven methods and the KVCM V2 cold-page codec ABI (`IKvCacheColdPageCodec`, fixed-size cold pages, `PageIndexPair` batches, enqueue-only encode/decode) for storage-bound methods. The configuration is orthogonal to active KV quantization and sparse attention; KVCM V2 retains ownership of pages, migration, reuse, and mapping publication.
- **Method-level**: NVFP4 cold-page quantization ships with encode and decode fused into the transfer, one compressed representation shared by the host and disk tiers, lossless handling of non-attention and side-buffer state, optional ModelOpt global scales, and support for MHA/MQA/GQA, key-only MLA, and DeepSeek-V4 layouts (DeepSeek-V4 support is in the current tree; see the feature documentation); it has been tested with the Qwen3, Qwen3.5, GLM, DeepSeek-R1, and DeepSeek-V4 families. TriAttention demonstrates the iteration-driven path with budget-triggered eviction during generation; it has been tested with the Qwen3, GPT-OSS, and Llama 3 families. The framework and both methods landed upstream in [PR #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957) (TriAttention), [PR #17512](https://github.com/NVIDIA/TensorRT-LLM/pull/17512) (cold-page codec support in KVCM V2), and [PR #18091](https://github.com/NVIDIA/TensorRT-LLM/pull/18091) (NVFP4 cold-page method).

A new compression method can be integrated by implementing one contract (hooks or codec) and its batched transform kernel, without modifying the attention kernels, the cache manager, or the serving infrastructure.

### Future Work

- **Higher-ratio cold-page formats**: The codec contract fixes only the cold-page size per lifecycle, so 2-bit and trellis-coded representations, as well as non-quantization codecs such as entropy coding or low-rank projection, can be added as new providers without changes to KVCM V2.
- **Native low-precision decode kernels**: No native FP4-KV decode kernel exists today for the head sizes used by the tested models, which is why the every-step control above is an emulation. Native kernels would allow active-KV quantization and cold-page compression to compose.
- **Host-tier sizing guidance**: A host-quota sweep at the official recipes, with exported migration counters, would turn the mechanism described above into deployment guidance.
- **Disaggregated serving co-design**: Cold-page compression covers the worker-local GPU-to-host and host-to-disk boundaries today, while the context-to-generation transfer moves the hot representation. Carrying the compressed representation across that link and into decode-side cold tiers is a natural extension.
- **More iteration-driven methods and hybrid-model state**: The five hooks are method-neutral; we expect further eviction and context-compression methods on them, and we are exploring a unified treatment of auxiliary and non-attention state so that hybrid models benefit beyond the attention share of their cache.
