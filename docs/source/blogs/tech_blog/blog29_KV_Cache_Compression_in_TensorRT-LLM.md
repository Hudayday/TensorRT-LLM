# KV Cache Compression in TensorRT LLM

**Table of Contents**
- [Introduction and Motivation](#introduction-and-motivation)
- [Overview of KV Cache Compression in TensorRT LLM](#overview-of-kv-cache-compression-in-tensorrt-llm)
- [KV Cache Compression Framework Design](#kv-cache-compression-framework-design)
  - [Design Philosophy](#design-philosophy)
  - [Architecture Overview](#architecture-overview)
  - [KV Cache Compression in the Executor Iteration Loop](#kv-cache-compression-in-the-executor-iteration-loop)
  - [KV Cache Compression in Cross-Request KV Management](#kv-cache-compression-in-cross-request-kv-management)
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

The tasks handed to Large Language Models (LLMs) keep getting more complex and more expensive to serve. Models grow, the text they read and write grows with them, and nowhere is this more visible than in agentic workflows, where a model works through a task in many steps instead of one reply.

An agent job is a chain of model calls separated by tool calls, and each tool result is appended to one growing conversation. Input length grows turn by turn, tool calls repeat dozens of times per job, and most of every prompt is text the system has already seen.

In the [InferenceX](https://inferencex.semianalysis.com/) AgentX coding traces, for example, the median input length per request is 14.4k tokens and over 96% of the prompt tokens are reusable prefix. Our earlier blog on [evaluating agentic serving with trace replay](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.html) characterizes these workloads and the role of KV cache reuse in detail. Figure 1 shows that the same shift is visible at the scale of a whole serving platform.

<div align="center">
<figure>
  <img src="../media/tech_blog29_workload_facts.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 1: LLM serving today. On a one-year production trace, requests are prompt-heavy, outputs are getting shorter, a single long context carries tens of gigabytes of KV, and almost all reuse arrives within minutes. In agentic coding traces, almost the entire prompt is reusable prefix. Sources: <a href="https://arxiv.org/abs/2608.13573">Nixon et al., A Year in LLM Serving (2026)</a> for the first six tiles, and our <a href="https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.html">trace-replay blog</a> for the last one.</em></sub></p>

This pressure grows while the space for the KV cache does not: GPU memory is fixed, and the DRAM behind it is finite too. When the KV cache no longer fits, the consequences are severe.

The most direct one is failing to serve: requests cannot be admitted until memory frees up. The next is slow serving, as the system spends its time evicting and refilling cache instead of computing. In agentic workloads especially, a shortage of storage means missing the opportunity to reuse KV: a prefix computed a moment ago is gone when the next turn arrives, so it is computed again, and that extra computation is paid on almost every turn of the job.

To serve these workloads well, a system has to keep more KV cache and keep it for longer, without adding memory. These characteristics create three opportunities that KV cache compression addresses directly:

- **The KV cache keeps growing.** Agent jobs produce far more KV than a GPU can hold, and the same prefix pages are needed again and again. Compressing the stored KV is the most direct way to keep more of it.
- **Serving already relies on host and disk tiers.** Prefixes that do not fit on the GPU are kept in host memory or on disk and brought back on the next turn. Compression lets those tiers hold more pages and moves fewer bytes between them.
- **All models face this pressure.** A method tied to one attention type or one KV data type helps only one deployment. Compression at the level of KV cache pages works for every model.

A wide range of KV cache compression methods have been proposed, from prompt compression and token eviction to low-precision storage, and TensorRT LLM already applies some of them inside the model: the active KV cache can be quantized, and the [sparse attention framework](blog17_Sparse_Attention_in_TensorRT-LLM.md) lets the attention kernel read only part of the cache. This blog is about the other places where compression can act: between prefill chunks, between decoding steps, around tool calls, and after the KV has left the GPU for host or disk memory.

Two things set our approach apart. First, we introduce concrete methods that compress the KV cache at these points successfully, with accuracy and serving results to back them. Second, and more importantly, the framework treats all of these points the same way: each moment in the life of a KV cache where compression can run is exposed as a well-defined attachment point, a method plugs into the points it needs, and the rest of the serving stack is untouched.

This is what makes the framework apply to any model, and it is also what lets future methods be added by plugging into an existing point rather than by modifying the serving loop or the attention kernels.

On this framework, TensorRT LLM currently ships two methods:

- **[NVFP4 cold-page quantization](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md#cold-page-quantization)**: keeps attention KV pages in NVFP4 only while they are cold, that is, while a page has left the GPU for host or disk memory. The conversion runs as part of the copy itself, and the page is restored to its original precision when it comes back to the GPU.
- **[TriAttention](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md#triattention)**: a training-free method that periodically scores the tokens generated so far and evicts the least useful ones between decoding steps once a sequence exceeds its token budget.

In the following sections, we first provide an overview of the KV cache compression capabilities in TensorRT LLM, then describe the framework design that makes them possible, walk through how each method is implemented on top of it, and finally present evaluation results.

## Overview of KV Cache Compression in TensorRT LLM

KV cache compression, as used in this blog, means shrinking the KV cache at any point in a workflow, however complex the workflow is. The goal is to cut KV cache pressure while keeping the information in the cache accurate.

Figure 2 shows where this sits in TensorRT LLM. It is one of three ways to cut memory and compute, next to quantization and sparse attention. It works on the KV pages that the cache manager owns.

<div align="center">
<figure>
  <img src="../media/tech_blog29_trtllm_stack.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 2: The TensorRT LLM stack. KV cache compression sits in the compression layer next to quantization and sparse attention. It changes what is stored in the KV cache and touches neither the kernels below nor the serving layers above.</em></sub></p>

KV cache compression in TensorRT LLM covers the whole life of a KV cache, not only the gap between two forward steps. It can run during inference, after prefill and between decode steps. It can also run when the system moves KV around, for example when pages are offloaded or transferred, and after a request ends and its KV is kept for reuse. The result is one optimization for the whole serving system rather than for one kernel.

In detail, we define five stages in the life of a KV cache, shown in Figure 3: the prefill-chunk stage, the after-prefill stage, the decode stage, the tool-call stage, and the after-request stage. Each stage is a moment when nothing is reading the cache, so compression can run there safely. Together the five stages cover every KV cache in the system, and they apply to any large language model.

<div align="center">
<figure>
  <img src="../media/tech_blog29_kv_lifetime_stages.svg" width="900">
</figure>
</div>
<p align="center"><sub><em>Figure 3: Five stages in the life of a KV cache where compression can run. TriAttention acts at the decode stage (stage 3) and cold-page compression at the after-request stage (stage 5). The dashed stages are in scope and have no method yet.</em></sub></p>

Working in stages has two benefits. A method picks only the stages it needs, and a stage works the same way for every model.

We defined the stages this way so that the framework needs one small contract per kind of stage and nothing more. Stages inside a request are reached by compression in the executor iteration loop. Stages beyond a single request, where the KV cache is kept and managed across requests, are reached through the KV cache manager, for example when pages are offloaded and onboarded.

In both cases the cache manager keeps full ownership of pages. It allocates them, moves them between tiers, and reuses them. A compression method only changes their contents.

We have built two methods on the framework, using two of the five stages. NVFP4 cold-page quantization works at the after-request stage and TriAttention at the decode stage. The same framework lets us add methods at the other stages and support more complex algorithms in the future. The two methods are:

*   **NVFP4 cold-page quantization**: stores attention KV pages as NVFP4 while they sit in host or disk memory. A page is converted on the way out and restored on the way back. The GPU cache and the attention kernels keep the model's normal KV data type.
*   **TriAttention**: runs between decoding steps. It scores the generated tokens with an importance measure calibrated offline for each attention head and keeps only a fixed budget of them. The prompt is never touched.

The two tables below summarize the current coverage.

<div align="center">

| Method | When It Runs | What It Changes | Supported Attention Types |
| :--- | :--- | :--- | :--- |
| **NVFP4 cold-page quantization** | After-request stage: when a page moves between the GPU and host or disk memory | How attention KV is stored while off the GPU | MHA / MQA / GQA; MLA; hybrid models (attention KV only) |
| **TriAttention** | Decode stage: periodically between decode steps | Which KV tokens are kept | MHA / MQA / GQA |

<p align="center"><sub><em>Table 1. The two methods built on the framework: the stage each one runs at, what it changes, and the attention types it supports.</em></sub></p>

</div>

<div align="center">

| Memory Tier | NVFP4 Cold-Page Quantization | TriAttention |
| :--- | :--- | :--- |
| **GPU (active KV)** | Unchanged (FP16 / BF16 / FP8) | Generated tokens evicted periodically; prompt kept |
| **Host memory** | NVFP4 (4-bit values with FP8 block scales) | Not affected |
| **Disk** | Same NVFP4 data as host memory, copied as is | Not affected |

<p align="center"><sub><em>Table 2. What each method does to the KV cache in each memory tier.</em></sub></p>

</div>

**Note**: Currently, this design targets and is validated on NVIDIA Blackwell GPUs (B200 and GB300).

This blog covers the framework design shared by all methods, with NVFP4 cold-page quantization as the main worked example. The C++ interface between the cache manager and a page encoder is documented in the [cold-page codec design guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-cold-page-codec.md). The APIs for adding a new method are in the [KV Cache Compression Development Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-compression-development.md).

## KV Cache Compression Framework Design

TensorRT LLM provides a general framework for KV cache compression, built around two ideas. First, users turn on a compression method with one designated compression config and nothing else. Second, developers add a new compression method on top of the framework with plain Python and a just-in-time compiled kernel, without touching the runtime. The framework takes care of the rest: when a method runs, how it reaches the KV cache, and how the cache manager keeps control of memory. The sections below describe how this works.

### Design Philosophy

The design starts from one observation: compression does not need to live inside the model or the attention kernel. It only needs to run at the right moment, on the KV that is already there. So the runtime pauses at a well-defined point, hands the KV cache to the compression method, and continues once the method returns. Compression is extra work inserted at the right points of the runtime, and the whole serving system benefits from the smaller cache.

Three principles follow from this.

- **Attention stays untouched.** Sparse attention changes how the attention kernel reads the cache. KV cache compression never does. Attention always reads the normal GPU representation, and a compressed page is restored before anything reads it.
- **Insertion is seamless.** A method plugs into the runtime without changes to the scheduler, the executor loop, or the model. It only changes the contents of KV pages. The cache manager keeps control of memory: it allocates pages, moves them between tiers, and reuses them.
- **Insertion points are well defined.** They follow the life of a request and of the server, the five stages of Figure 3: after each prefill chunk, right after prefill, between decode steps, around a tool call, and after the request. We call these points hooks, but the idea is simply a place in the runtime where a compression algorithm can be inserted and run to completion.

A compression method is selected with one configuration block, `kv_cache_compression_config`. It is kept separate from the KV cache configuration, which sets capacity, memory tiers, block reuse, and the active KV data type. It is also separate from the sparse attention configuration, which controls how attention computes. One method can be active per LLM instance. A method must handle every cache layout it is given, or pass the parts it does not understand through unchanged.

### Architecture Overview

The framework has three well-defined parts that together form KV cache compression (KVCC) in TensorRT LLM.

- **Compression config.** One configuration block collects everything the user asks for, validates it, and routes it to the concrete method. A factory then builds that method's manager before the model runs.
- **Compression manager base.** This base class is the core of the framework: it defines where in the runtime compression is injected. For a method that works between decode steps, such as TriAttention, the base provides the ability to run after every decode step. For a method that works on pages leaving the GPU, such as cold-page quantization, the base hooks into the KV cache manager and adds compression to offloading and onboarding. A concrete method inherits the base and is injected at the matching points automatically.
- **Method-specific kernels.** Each method brings its own kernels that compress and decompress the KV cache. The architecture lets a method define a new series of such kernels, optimize them, and fuse them with neighboring kernels, for example with the copy that moves a page off the GPU.

The executor and the KV cache manager are existing components of TensorRT LLM. The framework does not replace them. It interacts with them at the insertion points and leaves scheduling and memory ownership where they are.

<div align="center">
<figure>
  <img src="../media/tech_blog29_framework.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4: The KV cache compression framework and its execution order. The config builds the manager (1). The manager base inserts a hook into the executor and one into the KV cache manager (2). The concrete method inherits the base and runs inside those hooks (3), launching its own kernels (4). Highlighted boxes are the KVCC parts; white boxes are existing system components, with the stages each one hosts today and in the future.</em></sub></p>

Figure 4 introduces the whole framework and how it follows the lifetime of a KV cache. The framework interacts with different parts of the TensorRT LLM runtime so that compression can be injected at the five stages defined above. The executor iteration loop hosts the stages inside a request: the prefill-chunk, after-prefill, and decode stages. The KV cache manager hosts the stages beyond a single request: the tool-call and after-request stages, when the KV cache is kept and managed across requests.

Some of these stages have a method today and some do not. The decode stage is implemented by TriAttention. The after-request stage is implemented in part by cold-page compression, which handles the pages that leave the GPU for host or disk memory. The rest of this section walks through the framework in detail: how each path works, how the other stages will be covered, and how a user configures it.

### KV Cache Compression in the Executor Iteration Loop

This path serves the stages inside a request: the prefill-chunk stage, the after-prefill stage, and the decode stage. Today the decode stage is the one with a shipped method. While a prefill or decode step runs, attention reads a fixed view of the KV cache. Between two steps that view can change, and that is where the hooks fire. A method overrides only the hooks it needs, and all of them do nothing by default.

The way in is simple. The executor already keeps a list of resource managers and calls every one of them at fixed points of each iteration: before the forward pass, after it, and when a request ends. The compression manager base inherits the same resource manager class and is registered as the last one in that list, so it runs after the KV cache manager has updated its state. Inside those three callbacks it fires the five hooks, as Figure 5 shows. A method inherits the base and overrides only the hooks it needs; **TriAttention** overrides the one after each decode step.

<div align="center">
<figure>
  <img src="../media/tech_blog29_hooks_flow.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 5: How the hooks reach the executor iteration loop. Every BaseResourceManager receives three callbacks per iteration: before the forward pass, after it, and when a request ends. KVCacheCompressionManager inherits BaseResourceManager and turns the three callbacks into the five hooks. A method overrides the hooks it needs.</em></sub></p>

<div align="center">

| Hook | When it fires | Stage it serves |
| :--- | :--- | :--- |
| `on_request_init` | Before a request's first prefill chunk | Prefill-chunk stage, set-up before the first chunk |
| `on_context_step_end` | After a request's last prefill chunk | After-prefill stage |
| `on_generation_step_begin` | Before each decode step | Decode stage |
| `on_generation_step_end` | After each decode step, once the cache has been updated | Decode stage (used by **TriAttention**) |
| `on_request_finish` | When a request completes or is aborted | Tool-call and after-request stages, when a request pauses or ends |

<p align="center"><sub><em>Table 3. The five hooks in the executor iteration loop, when each one fires, and the stage it serves.</em></sub></p>

</div>

We currently define these five hooks. **TriAttention** uses the generation-end hook, so the decode stage is the one exercised by a shipped method. The other hooks are already in place for the prefill-chunk and after-prefill stages and for the request-level events of the tool-call and after-request stages, and a new method can use them without changes to the framework.

The hooks ride on the executor's existing request cycle, and the framework wires them up. Methods on this path change which tokens are kept or how they are arranged in the paged cache. Two obligations come with it. The policy that chooses tokens stays separate from the shared compaction kernel. And a method must finish its GPU work before it shrinks or frees any cache pages.

### KV Cache Compression in Cross-Request KV Management

This path is the cross-request hook of the framework. It serves the stages outside a single request: the tool-call stage, when a request pauses and its KV waits for the tool to return, and the after-request stage, when a request has finished and its KV is kept for reuse, transferred to another worker, or offloaded to host or disk memory.

Today we cover the offloading part of the after-request stage with **NVFP4 cold-page quantization**, enabled by `quantization_for_cold_page` with `quant: nvfp4`. A page is encoded as it is copied from the GPU to host or disk memory, and decoded into the normal GPU representation when it comes back, before anything reads it. Figure 6 follows one page out and back. The cache manager steps are the same for any codec.

<div align="center">
<figure>
  <img src="../media/tech_blog29_nvfp4_cold_page_pipeline.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 6: One page leaving and returning to the GPU. When a request ends, the cache manager evicts its pages and hands them to the encode kernel, which writes compressed pages into host or disk memory. When the request resumes, the decode kernel restores the pages in their original data type before attention reads them.</em></sub></p>

Under the hood, the contract between the cache manager and a codec is small. A GPU page may consist of several buffers: K and V, an MLA latent buffer plus an index buffer, or the mixed buffers of a hybrid model. A compressed page is one fixed-size block of bytes. The codec converts between the two, and four properties keep it simple:

- **Fixed compressed size.** Each layer group has one compressed page size. Every tier can then address pages with a plain slot allocator.
- **Batched calls.** The cache manager hands the codec a whole batch of page indices and a stream, not one page at a time.
- **Asynchronous by design.** Encode and decode only enqueue GPU work on that stream. The cache manager tracks completion and owns the events, so the codec inherits the same ordering and failure handling as an ordinary copy.
- **Compressed pages stay compressed between tiers.** Moves between host and disk copy the bytes as they are. Only the GPU boundary runs the codec.

Compression is optional at this layer. The default codec simply packs the page buffers into the block at full size. A compressing codec produces a smaller block through the same path. It can also mark some buffers as lossless, so the recurrent state of hybrid models or side buffers such as an MLA index pass through unchanged inside the same page. Adding a new format means adding a kernel and its layout description. Tier routing, staging, batching, and ordering are shared. The C++ interface and the Python API are documented in the [cold-page codec design guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-cold-page-codec.md). The tool-call stage and the rest of the after-request stage, such as KV transfer, will be reached by extending this path.

### Covering the Other Stages

Together, the two paths cover the decode stage and one part of the after-request stage, which is what the two shipped methods need. The base class keeps the ability to define more hooks, so every stage that offers a compression opportunity can be reached the same way. How those hooks look is future work, and we will extend the framework as new methods need them.

### Configuration and Ownership

The framework spans the executor, the cache manager, and native kernels. Its most important decision is who owns what:

- The **configuration and factory** own method selection and compatibility checks. An unsupported combination is rejected before anything is built. Nothing falls back silently at run time.
- The **compression manager** owns the method's cadence, its per-request state, its format metadata, and its kernel launches.
- The **cache manager** owns pages, tiers, migration, reuse, and the ordering of copies.
- The **attention backend** owns the active GPU representation it reads, and nothing else.

Configuration is opt-in and small. Cold-page quantization is turned on with `algorithm: quantization_for_cold_page` and `quant: nvfp4` under `kv_cache_compression_config`. It also needs a host tier under `kv_cache_config`, set with `host_cache_size` and optionally `disk_cache_size` and `disk_cache_path`. The active cache keeps its normal data type. TriAttention is turned on with `algorithm: triattention` plus `budget`, `beta`, `eviction_mode`, and `calibration_path`. The same fields work in the Python API, in `trtllm-serve` YAML, and in `trtllm-bench`. Calibration files are validated at start-up, and the runtime fails fast if they are missing or incompatible. The full option table is in the [KV Cache Compression feature documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md).

## Algorithm Implementations

In this section, we walk through the two KV cache compression methods currently implemented in TensorRT LLM, focusing on how each method works and how it integrates with the framework. For a quick-start guide and runnable configurations, please refer to the [KV cache compression examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/kv_cache_compression).

### NVFP4 Cold-Page Quantization

#### Introduction

NVFP4 is a 4-bit floating-point format (E2M1) in which every group of 16 values shares one FP8 (E4M3) block scale, optionally combined with a per-tensor global scale. TensorRT LLM already uses it for NVFP4 weights and for the NVFP4 active KV cache, so its rounding behavior and conversion kernels are well established. Cold-page quantization applies the same format at a different point in the system: instead of keeping the *active* KV cache in NVFP4 and asking the attention kernel to read it, it keeps attention KV in NVFP4 **only while a page is in host or disk memory**. The GPU cache and the attention kernels keep the model's normal KV data type (FP16, BF16, or FP8).

Host and disk memory are the right place for this transform for three reasons. In a reuse-heavy deployment the quantities that bind are the bytes *stored* off the GPU and the bytes *moved* over PCIe and storage links, so a smaller page buys more retained pages per gigabyte and fewer bytes per transfer. Because the active KV is never quantized, no error accumulates step after step, and a page that never leaves the GPU is never touched. And the transform is invisible to everything above it: page identity, token identity, block reuse, scheduling, and the GPU layout that attention sees are unchanged.

The arithmetic is simple. Packed E2M1 data plus one E4M3 scale per 16 values costs 0.5625 bytes per value, against 2 bytes for FP16 or BF16 and 1 byte for FP8. After the alignment the layout imposes, the measured page sizes are:

- **Models with FP8 active KV**: 43.75% fewer bytes per attention page off the GPU (Qwen3.5-397B-A17B 1,048,576 -> 589,824 B; Qwen3-8B with FP8 KV 2,359,296 -> 1,327,104 B), that is 1.78x more pages per byte of host or disk memory.
- **MLA latent pages (GLM-5.2)**: 41.1% fewer bytes (3,098,112 -> 1,824,000 B), or 1.70x; the ratio is lower because the index buffer in the same page is kept lossless.
- **Models with BF16 active KV**: 71.9% fewer bytes (Qwen3-8B with BF16 KV 4,718,592 -> 1,327,104 B), or 3.56x.

These figures are for attention KV. The recurrent-state buffers of hybrid models (Gated DeltaNet, state-space, and convolution state) are stored as they are and do not shrink.

#### How It Works in TensorRT LLM

Within TensorRT LLM, NVFP4 cold-page quantization is a compression manager that plugs into the cross-request KV management path of the framework. Below we highlight the key design choices in layout, kernels, scales, and verification.

**Layout policy.** Enabling `quantization_for_cold_page` with `quant: nvfp4` selects the manager, which is built before the cache manager because the cache manager needs the compressed page size to lay out its host and disk tiers. At start-up the manager receives the description of every GPU page buffer once and builds a layout table per layer group: for MHA, MQA, and GQA pages the K and V buffers become NVFP4 data plus block scales; for MLA pages the latent attention key is encoded; side buffers in the same page, such as the index buffer of a sparse-attention model, are appended as they are; and the recurrent-state buffers of hybrid models are routed to the default lossless path. For DeepSeek-V4, the part of the attention history without positional encoding is encoded as NVFP4 while the positional part and the remaining specialized state are kept losslessly in the same page. A GPU page may span several buffers in the normal KV type; the compressed page is one fixed-size block: the packed NVFP4 spans, then their E4M3 block scales, then the lossless spans, each 16-byte aligned.

**Fused encode and decode kernels.** Encoding and transfer are one operation. The offload kernel reads the GPU page, quantizes it, and writes the packed page straight into the mapped host slot; the onboard kernel reads the compressed page from host memory, dequantizes it, and writes the full-precision page into the GPU pool. Each launch takes the layout table, a batch of page indices, the destination base address, and the stream the cache manager supplies, with no per-page host work. There is **no model-specific kernel**: the same launcher covers MHA, GQA, MQA, MLA, and the DeepSeek-V4 layout, driven entirely by the layout table; adding DeepSeek-V4 support meant a new layout policy and its CUDA operator, with no change to the cache manager or to the shared codec path. Host and disk share the encoded bytes, so moving a page to disk never re-quantizes it.

**Scales and calibration.** By default the codec uses identity global scales, so a regular FP16, BF16, or FP8 checkpoint needs no calibration step; the block scale for each group of 16 values is computed during encode and stored with the page. A compatible NVIDIA ModelOpt NVFP4 checkpoint can optionally supply per-layer K and V global scales through `scale_checkpoint_path`; this applies to the K/V layout, while MLA pages and draft-model pages use identity scales. Weight quantization and KV compression are independent: a checkpoint with NVFP4 weights and FP8 active KV uses this feature as usual, and a model whose active KV is already NVFP4 is moved losslessly. Target and draft caches are both covered, so one-model MTP-EAGLE and EAGLE3 speculative decoding work with the codec enabled.

**Verifying that it is active.** A parsed configuration does not prove that any page was compressed, because a short request may never leave the GPU. The feature therefore exposes two counters on the `/metrics` endpoint, `trtllm_kv_cache_offload_bytes_total` and `trtllm_kv_cache_onboard_bytes_total`, which become nonzero only when pages cross a tier boundary. Because the counters count every migration, including lossless ones, a profiler trace is the definitive check: with the codec enabled, Nsight Systems shows the fused offload and onboard kernels on the timeline, and every page read back from host memory has a matching encode and decode. All accuracy and serving results in this blog were gated on this check. Functional coverage includes Qwen3 MHA and GQA models on the host and disk paths, Qwen3.5 with one-model MTP (target and draft caches), one-model EAGLE3 on host and disk, and an MLA model with FP8 active KV under pipeline parallelism.

The concrete implementation can be found in `tensorrt_llm/_torch/kv_cache_compression/` (the manager and layout policy), `cpp/tensorrt_llm/batch_manager/kv_cache_compression/` (the native codec path), `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu` (the fused encode and decode kernels), and `cpp/tensorrt_llm/nanobind/kvCacheCompression/bindings.cpp` (the Python-to-native bridge).

### TriAttention

#### Introduction

[TriAttention](https://arxiv.org/abs/2604.04921) (ICML 2026) is a training-free KV cache eviction method for long generations. During decoding it periodically scores the cached tokens with a trigonometric importance measure derived from per-head query statistics collected offline, keeps the most important `budget` tokens, and physically compacts the cache, so that more sequences fit on a GPU at once. The prompt is always preserved; only generated tokens are evicted. For technical details, please refer to the paper and to the official implementation at [github.com/WeianMao/triattention](https://github.com/WeianMao/triattention).

#### How It Works in TensorRT LLM

TriAttention is a compression manager for compression in the executor iteration loop. It overrides the generation-end hook. Every `beta` confirmed generation tokens, once a sequence exceeds its budget, the manager scores the evictable generated tokens with CuTe DSL and Triton kernels, selects the `budget` tokens to keep according to `eviction_mode` (`union`, `per_head`, or `per_layer_perhead`), and compacts the paged KV cache in place with a native CUDA kernel. A speculative step may confirm several tokens at once; crossing more than one eviction period in one step is coalesced into one eviction. The calibration file (each head's mean and magnitude of the pre-RoPE query, produced by the official tool) is loaded and converted once when the manager is created. Figure 7 follows one eviction round from the hook to the compacted cache.

<div align="center">
<figure>
  <img src="../media/tech_blog29_triattention_pipeline.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 7: TriAttention between two decode steps: the generation-end hook picks the due requests, a fused CuTe DSL kernel scores the generated tokens from calibration statistics, the scores are reduced per eviction mode and selected with a radix top-k, and a native compaction kernel moves K and V in place before the manager reports the new cache length and returns the freed pages.</em></sub></p>

The method changes the physical cache length but keeps block reuse valid, because it compacts only the generated suffix and preserves the prompt prefix that the cache manager reuses. There is no sparse-attention configuration and no custom attention backend; decoding runs the model's standard attention kernel over the compacted cache. For calibration, configuration parameters, and current requirements, please refer to the [TriAttention example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md).

## Evaluation

This section consolidates accuracy and performance results for the KV cache compression methods supported in TensorRT LLM. Every comparison is against TensorRT LLM's own uncompressed configuration on the same hardware, software build, and serving configuration; only the compression setting differs.

### NVFP4 Cold-Page Quantization

Unless otherwise specified, the experiments below use GB300 GPUs, the PyTorch backend, the paged KV cache manager, and block reuse enabled. Before any result was accepted we confirmed that pages were actually compressed during the measured window, using the counters and profiler trace described above.

#### Accuracy

We evaluate accuracy on AIME25 with 16 seeds per configuration, using each model's official sampling settings (Qwen3.5-397B-A17B: 64,000 output tokens, temperature 0.6, top-p 0.95, top-k 20; Qwen3-8B: 38,912 output tokens). The protocol pushes a controlled share of the reused prompt KV through the codec before scoring: the prompts are served once to populate the cache, part of the GPU cache is flushed so that those pages are moved to host memory and compressed, and the scoring pass serves the prompts again, reading the compressed pages back. Generated tokens are never compressed. "Medium" and "high" pressure mean that roughly 30% and 60% of the reused prompt KV pages had been NVFP4-compressed (measured: 28% and 61% for Qwen3.5-397B-A17B). Qwen3.5-397B-A17B runs NVFP4 weights with FP8 KV on four GB300 (tensor parallel 4); Qwen3-8B runs BF16 KV on one GB300. The baseline is the same model with all KV on the GPU in its normal KV type and no host tier.

| Model | Dataset | Uncompressed baseline | Cold-page NVFP4, medium pressure | Cold-page NVFP4, high pressure |
| ------------------ | ------------------ | --------------------- | -------------------------------- | ------------------------------ |
| Qwen3.5-397B-A17B | AIME25 (16 seeds) | 90.0 | 90.0 | 90.2 |
| Qwen3-8B | AIME25 (16 seeds) | 68.5 | 67.7 | 68.8 |

<p align="center"><sub><em>Table 4. AIME25 accuracy (%) of the uncompressed baseline and of cold-page NVFP4 at medium and high pressure, mean over 16 seeds.</em></sub></p>

<div align="center">
<figure>
  <img src="../media/tech_blog29_accuracy_aime25.svg" width="900">
</figure>
</div>
<p align="center"><sub><em>Figure 8: AIME25 accuracy over 16 seeds for Qwen3-8B (left) and Qwen3.5-397B-A17B (right): uncompressed baseline, cold-page NVFP4 at medium and high pressure, and an every-step NVFP4 control that quantizes the active KV after each forward step. The band is the baseline mean plus or minus one standard deviation.</em></sub></p>

Compared with the uncompressed baseline, no significant degradation is observed in the tested scope. The paired differences are +0.00 and +0.21 percentage points (medium and high) for Qwen3.5-397B-A17B, with standard errors of 0.68 and 0.57, and -0.83 and +0.21 for Qwen3-8B, with a standard error of 1.39; all four are within one standard error of zero. Figure 8 also includes a control that quantizes the *active* KV to NVFP4 after every forward step: it lands 2.08 points below the baseline for Qwen3-8B and 0.42 below for Qwen3.5-397B-A17B, inside the seed noise at 16 seeds but consistently under the cold-page results. Because no native FP4 KV decode kernel exists for these head sizes on the tested release, this control is a quantize-dequantize emulation in PyTorch rather than a native kernel result, and we report it as directional only.

#### Performance

We benchmark the NVFP4 host cache against the uncompressed host cache on two models: GLM-5.2 (756B, MLA attention) and Qwen3.5-397B-A17B (hybrid attention plus Gated DeltaNet, NVFP4 weights). The workload is a 3,600-second replay of the InferenceX AgentX 256k agentic trace, which has heavy prefix reuse across turns; these runs use the public trace and the published InferenceX configurations but are not an official InferenceX submission. In the uncompressed configuration the host tier stores pages in FP8, the normal KV type of both models. We sweep the published configurations and a large set of derived ones (concurrency, prefill/decode split, and GPU count) so that both settings are measured on the same grid: 54 matched configurations and 218 accepted runs (two repeats each) for GLM-5.2 on 8 to 48 GB300, and 131 matched configurations and 330 accepted runs (mostly single repeat) for Qwen3.5-397B-A17B on 3 to 60 GB300. The host tier is 128 GiB per prefill rank in every run. We report **total token throughput per reserved GPU** against **P90 end-to-end normalized interactivity** (tokens per second per user, higher is better), the Pareto view in Figure 9.

<div align="center">
<figure>
  <img src="../media/tech_blog29_pareto.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 9: Throughput per reserved GB300 versus P90 interactivity for the uncompressed (FP8) and NVFP4 host caches on GLM-5.2 (left) and Qwen3.5-397B-A17B (right). Each point is one configuration averaged over repeats; outlined points are the published InferenceX configurations; lines are the Pareto frontiers of the two settings.</em></sub></p>

Figure 9 shows the throughput-interactivity Pareto frontiers; curves further to the upper right deliver more throughput at the same interactivity. On GLM-5.2 the NVFP4 frontier lies on or above the uncompressed one: it gains in the 31 to 51 tokens/s/user band (median +12.1% across the frontier points in the overlapping range, at most +28.5% at 36 tokens/s/user) and coincides with it above roughly 51 tokens/s/user, where both frontiers are formed by the same low-concurrency configurations. Below 31 tokens/s/user only NVFP4 has points, at 55,000 to 59,500 tokens/s per GPU, and the uncompressed cache has no tested point above 46,461 tokens/s per GPU. On Qwen3.5-397B-A17B the two frontiers largely coincide, and NVFP4 pulls ahead only at the throughput-bound, low-interactivity end. We summarize the results with four metrics:

| Model | Workload | Peak throughput/GPU gain | Median gain at the same configuration | Cache-read hit rate | TTFT p90 (median) |
| ------------------------------- | -------------------------- | ------------------------ | ------------------------------------- | ------------------------ | ----------------- |
| GLM-5.2 (MLA), 8-48 GB300 | AgentX 256k replay, 3600 s | +28.0% | +3.5% (54 configurations) | +1.7 pp median | -32.5% (50 configurations) |
| Qwen3.5-397B-A17B, 3-60 GB300 | AgentX 256k replay, 3600 s | not reported (single-repeat points; see text) | +0.9% (131 configurations) | +0.3 pp median | -6.1% (131 configurations) |

<p align="center"><sub><em>Table 5. Serving gains of the NVFP4 host cache over the uncompressed host cache on the AgentX 256k replay.</em></sub></p>

The +28.0% for GLM-5.2 compares the best NVFP4 point (24 GB300) with the best uncompressed point anywhere on the grid (also 24 GB300, at a different concurrency); the same +28.0% holds for the best throughput at a P90 interactivity of at least 5 or 10 tokens/s/user, and +25.8% at 25 tokens/s/user. Across the 54 matched GLM-5.2 configurations the medians at the same configuration are +3.5% in throughput per GPU and -32.5% in TTFT p90 (over the 50 configurations with TTFT p90 in both settings), and 25 of 54 configurations gain more than 10%. For Qwen3.5-397B-A17B, the peak comparison (+6.0%, single-repeat points on 36 versus 28 GB300) and the best throughput at 10 tokens/s/user (+5.4%) sit at the edge of single-repeat noise, so we report them as observations; the medians over 131 matched configurations (+0.9% throughput per GPU, -6.1% TTFT p90) are the representative figures.

To understand where the benefit comes from, look at the cache-read hit rate, the share of KV that attention reads from the cache instead of recomputing. The gain tracks how far the uncompressed host cache falls below the maximum hit rate the trace allows: where the uncompressed host tier already holds the working set, both settings reach hit rates of 93.6% to 97.9%, within about 3 points of that maximum (95.5% to 98.0%), and NVFP4 has nothing to recover; where the uncompressed tier thrashes, NVFP4 keeps more of the reusable prefix resident and throughput follows. On GLM-5.2 the correlation between the hit-rate difference and the throughput gain over the 54 configurations is 0.959. The largest hit-rate difference, +34.2 points (52.6% to 86.7%), comes from an 8-GPU configuration too small for the uncompressed cache, which collapsed there; it is one of 9 of the 54 configurations where the uncompressed setting falls below half of the NVFP4 throughput, which we treat as robustness under pressure rather than as a gain to quote. At all 7 published InferenceX configurations for GLM-5.2 and all 6 for Qwen3.5-397B-A17B, NVFP4 is **neutral** (GLM-5.2 -0.27% to +0.84%, Qwen3.5-397B-A17B -0.48% to +1.13% throughput per GPU, within repeat noise), because those configurations are tuned so that the uncompressed host tier already holds the working set. Qwen3.5-397B-A17B stores its recurrent state losslessly and its host tier is rarely the bottleneck in these configurations, which is why its medians are near zero.

Two qualifiers apply to every number above. The host tier is fixed at 128 GiB per prefill rank in all runs, so these measurements show what NVFP4 buys at a *given* host budget and make no claim about host memory saved at equal hit rate. And GPU counts differ across configurations, so peak comparisons mix compression and topology effects; the same-configuration medians do not.

### TriAttention

For TriAttention configuration, calibration workflow, validated modes, and evaluation, please refer to the [TriAttention example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md) and to [PR #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957).

## Summary and Future Work

### Current State

TensorRT LLM now has one KV cache compression framework and two methods built on it:

- **Framework**: one configuration block, `kv_cache_compression_config`, selects a method and is checked for compatibility before anything is built. A method implements one of two small contracts: hooks that run in the executor iteration loop, or a page encoder and decoder that run when pages move between the GPU and the host or disk tiers. Attention kernels never see a compressed page, and the cache manager keeps ownership of pages, tiers, migration, and reuse. Compression is independent of the active KV data type and of sparse attention.
- **NVFP4 cold-page quantization**: attention pages are stored as NVFP4 while they are in host or disk memory and restored to the model's normal KV data type before attention reads them. Encoding and decoding are fused into the copy itself, host and disk share one encoded form, and the recurrent state of hybrid models and side buffers such as an MLA index pass through losslessly in the same page. It covers MHA, MQA, GQA, MLA, and DeepSeek-V4 cache layouts and has been tested with the Qwen3, Qwen3.5, GLM, DeepSeek-R1, and DeepSeek-V4 families.
- **TriAttention**: demonstrates compression in the executor iteration loop by evicting generated tokens during decoding, with the model's standard attention kernel running over the compacted cache. It has been tested with the Qwen3, GPT-OSS, and Llama 3 families.

The framework and both methods are upstream in [PR #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957) (TriAttention), [PR #17512](https://github.com/NVIDIA/TensorRT-LLM/pull/17512) (page encoder and decoder support in the cache manager), and [PR #18091](https://github.com/NVIDIA/TensorRT-LLM/pull/18091) (NVFP4 cold-page quantization). A new method needs one contract and its kernel, and no change to the attention kernels, the cache manager, or the serving loop.

### Future Work

- **Smaller cold-page formats.** The page contract fixes only the compressed page size, so 2-bit formats, entropy coding, or low-rank projection can be added as new codecs without changes to the cache manager.
- **Native low-precision decode kernels.** No attention kernel today reads a 4-bit KV cache for the head sizes of the tested models, which is why the every-step NVFP4 setting in the Evaluation section is emulated. Native kernels would let an NVFP4 active cache and NVFP4 cold pages work together.
- **Sizing guidance for the host tier.** The serving results above used one host tier size. A sweep over host tier sizes at the published InferenceX configurations, with the offload and onboard counters recorded, would show how much host memory a compressed cache actually needs.
- **Disaggregated serving.** Cold-page compression covers the GPU-to-host and host-to-disk boundaries inside one worker today. Carrying the encoded page across the transfer from the prefill worker to the decode worker, and into the decode worker's host tier, is the natural next step.
- **More methods in the executor iteration loop and hybrid-model state.** The hooks are not tied to TriAttention, and we expect further eviction and context-compression methods to use them. We are also exploring compression for the recurrent state of hybrid models, which today passes through losslessly.
