# KV Cache Compression in TensorRT LLM

**Table of Contents**
- [Introduction and Motivation](#introduction-and-motivation)
- [Overview of KV Cache Compression in TensorRT LLM](#overview-of-kv-cache-compression-in-tensorrt-llm)
- [KV Cache Compression Framework Design](#kv-cache-compression-framework-design)
  - [Design Philosophy](#design-philosophy)
  - [Architecture Overview](#architecture-overview)
  - [Configuration and Integration with the Cache Manager](#configuration-and-integration-with-the-cache-manager)
  - [KV Cache Compression in the Executor Iteration Loop](#kv-cache-compression-in-the-executor-iteration-loop)
  - [KV Cache Compression in Cross-Request KV Management](#kv-cache-compression-in-cross-request-kv-management)
  - [Covering the Other Stages](#covering-the-other-stages)
- [Algorithm Implementations](#algorithm-implementations)
  - [NVFP4 Cold-Page Compression](#nvfp4-cold-page-compression)
  - [TriAttention](#triattention)
- [Evaluation](#evaluation)
  - [NVFP4 Cold-Page Compression](#nvfp4-cold-page-compression-1)
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

- **[NVFP4 cold-page compression](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md#cold-page-quantization)**: keeps attention KV pages in NVFP4 only while they are cold, that is, while a page has left the GPU for host or disk memory. The conversion runs as part of the copy itself, and the page is restored to its original precision when it comes back to the GPU.
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

In detail, we define five stages in the life of a KV cache, shown in Figure 3: the prefill-chunk stage (stage 1), the after-prefill stage (stage 2), the decode stage (stage 3), the tool-call stage (stage 4), and the after-request stage (stage 5). Each stage is a moment when nothing is reading the cache, so compression can run there safely. Together the five stages cover every KV cache in the system, and they apply to any large language model.

<div align="center">
<figure>
  <img src="../media/tech_blog29_kv_lifetime_stages.svg" width="900">
</figure>
</div>
<p align="center"><sub><em>Figure 3: Five stages in the life of a KV cache where compression can run. TriAttention acts at the decode stage (stage 3) and cold-page compression at the after-request stage (stage 5). The dashed stages are in scope and have no method yet.</em></sub></p>

Working in stages has two benefits. A method picks only the stages it needs, and a stage works the same way for every model.

We defined the stages this way so that the framework needs one small contract per kind of stage and nothing more. Stages inside a request are reached by compression in the executor iteration loop. Stages beyond a single request, where the KV cache is kept and managed across requests, are reached through the KV cache manager, for example when pages are offloaded and onboarded.

In both cases the cache manager keeps full ownership of pages. It allocates them, moves them between tiers, and reuses them. A compression method only changes their contents.

We have built two methods on the framework, using two of the five stages. NVFP4 cold-page compression works at the after-request stage (stage 5) and TriAttention at the decode stage (stage 3). The same framework lets us add methods at the other stages and support more complex algorithms in the future. The two methods are:

*   **NVFP4 cold-page compression**: stores attention KV pages as NVFP4 while they sit in host or disk memory. A page is converted on the way out and restored on the way back. The GPU cache and the attention kernels keep the model's normal KV data type.
*   **TriAttention**: runs between decoding steps. It scores the generated tokens with an importance measure calibrated offline for each attention head and keeps only a fixed budget of them. The prompt is never touched.

The two tables below summarize the current coverage.

<div align="center">

| Method | When It Runs | What It Changes | Supported Attention Types |
| :--- | :--- | :--- | :--- |
| **NVFP4 cold-page compression** | After-request stage (stage 5): when a page moves between the GPU and host or disk memory | How attention KV is stored while off the GPU | MHA / MQA / GQA; MLA; hybrid models (attention KV only) |
| **TriAttention** | Decode stage (stage 3): periodically between decode steps | Which KV tokens are kept | MHA / MQA / GQA |

<p align="center"><sub><em>Table 1. The two methods built on the framework: the stage each one runs at, what it changes, and the attention types it supports.</em></sub></p>

</div>

<div align="center">

| Memory Tier | NVFP4 Cold-Page Compression | TriAttention |
| :--- | :--- | :--- |
| **GPU (active KV)** | Unchanged (FP16 / BF16 / FP8) | Generated tokens evicted periodically; prompt kept |
| **Host memory** | NVFP4 (4-bit values with FP8 block scales) | Not affected |
| **Disk** | Same NVFP4 data as host memory, copied as is | Not affected |

<p align="center"><sub><em>Table 2. What each method does to the KV cache in each memory tier.</em></sub></p>

</div>

**Note**: Currently, this design targets and is validated on NVIDIA Blackwell GPUs (B200 and GB300).

This blog covers the framework design shared by all methods, with NVFP4 cold-page compression as the main worked example. The C++ interface between the cache manager and a page encoder is documented in the [cold-page codec design guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-cold-page-codec.md). The APIs for adding a new method are in the [KV Cache Compression Development Guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-compression-development.md).

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
- **Compression manager base.** This base class is the core of the framework: it defines where in the runtime compression is injected. For a method that works between decode steps, such as TriAttention, the base provides the ability to run after every decode step. For a method that works on pages leaving the GPU, such as cold-page compression, the base hooks into the KV cache manager and adds compression to offloading and onboarding. A concrete method inherits the base and is injected at the matching points automatically.
- **Method-specific kernels.** Each method brings its own kernels that compress and decompress the KV cache. The architecture lets a method define a new series of such kernels, optimize them, and fuse them with neighboring kernels, for example with the copy that moves a page off the GPU.

The executor and the KV cache manager are existing components of TensorRT LLM. The framework does not replace them. It interacts with them at the insertion points and leaves scheduling and memory ownership where they are.

<div align="center">
<figure>
  <img src="../media/tech_blog29_framework.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 4: The KV cache compression framework and its execution order. The config builds the manager (1). The manager base inserts a hook into the executor and one into the KV cache manager (2). The concrete method inherits the base and runs inside those hooks (3), launching its own kernels (4). Highlighted boxes are the KVCC parts; white boxes are existing system components, with the stages each one hosts today and in the future.</em></sub></p>

Figure 4 introduces the whole framework and how it follows the lifetime of a KV cache. The framework interacts with different parts of the TensorRT LLM runtime so that compression can be injected at the five stages defined above. The executor iteration loop hosts the stages inside a request: the prefill-chunk, after-prefill, and decode stages (stages 1 to 3). The KV cache manager hosts the stages beyond a single request: the tool-call and after-request stages (stages 4 and 5), when the KV cache is kept and managed across requests.

Some of these stages have a method today and some do not. The decode stage (stage 3) is implemented by TriAttention. The after-request stage (stage 5) is implemented in part by cold-page compression, which handles the pages that leave the GPU for host or disk memory. The rest of this section walks through the framework in detail: how a user configures it and which cache manager it works with, how each path works, and how the other stages will be covered.

### Configuration and Integration with the Cache Manager

Configuration is deliberately small. One block, `kv_cache_compression_config`, collects only the settings that belong to compression: which method to run and that method's own options. A factory validates the block, rejects unsupported combinations before anything is built, and hands the settings to the compression manager, which handles everything from there. Cold-page compression is turned on with `algorithm: quantization_for_cold_page` and `quant: nvfp4`. TriAttention is turned on with `algorithm: triattention` plus its budget and calibration options. The same fields work in the Python API, in `trtllm-serve` YAML, and in `trtllm-bench`. The full option table is in the [KV Cache Compression feature documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kv-cache-compression.md).

The cache manager and the attention backend are outside the framework. The compression manager is designed to adapt to any cache manager and any attention backend. It changes only the contents of KV pages, and it never owns pages, tiers, or the kernels that read them.

Today the framework integrates with KVCacheManagerV2, the KV cache manager built around a flexible, hierarchical storage model. V2 can give different layers pools of different types and sizes, groups layers by their lifecycle, and coalesces buffers of the same size within each group, which keeps fragmentation low even for models that mix full-attention, sliding-window, and recurrent layers. It also manages the host and disk tiers and the migration of pages between them, and it exposes a clean Python API for per-layer buffer configuration. These properties are what make cross-request compression possible. The compression manager binds to V2 once V2 is built, and the cold-page codec plugs into V2's page migration path, so V2 keeps ownership of pools, mappings, and migration while the codec decides how a page is stored off the GPU.

### KV Cache Compression in the Executor Iteration Loop

This path serves the stages inside a request: the prefill-chunk stage (stage 1), the after-prefill stage (stage 2), and the decode stage (stage 3). Today the decode stage (stage 3) is the one with a shipped method. While a prefill or decode step runs, attention reads a fixed view of the KV cache. Between two steps that view can change, and that is where the hooks fire. A method overrides only the hooks it needs, and all of them do nothing by default.

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
| `on_request_init` | Before a request's first prefill chunk | Prefill-chunk stage (stage 1), set-up before the first chunk |
| `on_context_step_end` | After a request's last prefill chunk | After-prefill stage (stage 2) |
| `on_generation_step_begin` | Before each decode step | Decode stage (stage 3) |
| `on_generation_step_end` | After each decode step, once the cache has been updated | Decode stage (stage 3) (used by **TriAttention**) |
| `on_request_finish` | When a request completes or is aborted | Tool-call and after-request stages (stages 4 and 5), when a request pauses or ends |

<p align="center"><sub><em>Table 3. The five hooks in the executor iteration loop, when each one fires, and the stage it serves.</em></sub></p>

</div>

We currently define these five hooks. **TriAttention** uses the generation-end hook, so the decode stage (stage 3) is the one exercised by a shipped method. The other hooks are already in place for the prefill-chunk and after-prefill stages (stages 1 and 2) and for the request-level events of the tool-call and after-request stages (stages 4 and 5), and a new method can use them without changes to the framework.

The hooks ride on the executor's existing request cycle, and the framework wires them up. Methods on this path change which tokens are kept or how they are arranged in the paged cache. Two obligations come with it. The policy that chooses tokens stays separate from the shared compaction kernel. And a method must finish its GPU work before it shrinks or frees any cache pages.

### KV Cache Compression in Cross-Request KV Management

This path is the cross-request hook of the framework. It serves the stages outside a single request: the tool-call stage (stage 4), when a request pauses and its KV waits for the tool to return, and the after-request stage (stage 5), when a request has finished and its KV is kept for reuse, transferred to another worker, or offloaded to host or disk memory.

This whole period belongs to KVCacheManagerV2, introduced above. V2 decides which pages stay on the GPU, move to host or disk memory, are reused by a later request, or are transferred to another worker, and it does no compression itself. The framework therefore injects its compression management into V2 at the points where KV can be compressed: when pages are offloaded and onboarded, when they are transferred, and when a request ends. At each of these points a method's Python code and kernels can be plugged in, and V2 keeps working as before.

Today we cover the offloading part of the after-request stage (stage 5) with **NVFP4 cold-page compression**, enabled by `quantization_for_cold_page` with `quant: nvfp4`. It is a good example of how the framework interacts with V2. The storage path of V2 exposes two hook points, one when a page leaves the GPU and one when it returns. A codec plugged into these points encodes the page on the way out and decodes it on the way back, and the cache manager never sees the difference. How this works in detail, and how the NVFP4 codec is built, is described in the NVFP4 cold-page compression section below and in the [cold-page codec design guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-cache-cold-page-codec.md).

The tool-call stage (stage 4) and the rest of the after-request stage (stage 5), such as KV transfer, will be reached by extending this path.

### Covering the Other Stages

Together, the two paths cover the decode stage (stage 3) and one part of the after-request stage (stage 5), which is what the two shipped methods need. The base class keeps the ability to define more hooks, so every stage that offers a compression opportunity can be reached the same way. How those hooks look is future work, and we will extend the framework as new methods need them.

## Algorithm Implementations

In this section, we walk through the two KV cache compression methods currently implemented in TensorRT LLM, focusing on how each method works and how it integrates with the framework. For a quick-start guide and runnable configurations, please refer to the [KV cache compression examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/kv_cache_compression).

### NVFP4 Cold-Page Compression

#### Introduction

NVFP4 is NVIDIA's 4-bit floating-point format. Every value is stored in 4 bits (E2M1), and every group of 16 values shares one 8-bit (E4M3) scale, so a value costs 4.5 bits on average. TensorRT LLM already uses NVFP4 for model weights and for the active KV cache, so its numerics and conversion kernels are well established.

Cold-page compression applies this format at a different point in the system. Attention KV pages keep their normal data type, FP16, BF16, or FP8, while they are on the GPU. Only when a page leaves the GPU for host or disk memory is it converted to NVFP4, and it is converted back when it returns. The attention kernels never see the compressed form.

Host and disk offloading is a good place for compression for four reasons.

- **Compress only under pressure.** A page is compressed once, when it goes cold, and only if it goes cold. The active cache is never touched, so no error accumulates step after step, and the accuracy impact stays smaller than compressing the cache every step.
- **Fused with the transfer.** The page has to be copied anyway. The encode happens inside that copy, so compression adds no separate pass over the data.
- **Less to move.** A smaller page means fewer bytes over PCIe and the storage link, so offloading and onboarding get faster as well as more capacity.
- **One payload per page.** The compressed page is a single fixed-size block with one base address per tier, instead of separate data and scale buffers. Storage tiers address it like any other page.

Figure 6 compares the three layouts of one KV page.

<div align="center">
<figure>
  <img src="../media/tech_blog29_page_layouts.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 6: One KV page of one layer, holding K and V for 32 tokens, in three layouts. The hot page on the GPU stores 16 bits per value in one pool. The NVFP4 active KV cache and the NVFP4 cold page hold the same bytes, 4-bit data plus one 8-bit block scale per 16 values. The active cache keeps them in two pools that attention reads together; the cold page packs them into one fixed-size block with one base address.</em></sub></p>

#### How It Works in TensorRT LLM

Within TensorRT LLM, NVFP4 cold-page compression is a compression manager on the cross-request KV management path, enabled with `algorithm: quantization_for_cold_page` and `quant: nvfp4`. Figure 7 follows one page out of the GPU and back.

**Encode, on the way out.** When the cache manager offloads pages, it hands the codec a batch of page indices and a stream. The codec launches `invokeNvfp4ColdPageEncode`, a fused kernel that reads the GPU pages, quantizes them, and writes the packed pages straight into the host slots, so quantization and transfer are one pass. The block scale of each group of 16 values is computed during the encode, so an FP16, BF16, or FP8 cache needs no calibration.

What gets quantized is decided once at start-up, from the buffers each layer keeps in its page. Only the attention K and V buffers are quantized, and for an MLA layer only the latent key buffer. Every other buffer in the page is copied as it is: the recurrent state of Gated DeltaNet and state-space layers, the convolution state, the index buffer of DeepSeek Sparse Attention, and the positional part of the DeepSeek-V4 cache. Everything that is not attention KV therefore stays lossless.

The result is one cold page per hot page: a single fixed-size block that holds the NVFP4 data, then the block scales, then the lossless buffers, each 16-byte aligned. Because the size is fixed per layer group, the host and disk tiers address cold pages as base plus slot times size, one copy moves a whole page instead of one copy per buffer, and moves between host and disk copy the bytes unchanged. This single-block layout is exactly what the cold-page codec support in KVCacheManagerV2 expects: V2 defines one cold-page size per lifecycle, calls the codec with batches of page indices, and treats encode and decode as asynchronous work on its own stream.

**Decode, on the way back.** When a request needs a page again, the cache manager onboards it and the codec launches `invokeNvfp4ColdPageDecode`, the fused counterpart that dequantizes the page and writes it in its original data type into the GPU pool before attention reads it. One launcher covers every layout, because the layout table drives the kernel. Target and draft caches are both covered, so speculative decoding works with the codec enabled.

<div align="center">
<figure>
  <img src="../media/tech_blog29_nvfp4_cold_page_pipeline.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 7: One page leaving and returning to the GPU inside the KV cache manager. A page goes cold when its request pauses or ends, or under memory pressure. A fused kernel quantizes the 16-bit hot page and copies it to the host and disk tiers as one 4.5-bit cold block in a single pass, and a fused decode kernel dequantizes and copies it back when an active request reuses it.</em></sub></p>

The implementation lives in `tensorrt_llm/_torch/kv_cache_compression/quantization_for_cold_page/` (the manager and layout policy), `cpp/tensorrt_llm/batch_manager/kv_cache_compression/` (the native codec path), and `cpp/tensorrt_llm/kernels/nvfp4ColdPageKernels.cu` (the fused encode and decode kernels).

### TriAttention

#### Introduction

[TriAttention](https://arxiv.org/abs/2604.04921) (ICML 2026) is a training-free KV cache eviction method for long generations. Its observation is that the importance of a cached token to future queries can be predicted from statistics of the queries themselves, collected once offline for each attention head. During decoding, TriAttention periodically scores the generated tokens with a trigonometric importance measure built from these statistics, keeps the most important `budget` tokens, and physically compacts the cache, so that more sequences fit on a GPU at once. The prompt is always preserved; only generated tokens are evicted. Figure 8 shows one eviction round. For technical details, please refer to the paper and to the [TriAttention example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md) in TensorRT LLM.

<div align="center">
<figure>
  <img src="../media/tech_blog29_triattention_concept.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 8: TriAttention keeps the prompt intact and evicts generated tokens. Every β new tokens, the generated tokens are scored with the calibrated query statistics of each head. The `budget` highest-scoring tokens stay, the rest are evicted, and the cache is compacted, so decoding continues over a shorter cache.</em></sub></p>

#### How It Works in TensorRT LLM

**The algorithm.** TriAttention scores a cached token by how much attention future queries are expected to pay to it, without waiting for those queries. With rotary position embedding, the attention logit between a query and a key is a sum over frequency bands. Each band contributes the product of the query and key magnitudes in that band times the cosine of the rotation angle between them, and that angle grows with the distance between the two positions. Averaging the query over a calibration corpus replaces the unknown future query by each head's mean pre-RoPE query and its magnitude, the `E_q` and `E_q_norm` entries of the calibration file, together with the RoPE frequencies `omega`. The expected logit of a key then depends only on the key itself and on its position, and it can be evaluated from the K cache alone. This trigonometric expected logit is the importance score.

Every `beta` generated tokens, once a sequence is over its budget, TriAttention gathers the K cache of the generated tokens, computes this score for every token and head, normalizes the scores per head over the decode window, and keeps the `budget` tokens with the highest scores. In the default `union` mode each KV head nominates its top tokens and the union is re-ranked by each token's best score; `per_head` and `per_layer_perhead` keep separate sets per head. The prompt is never scored or evicted, and the kept tokens are compacted so that the cache physically shrinks.

**Kernel optimizations.** Doing this in plain PyTorch would gather K, apply the rotation, and multiply for every head and token, several times per second per request. TensorRT LLM performs a series of kernel optimizations instead. A fused CuTe DSL kernel on Blackwell reads the K pages directly from the paged cache, applies the calibrated cosine and sine coefficients from a precomputed mean-phase table, and produces the per-head scores together with the statistics needed for normalization in one pass. Triton kernels reduce the scores per eviction mode, normalize them, and settle top-k ties deterministically. A native CUDA compaction kernel then moves the kept K and V in place. No K or V leaves the paged cache for scoring.

**Injection into decoding.** TriAttention is a compression manager for compression in the executor iteration loop, and it overrides one hook, `on_generation_step_end`, so an eviction round runs right after the KV cache manager has updated the cache and before the next decode step reads it. It is enabled with `algorithm: triattention` plus `budget`, `beta`, `eviction_mode`, and `calibration_path`. Figure 9 follows one round from the hook to the compacted cache. When the hook fires, the manager picks the requests that have generated `beta` new tokens since their last round and exceed their budget. For these requests, `_execute_eviction_round` collects the cache lengths and block offsets, launches the scoring kernel, reduces and selects the tokens to keep, and runs the compaction. The manager then reports the new cache length to the cache manager, which returns the freed pages. A speculative step that confirms several tokens at once triggers at most one round.

<div align="center">
<figure>
  <img src="../media/tech_blog29_triattention_pipeline.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 9: One TriAttention eviction round in TensorRT LLM, from the generation-end hook to the compacted cache.</em></sub></p>

The method changes the physical cache length but keeps block reuse valid, because it compacts only the generated suffix and preserves the prompt prefix that the cache manager reuses. Decoding runs the model's standard attention kernel over the compacted cache. For calibration, configuration parameters, and current requirements, please refer to the [TriAttention example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md).

## Evaluation

This section consolidates accuracy and performance results for the KV cache compression methods supported in TensorRT LLM. Every comparison is against TensorRT LLM's own uncompressed configuration on the same hardware, software build, and serving configuration; only the compression setting differs.

### NVFP4 Cold-Page Compression

Unless otherwise specified, the experiments below use GB300 GPUs, the PyTorch backend, the paged KV cache manager, and block reuse enabled. Before any result was accepted we confirmed that pages were actually compressed during the measured window, using the offload and onboard byte counters on the `/metrics` endpoint and an Nsight Systems trace.

#### Accuracy

Cold-page compression is lossy, so the first question is how much accuracy it can cost. Two facts bound the answer.

First, a page is quantized only when it leaves the GPU, and at most once per offload. The worst case is a deployment where every page is re-quantized on every turn. Even then a page never carries more than one NVFP4 rounding at a time, so the accuracy of cold-page compression is bounded below by that of a fully NVFP4 KV cache, a configuration TensorRT LLM already ships and whose loss is small and well characterized. In practice most pages are offloaded once or not at all, so the typical case sits far from this bound.

Second, repeated quantize-and-dequantize round trips do not compound. Quantization is a projection onto a finite set of values, not a fresh random error each time. For NVFP4 with 8-bit block scales we showed analytically that after the first round trip the encoded state can change at most once more, and on real KV values it does not change at all. Figure 10 shows the measurement on Qwen3-8B with a BF16 KV cache on AIME24 and AIME25, 30 questions with 8 seeds each, using the NVFP4 quantizer from Transformer Engine in a standalone harness on B200 and applying the round trips to every page. The first round trip changes 98.5% of the 13 billion compared KV values, which is the expected rounding. The second round trip changes none of them, and neither do the fourth or the eighth. End to end, one round trip moves accuracy by 2.08 and 1.67 points on the two sets, within the noise of this setup where most generations hit the 32K output cap. The second round trip reproduces all 480 outputs of the first exactly, so accuracy after two round trips is identical to accuracy after one.

<div align="center">
<figure>
  <img src="../media/tech_blog29_nvfp4_roundtrip.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 10: Repeated NVFP4 round trips on Qwen3-8B with a BF16 KV cache. Left: share of KV values changed by one more quantize-and-dequantize round trip; the first changes the expected 98.5%, the second, fourth, and eighth change none. Right: AIME24 and AIME25 accuracy after 0, 1, and 2 round trips; the first and second round trips produce identical outputs on all 480 samples.</em></sub></p>

Together, these two facts mean that cold-page compression cannot drift with repeated offloading. Its accuracy cost is at most one NVFP4 rounding of the pages that actually leave the GPU, and it does not grow with the number of times a page is offloaded and brought back.

#### Performance

The compression ratio follows from the format. Packed NVFP4 data plus one 8-bit scale per 16 values costs 0.5625 bytes per value, against 2 bytes for FP16 or BF16 and 1 byte for FP8. Table 4 lists the measured size of one attention KV page off the GPU. The recurrent-state buffers of hybrid models are stored as they are and do not shrink.

| Active KV type | Model | Uncompressed page | NVFP4 cold page | Reduction |
| :--- | :--- | :--- | :--- | :--- |
| FP8 | Qwen3.5-397B-A17B | 1,048,576 B | 589,824 B | 43.75% (1.78x more pages per byte) |
| FP8 | Qwen3-8B | 2,359,296 B | 1,327,104 B | 43.75% (1.78x) |
| FP8 MLA latent, index buffer kept lossless | GLM-5.2 | 3,098,112 B | 1,824,000 B | 41.1% (1.70x) |
| BF16 | Qwen3-8B | 4,718,592 B | 1,327,104 B | 71.9% (3.56x) |

<p align="center"><sub><em>Table 4. Measured size of one attention KV page off the GPU, uncompressed and as an NVFP4 cold page.</em></sub></p>

We benchmark the NVFP4 host cache against the uncompressed host cache on two models: GLM-5.2 (756B, MLA attention) and Qwen3.5-397B-A17B (hybrid attention plus Gated DeltaNet, NVFP4 weights). The workload is a 3,600-second replay of the InferenceX AgentX 256k agentic trace, which has heavy prefix reuse across turns; these runs use the public trace and the published InferenceX configurations but are not an official InferenceX submission. In the uncompressed configuration the host tier stores pages in FP8, the normal KV type of both models. We sweep the published configurations and a large set of derived ones (concurrency, prefill/decode split, and GPU count) so that both settings are measured on the same grid: 54 matched configurations and 218 accepted runs (two repeats each) for GLM-5.2 on 8 to 48 GB300, and 131 matched configurations and 330 accepted runs (mostly single repeat) for Qwen3.5-397B-A17B on 3 to 60 GB300. The host tier is 128 GiB per prefill rank in every run. We report **total token throughput per reserved GPU** against **P90 end-to-end normalized interactivity** (tokens per second per user, higher is better), the Pareto view in Figure 11.

<div align="center">
<figure>
  <img src="../media/tech_blog29_pareto.svg" width="1000">
</figure>
</div>
<p align="center"><sub><em>Figure 11: Throughput per reserved GB300 versus P90 interactivity for the uncompressed (FP8) and NVFP4 host caches on GLM-5.2 (left) and Qwen3.5-397B-A17B (right). Each point is one configuration averaged over repeats; outlined points are the published InferenceX configurations; lines are the Pareto frontiers of the two settings.</em></sub></p>

Figure 11 shows the throughput-interactivity Pareto frontiers; curves further to the upper right deliver more throughput at the same interactivity. On GLM-5.2 the NVFP4 frontier lies on or above the uncompressed one: it gains in the 31 to 51 tokens/s/user band (median +12.1% across the frontier points in the overlapping range, at most +28.5% at 36 tokens/s/user) and coincides with it above roughly 51 tokens/s/user, where both frontiers are formed by the same low-concurrency configurations. Below 31 tokens/s/user only NVFP4 has points, at 55,000 to 59,500 tokens/s per GPU, and the uncompressed cache has no tested point above 46,461 tokens/s per GPU. On Qwen3.5-397B-A17B the two frontiers largely coincide, and NVFP4 pulls ahead only at the throughput-bound, low-interactivity end. We summarize the results with four metrics:

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
- **NVFP4 cold-page compression**: attention pages are stored as NVFP4 while they are in host or disk memory and restored to the model's normal KV data type before attention reads them. Encoding and decoding are fused into the copy itself, host and disk share one encoded form, and the recurrent state of hybrid models and side buffers such as an MLA index pass through losslessly in the same page. It covers MHA, MQA, GQA, MLA, and DeepSeek-V4 cache layouts and has been tested with the Qwen3, Qwen3.5, GLM, DeepSeek-R1, and DeepSeek-V4 families.
- **TriAttention**: demonstrates compression in the executor iteration loop by evicting generated tokens during decoding, with the model's standard attention kernel running over the compacted cache. It has been tested with the Qwen3, GPT-OSS, and Llama 3 families.

The framework and both methods are upstream in [PR #16957](https://github.com/NVIDIA/TensorRT-LLM/pull/16957) (TriAttention), [PR #17512](https://github.com/NVIDIA/TensorRT-LLM/pull/17512) (page encoder and decoder support in the cache manager), and [PR #18091](https://github.com/NVIDIA/TensorRT-LLM/pull/18091) (NVFP4 cold-page compression). A new method needs one contract and its kernel, and no change to the attention kernels, the cache manager, or the serving loop.

### Future Work

- **Smaller cold-page formats.** The page contract fixes only the compressed page size, so 2-bit formats, entropy coding, or low-rank projection can be added as new codecs without changes to the cache manager.
- **Native low-precision decode kernels.** No attention kernel today reads a 4-bit KV cache for the head sizes of the tested models. Native kernels would let an NVFP4 active cache and NVFP4 cold pages work together.
- **Sizing guidance for the host tier.** The serving results above used one host tier size. A sweep over host tier sizes at the published InferenceX configurations, with the offload and onboard counters recorded, would show how much host memory a compressed cache actually needs.
- **Disaggregated serving.** Cold-page compression covers the GPU-to-host and host-to-disk boundaries inside one worker today. Carrying the encoded page across the transfer from the prefill worker to the decode worker, and into the decode worker's host tier, is the natural next step.
- **More methods in the executor iteration loop and hybrid-model state.** The hooks are not tied to TriAttention, and we expect further eviction and context-compression methods to use them. We are also exploring compression for the recurrent state of hybrid models, which today passes through losslessly.
