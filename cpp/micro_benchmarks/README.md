# Micro Benchmarks

This folder contains benchmarks for specific components in TRT-LLM,
using [google-benchmark](https://github.com/google/benchmark/tree/main)

## Building

To build add the `--micro_benchmark` flag to `build_wheel.py` or pass `-DBUILD_MICRO_BENCHMARKS=ON` to cmake

## Benchmark Documentations

### Mixture Of Experts Backend Benchmark

> [!CAUTION]
> Disclaimer this benchmark is intended for developers to help evaluating the impact of new optimisations. This benchmark does not meet the same quality standards as other parts of TRT-LLM. Please use with caution

Target `mixtureOfExpertsBackendBenchmark`

This benchmark covers the backend used by the `MixtureOfExperts` plugin. It allows you to benchmark different MOE
configurations without building a TRT engine.

Usage:

```bash
./mixtureOfExpertsBackendBenchmark

# or

./mixtureOfExpertsBackendBenchmark --input_file <JSON benchmark definition>
```

For more information see:

```
./mixtureOfExpertsBackendBenchmark --help
```

The `gen-moe-workload-file.py` is a helper script that can generate workload files for MOE benchmarks. This is useful
for sharing or comparing configurations, such as when generating a reproduction case for a performance bug

### NVFP4 Boundary Kernels Benchmark

Target `nvfp4BoundaryKernelsBenchmark` compares the complete fused boundary
pipeline against the same quantization/dequantization through GPU compact
staging. `staged_sm` uses the real KVCM V2 `batchedCopy` implementation once
per packed-K, packed-V, scale-K, and scale-V Pool. `staged_dma` is a secondary
CUDA 13 `cudaMemcpyBatchAsync` reference. This is a kernel microbenchmark, not
a KVCM or serving benchmark.

One binary process measures one fixed cell. Use the runner for Page-batch,
dtype, direction, address-mode, and PDL sweeps:

```bash
python3 cpp/micro_benchmarks/run_nvfp4_boundary_benchmark.py \
  --binary ./cpp/build/micro_benchmarks/nvfp4BoundaryKernelsBenchmark \
  --output-dir /tmp/nvfp4-boundary-results \
  --pages 1,2,4,8,12,16,32,64,72,80,128,256,512 \
  --dtypes fp16,bf16,fp8_e4m3 \
  --directions offload,onboard \
  --address-modes contiguous,permuted \
  --pdl 0,1 \
  --warmup 20 --iterations 100 --samples 21 \
  --cuda-visible-device 0 --keep-going
```

The runner starts a separate process for every cell because PDL is read once
per process. It preserves the exact command, environment, source and binary
hashes, preflight/postflight probes, per-cell logs, raw CSV, and checksums. It
fails when a result does not match the requested cell or lacks a required
variant/sample. PDL modes and GPU selections must not share a process.
