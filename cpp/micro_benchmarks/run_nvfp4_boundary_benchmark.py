#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the NVFP4 boundary-kernel A/B matrix with reproducible provenance.

The C++ binary owns allocations, correctness checks, CUDA-event timing, and
the result CSV. This runner intentionally starts one process per matrix cell:
TensorRT-LLM reads ``TRTLLM_ENABLE_PDL`` once per process, so PDL-on and
PDL-off measurements must never share a process.
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import itertools
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time

DESIGN_RECORD_NAME = "nvfp4-boundary-kernel-ab-benchmark-design-2026-08-05.md"
FULL_PAGES = (1, 2, 4, 8, 16, 31, 32, 33, 64, 65, 128, 129, 256, 257)
QUICK_PAGES = (1, 33, 257)
DTYPES = ("fp16", "bf16", "fp8_e4m3")
DIRECTIONS = ("offload", "onboard")
ADDRESS_MODES = ("contiguous", "permuted")
PDL_MODES = (0, 1)
RUNNER_COLUMNS = (
    "runner_run_id", "runner_pdl", "runner_pages", "runner_dtype",
    "runner_direction", "runner_address_mode",
)
RECORDED_ENV = (
    "CONTAINER_IMAGE", "CUDA_DRIVER_VERSION", "CUDA_VISIBLE_DEVICES", "ENROOT_IMAGE",
    "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "PATH", "PYTHONPATH",
    "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "TRTLLM_ENABLE_PDL",
)


@dataclasses.dataclass(frozen=True)
class Cell:
    pages: int
    dtype: str
    direction: str
    address_mode: str
    pdl: int

    @property
    def run_id(self) -> str:
        return f"pdl{self.pdl}-pages{self.pages:04d}-{self.dtype}-{self.direction}-{self.address_mode}"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: pathlib.Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json(path: pathlib.Path, value: object) -> None:
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_pages(raw: str) -> tuple[int, ...]:
    try:
        pages = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--pages must be comma-separated integers") from error
    if not pages or any(value <= 0 for value in pages) or len(set(pages)) != len(pages):
        raise argparse.ArgumentTypeError("--pages must contain unique positive integers")
    return pages


def parse_choices(raw: str, allowed: tuple, label: str, cast: type = str) -> tuple:
    try:
        values = tuple(cast(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} contains an invalid value") from error
    invalid = tuple(value for value in values if value not in allowed)
    if not values or invalid or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"{label} must contain unique values from {', '.join(map(str, allowed))}"
        )
    return values


def capture(argv: list[str], cwd: pathlib.Path) -> dict[str, object]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv, cwd=cwd, check=False, capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "argv": list(argv), "command": shlex.join(argv),
            "duration_seconds": time.monotonic() - started, "error": str(error),
        }
    return {
        "argv": list(argv), "command": shlex.join(argv),
        "duration_seconds": time.monotonic() - started, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }


def git_output(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def tracked_diff(repo: pathlib.Path) -> bytes:
    return subprocess.run(
        ["git", "--no-pager", "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def probes(binary: pathlib.Path, gpu: str, repo: pathlib.Path, *, post: bool) -> dict[str, object]:
    commands: list[list[str]] = [
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ]
    if not post:
        commands += [
            ["nvidia-smi", "-L"],
            ["nvidia-smi", "topo", "-m"],
            ["nvcc", "--version"],
            ["ldd", str(binary)],
            ["uname", "-a"],
        ]
    return {shlex.join(command): capture(command, repo) for command in commands}


def build_command(
    binary: pathlib.Path,
    cell: Cell,
    output_csv: pathlib.Path,
    *,
    warmup: int,
    iterations: int,
    samples: int,
    seed: int,
) -> list[str]:
    return [
        str(binary), "--pages", str(cell.pages), "--dtype", cell.dtype,
        "--direction", cell.direction, "--address-mode", cell.address_mode,
        "--warmup", str(warmup), "--iterations", str(iterations),
        "--samples", str(samples), "--seed", str(seed),
        "--output-csv", str(output_csv),
    ]


def validate_csv(path: pathlib.Path, cell: Cell, samples: int) -> tuple[list[str], int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("benchmark did not produce a non-empty result CSV")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("result CSV has no header")
        rows = list(reader)
    if not rows:
        raise ValueError("result CSV has no data rows")
    expected = {
        "pages": str(cell.pages), "dtype": cell.dtype,
        "direction": cell.direction, "address_mode": cell.address_mode,
        "pdl": str(cell.pdl),
    }
    for index, row in enumerate(rows):
        mismatches = {
            key: (expected_value, row.get(key))
            for key, expected_value in expected.items()
            if row.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"row {index} does not match requested cell: {mismatches}")

    variants = ("fused", "staged_sm", "staged_dma")
    summaries = {row.get("variant"): row for row in rows if row.get("row_kind") == "summary"}
    if set(summaries) != set(variants):
        raise ValueError(f"summary variants differ: {sorted(summaries)}")
    for variant in variants:
        summary_status = summaries[variant].get("status")
        sample_rows = [
            row for row in rows
            if row.get("row_kind") == "sample" and row.get("variant") == variant
        ]
        if variant in ("fused", "staged_sm") and summary_status != "ok":
            raise ValueError(f"required variant {variant} is not supported")
        if summary_status == "ok" and len(sample_rows) != samples:
            raise ValueError(
                f"variant {variant} has {len(sample_rows)} samples; expected {samples}"
            )
        if summary_status == "unsupported" and sample_rows:
            raise ValueError(f"unsupported variant {variant} unexpectedly has samples")
        if summary_status not in ("ok", "unsupported"):
            raise ValueError(f"variant {variant} has invalid summary status {summary_status!r}")
    return list(reader.fieldnames), len(rows)


def aggregate_csv(output: pathlib.Path, cells: dict[str, Cell], successful: list[pathlib.Path]) -> int:
    if not successful:
        return 0
    expected: list[str] | None = None
    rows: list[dict[str, str]] = []
    for result_path in successful:
        run_id = result_path.parent.name
        cell = cells[run_id]
        with result_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or ())
            if expected is None:
                expected = fields
            elif fields != expected:
                raise ValueError(f"CSV schema differs in {result_path}")
            for row in reader:
                row.update(
                    {
                        "runner_run_id": run_id,
                        "runner_pdl": str(cell.pdl),
                        "runner_pages": str(cell.pages),
                        "runner_dtype": cell.dtype,
                        "runner_direction": cell.direction,
                        "runner_address_mode": cell.address_mode,
                    }
                )
                rows.append(row)
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=[*RUNNER_COLUMNS, *(expected or ())])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--design-record",
        type=pathlib.Path,
        help="Optional design record to snapshot; benchmark execution does not depend on it",
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--pages", type=parse_pages, help="Override the preset Page cohorts")
    parser.add_argument("--dtypes", type=lambda raw: parse_choices(raw, DTYPES, "--dtypes"))
    parser.add_argument(
        "--directions", type=lambda raw: parse_choices(raw, DIRECTIONS, "--directions")
    )
    parser.add_argument(
        "--address-modes",
        type=lambda raw: parse_choices(raw, ADDRESS_MODES, "--address-modes"),
    )
    parser.add_argument("--pdl", type=lambda raw: parse_choices(raw, PDL_MODES, "--pdl", int))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--cuda-visible-device",
        required=True,
        help="Exactly one physical GPU index, GPU UUID, or MIG UUID",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = pathlib.Path(__file__).resolve()
    repo = pathlib.Path(git_output(script.parent, "rev-parse", "--show-toplevel"))
    binary = args.binary.expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"benchmark binary is not executable: {binary}")
    if not args.cuda_visible_device.strip() or "," in args.cuda_visible_device:
        raise SystemExit("--cuda-visible-device must identify exactly one GPU")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")

    default_design = repo.parent / "docs" / "trtllm-kv-cache" / DESIGN_RECORD_NAME
    design = args.design_record.expanduser().resolve(strict=True) if args.design_record else None
    if design is None and default_design.is_file():
        design = default_design.resolve()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runs_dir = output / "runs"
    runs_dir.mkdir()

    pages = args.pages or (QUICK_PAGES if args.preset == "quick" else FULL_PAGES)
    dtypes = args.dtypes or DTYPES
    directions = args.directions or DIRECTIONS
    address_modes = args.address_modes or ADDRESS_MODES
    pdl_modes = args.pdl or PDL_MODES
    defaults = {"quick": (5, 20, 3), "full": (20, 100, 5)}[args.preset]
    warmup = args.warmup if args.warmup is not None else defaults[0]
    iterations = args.iterations if args.iterations is not None else defaults[1]
    samples = args.samples if args.samples is not None else defaults[2]
    if min(warmup, iterations, samples) <= 0:
        raise SystemExit("warmup, iterations, and samples must be positive")

    cells = [
        Cell(page_count, dtype, direction, address_mode, pdl)
        for pdl, page_count, dtype, direction, address_mode in itertools.product(
            pdl_modes, pages, dtypes, directions, address_modes
        )
    ]
    cells_by_id = {cell.run_id: cell for cell in cells}

    diff = tracked_diff(repo)
    write_bytes(output / "tracked.diff", diff)
    design_record = None
    if design is not None:
        design_snapshot = output / design.name
        write_bytes(design_snapshot, design.read_bytes())
        design_record = {
            "source": str(design), "snapshot": design_snapshot.name,
            "sha256": sha256_file(design_snapshot),
        }
    commands = []
    for cell in cells:
        result_path = runs_dir / cell.run_id / "result.csv"
        argv = build_command(
            binary,
            cell,
            result_path,
            warmup=warmup,
            iterations=iterations,
            samples=samples,
            seed=args.seed,
        )
        commands.append(
            {
                "run_id": cell.run_id,
                "environment_overrides": {
                    "CUDA_VISIBLE_DEVICES": args.cuda_visible_device,
                    "TRTLLM_ENABLE_PDL": str(cell.pdl),
                },
                "argv": argv,
                "command": shlex.join(argv),
            }
        )

    source_paths = (
        script,
        repo / "cpp" / "micro_benchmarks" / "nvfp4BoundaryKernelsBenchmark.cu",
        repo / "cpp" / "micro_benchmarks" / "CMakeLists.txt",
        repo / "cpp" / "tensorrt_llm" / "kernels" / "nvfp4BoundaryKernels.cu",
        repo / "cpp" / "tensorrt_llm" / "kernels" / "nvfp4BoundaryKernels.h",
        repo / "cpp" / "tensorrt_llm" / "batch_manager" / "kvCacheManagerV2Utils.cu",
        repo / "cpp" / "tensorrt_llm" / "batch_manager" / "kvCacheManagerV2Utils.h",
    )
    source_records = []
    source_snapshot_root = output / "source-inputs"
    for source_path in source_paths:
        relative = source_path.relative_to(repo)
        snapshot = source_snapshot_root / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(snapshot, source_path.read_bytes())
        source_records.append(
            {
                "source": str(source_path), "snapshot": str(snapshot.relative_to(output)),
                "sha256": sha256_file(snapshot),
            }
        )

    contract = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "claim_boundary": (
            "Standalone boundary-kernel microbenchmark only; no KVCM, serving, "
            "model-accuracy, or end-to-end performance claim."
        ),
        "design_record": design_record,
        "repository": {
            "root": str(repo),
            "head": git_output(repo, "rev-parse", "HEAD"),
            "branch": git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_diff": "tracked.diff",
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "status_porcelain": git_output(repo, "status", "--porcelain", "--untracked-files=normal"),
        },
        "runner": {"path": str(script), "sha256": sha256_file(script), "argv": sys.argv},
        "source_inputs": source_records,
        "binary": {
            "path": str(binary), "bytes": binary.stat().st_size, "sha256": sha256_file(binary),
        },
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "container_identity": os.environ.get("CONTAINER_IMAGE") or os.environ.get("ENROOT_IMAGE"),
            "proc_1_cgroup": pathlib.Path("/proc/1/cgroup").read_text(errors="replace"),
            "recorded_environment": {name: os.environ[name] for name in RECORDED_ENV if name in os.environ},
            "selected_gpu": args.cuda_visible_device,
        },
        "matrix": {
            "preset": args.preset, "pages": pages, "dtypes": dtypes,
            "directions": directions, "address_modes": address_modes,
            "pdl_modes": pdl_modes, "warmup": warmup, "iterations": iterations,
            "samples": samples, "seed": args.seed, "processes": len(cells),
        },
        "commands": commands,
    }
    contract_path = output / "contract.json"
    write_json(contract_path, contract)
    contract_hash = sha256_file(contract_path)
    write_bytes(output / "contract.json.sha256", f"{contract_hash}  contract.json\n".encode())

    if not args.dry_run:
        write_json(output / "preflight.json", probes(binary, args.cuda_visible_device, repo, post=False))

    manifest_path = output / "run_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "contract_sha256": contract_hash,
        "started_utc": utc_now(),
        "status": "running",
        "runs": [],
    }
    write_json(manifest_path, manifest)
    successful: list[pathlib.Path] = []
    failures = 0

    for cell, command_record in zip(cells, commands):
        run_dir = runs_dir / cell.run_id
        run_dir.mkdir()
        result_path = run_dir / "result.csv"
        write_json(run_dir / "command.json", command_record)
        started_utc = utc_now()
        started = time.monotonic()
        returncode: int | None = None
        timed_out = False
        csv_error: str | None = None

        if args.dry_run:
            status = "dry_run"
            (run_dir / "stdout.log").touch()
            (run_dir / "stderr.log").touch()
        else:
            child_env = os.environ.copy()
            child_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_device
            child_env["TRTLLM_ENABLE_PDL"] = str(cell.pdl)
            with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
                try:
                    completed = subprocess.run(
                        command_record["argv"],
                        cwd=repo,
                        env=child_env,
                        check=False,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=args.timeout_seconds,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            if timed_out:
                status = "failed"
                csv_error = f"timed out after {args.timeout_seconds}s"
            elif returncode != 0:
                status = "failed"
                csv_error = f"benchmark exited with status {returncode}"
            else:
                try:
                    fields, row_count = validate_csv(result_path, cell, samples)
                    status = "success"
                    successful.append(result_path)
                except ValueError as error:
                    status = "failed"
                    csv_error = str(error)

        exit_record = {
            "run_id": cell.run_id,
            "status": status,
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "returncode": returncode,
            "timed_out": timed_out,
            "csv_error": csv_error,
            "result_csv_sha256": sha256_file(result_path) if result_path.is_file() else None,
            "result_fields": fields if status == "success" else None,
            "result_rows": row_count if status == "success" else 0,
        }
        write_json(run_dir / "exit.json", exit_record)
        manifest["runs"].append(exit_record)  # type: ignore[union-attr]
        if status == "failed":
            failures += 1
        write_json(manifest_path, manifest)
        if status == "failed" and not args.keep_going:
            break

    aggregate_rows = aggregate_csv(output / "raw.csv", cells_by_id, successful)
    if not args.dry_run:
        write_json(output / "postflight.json", probes(binary, args.cuda_visible_device, repo, post=True))
    manifest.update(
        {
            "finished_utc": utc_now(),
            "status": "failed" if failures else ("dry_run" if args.dry_run else "complete"),
            "successful_processes": len(successful),
            "failed_processes": failures,
            "aggregate_rows": aggregate_rows,
        }
    )
    write_json(manifest_path, manifest)

    checksum_lines = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    write_bytes(output / "SHA256SUMS", ("\n".join(checksum_lines) + "\n").encode())
    print(f"artifacts: {output}")
    print(f"contract SHA256: {contract_hash}")
    print(f"processes recorded: {len(manifest['runs'])}/{len(cells)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
