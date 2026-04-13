"""CLI entry point for flux-bend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _parse_params(params_str: str) -> dict[str, Any]:
    """Parse a 'k=v,k=v' string into a dict, with basic type coercion.

    Handles lists like heads=[0,1,2,3] by respecting bracket nesting.
    """
    if not params_str:
        return {}

    # Split on commas that are NOT inside brackets
    pairs: list[str] = []
    current = ""
    depth = 0
    for ch in params_str:
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            pairs.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        pairs.append(current.strip())

    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid param format (expected k=v): '{pair}'")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            result[key] = [int(x.strip()) for x in inner.split(",") if x.strip()]
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def cmd_bend(args: argparse.Namespace) -> None:
    """Run a single bending operation."""
    import hashlib as _hashlib

    from flux_bend.loader import (
        clone_tensors,
        load_transformer_tensors,
        resolve_model_path,
        save_model,
    )
    from flux_bend.metadata import compute_tensor_checksums, save_image_with_metadata
    from flux_bend.model_info import ModelInfo
    from flux_bend.naming import make_image_filename
    from flux_bend.registry import discover_modes
    from flux_bend.inference import (
        generate_image, generate_image_i2i, load_pipeline, load_source_image,
        patch_pipeline,
    )

    model_info = ModelInfo.klein_4b()
    modes = discover_modes()

    if args.mode not in modes:
        print(f"Unknown mode: '{args.mode}'. Available: {sorted(modes.keys())}")
        sys.exit(1)

    mode_cls = modes[args.mode]
    mode = mode_cls()
    params = _parse_params(args.params)
    params = mode_cls.validate(params, model_info)

    print(f"Mode: {args.mode}")
    print(f"Params: {params}")

    # Load and clone tensors
    print("Loading model tensors...")
    model_dir = resolve_model_path(args.model)
    tensors = load_transformer_tensors(model_dir)
    bent_tensors = clone_tensors(tensors)

    # Apply bending
    print("Applying bending...")
    mode.apply(bent_tensors, params, model_info)

    # Tensor checksums: sha256[:8] of each modified tensor after bending
    checksums = compute_tensor_checksums(tensors, bent_tensors)
    if checksums:
        print(f"Modified {len(checksums)} tensor(s), checksums:")
        for k, v in sorted(checksums.items()):
            print(f"  {k}: {v}")

    # Generate image
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(args.model)
    patch_pipeline(pipeline, bent_tensors, model_info.key_mapping)

    # Install inference hooks (e.g. RoPE warp) if the mode provides them
    hooks = mode.get_inference_hooks(params, model_info)
    cleanups = [hook(pipeline) for hook in hooks]

    seed = params.get("seed", args.seed) if args.seed else params.get("seed", 42)

    # Load source image for img2img if provided
    is_i2i = args.image is not None
    source_image_hash: str | None = None
    if is_i2i:
        source_path = Path(args.image)
        source_img = load_source_image(source_path, 1024, 1024)
        source_image_hash = _hashlib.sha256(source_path.read_bytes()).hexdigest()[:16]
        image = generate_image_i2i(pipeline, source_img, args.prompt, seed=seed)
    else:
        image = generate_image(pipeline, args.prompt, seed=seed)

    # Cleanup hooks
    for cleanup in cleanups:
        cleanup()

    # Filter internal params for metadata
    metadata_params = {
        k: v for k, v in sorted(params.items()) if not k.startswith("_")
    }

    filename = make_image_filename(args.mode, params, is_i2i=is_i2i)
    image_path = out_dir / filename
    save_image_with_metadata(
        image, image_path,
        model=args.model,
        mode=args.mode,
        params=metadata_params,
        prompt=args.prompt,
        seed=seed,
        steps=50,
        guidance_scale=4.0,
        width=1024,
        height=1024,
        tensor_checksums=checksums,
        source_image_hash=source_image_hash,
    )
    print(f"Saved: {image_path}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a sweep from a YAML config.

    Memory strategy:
    - original_tensors: loaded once on CPU, kept for the entire sweep (~8 GB).
    - Per job: clone_tensors creates a CPU copy (~8 GB), mode.apply() modifies
      it in float32, patch_pipeline() copies bf16 weights into the pipeline's
      transformer on GPU. After patching, the bent dict is no longer needed.
    - del bent + torch.cuda.empty_cache() frees CPU clone and leftover GPU
      allocations from the previous job, keeping peak memory bounded.
    """
    import hashlib as _hashlib

    import torch

    from flux_bend.grid import make_grid
    from flux_bend.inference import (
        generate_image, generate_image_i2i, load_pipeline, load_source_image,
        patch_pipeline,
    )
    from flux_bend.loader import (
        clone_tensors,
        load_transformer_tensors,
        resolve_model_path,
    )
    from flux_bend.metadata import compute_tensor_checksums, save_image_with_metadata
    from flux_bend.model_info import ModelInfo
    from flux_bend.naming import make_image_filename, make_pipeline_image_filename
    from flux_bend.registry import discover_modes
    from flux_bend.sweep import parse_sweep_config

    config = parse_sweep_config(Path(args.config))
    model_info = ModelInfo.klein_4b()
    modes = discover_modes()

    total_jobs = sum(len(g.jobs) for g in config.groups)
    print(f"Sweep: {len(config.groups)} grid(s), {total_jobs} jobs")
    print(f"Model: {config.model}")
    print(f"Prompt: {config.prompt}")
    print(f"Seed: {config.seed}")

    if config.image:
        print(f"Image: {config.image} (img2img mode)")

    if args.dry_run:
        for gi, group in enumerate(config.groups):
            print(f"\n--- Grid {gi} ---")
            if group.group_label:
                print(f"  Group: {group.group_label}")
            print(f"  Rows: {group.row_param_name} = {group.row_labels}")
            print(f"  Cols: {group.col_param_name} = {group.col_labels}")
            for job in group.jobs:
                if job.stages:
                    print(f"  Job {job.job_index}: pipeline ({len(job.stages)} stages)")
                    for si, stage in enumerate(job.stages):
                        print(f"    Stage {si}: {stage.mode_name} {stage.params}")
                else:
                    print(f"  Job {job.job_index}: {job.params}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source image for img2img if configured
    is_i2i = config.image is not None
    source_img = None
    source_image_hash: str | None = None
    if is_i2i:
        print(f"\nLoading source image: {config.image}")
        source_img = load_source_image(
            config.image,
            config.inference["width"],
            config.inference["height"],
        )
        source_image_hash = _hashlib.sha256(config.image.read_bytes()).hexdigest()[:16]
        print(f"  Source image hash: {source_image_hash}")

    # Load model tensors ONCE on CPU — kept for the entire sweep
    print("\nLoading model tensors...")
    model_dir = resolve_model_path(config.model)
    original_tensors = load_transformer_tensors(model_dir)

    # Load pipeline ONCE — reused for all jobs
    pipeline = load_pipeline(config.model)

    # Generate baseline (job 0 — unbent, but WITH source image if img2img)
    print("\nGenerating baseline image...")
    patch_pipeline(pipeline, original_tensors, model_info.key_mapping)
    if is_i2i:
        baseline = generate_image_i2i(
            pipeline,
            source_img,
            config.prompt,
            seed=config.seed,
            steps=config.inference["steps"],
            guidance_scale=config.inference["guidance_scale"],
            width=config.inference["width"],
            height=config.inference["height"],
        )
    else:
        baseline = generate_image(
            pipeline,
            config.prompt,
            seed=config.seed,
            steps=config.inference["steps"],
            guidance_scale=config.inference["guidance_scale"],
            width=config.inference["width"],
            height=config.inference["height"],
        )
    baseline_name = "i2i_baseline.png" if is_i2i else "baseline.png"
    baseline_path = out_dir / baseline_name
    save_image_with_metadata(
        baseline, baseline_path,
        model=config.model,
        mode="baseline",
        params={},
        prompt=config.prompt,
        seed=config.seed,
        steps=config.inference["steps"],
        guidance_scale=config.inference["guidance_scale"],
        width=config.inference["width"],
        height=config.inference["height"],
        source_image_hash=source_image_hash,
    )
    print(f"Saved baseline: {baseline_path}")

    # Read raw YAML config for embedding in manifest
    with open(Path(args.config), encoding="utf-8") as f:
        raw_config_yaml = f.read()

    sweep_t0 = time.time()

    # Manifest collects metadata for all jobs across all grids
    manifest: dict[str, Any] = {
        "model": config.model,
        "prompt": config.prompt,
        "seed": config.seed,
        "inference": config.inference,
        "baseline": baseline_name,
        "total_jobs": total_jobs,
        "config_yaml": raw_config_yaml,
        "grids": [],
    }
    if source_image_hash:
        manifest["source_image_hash"] = source_image_hash

    from alive_progress import alive_bar

    # Track global job counter for [N/total] progress display
    global_job_num = 0

    for gi, group in enumerate(config.groups):
        print(f"\n--- Grid {gi} ---")
        images: list = []
        grid_jobs_manifest: list[dict[str, Any]] = []
        grid_filename = ""

        with alive_bar(len(group.jobs), title=f"Grid {gi}") as bar:
            for job_idx, job in enumerate(group.jobs):
                t0 = time.time()
                global_job_num += 1

                # Clone original tensors on CPU (~8 GB)
                bent = clone_tensors(original_tensors)
                all_hooks: list = []

                if job.stages:
                    # Pipeline job — apply each stage in order.
                    # Order matters: each stage sees previous stages' changes.
                    validated_stages: list = []
                    for stage in job.stages:
                        if stage.mode_name not in modes:
                            print(f"  [{global_job_num}/{total_jobs}] "
                                  f"Unknown mode in pipeline: {stage.mode_name}")
                            continue
                        mode_cls = modes[stage.mode_name]
                        mode = mode_cls()
                        validated = mode_cls.validate(stage.params, model_info)
                        mode.apply(bent, validated, model_info)
                        all_hooks.extend(mode.get_inference_hooks(validated, model_info))
                        validated_stages.append((stage.mode_name, validated))
                else:
                    # Single-mode job
                    if job.mode_name not in modes:
                        print(f"  [{global_job_num}/{total_jobs}] "
                              f"Unknown mode: {job.mode_name}")
                        del bent
                        bar()
                        continue
                    mode_cls = modes[job.mode_name]
                    mode = mode_cls()
                    validated = mode_cls.validate(job.params, model_info)
                    mode.apply(bent, validated, model_info)
                    all_hooks.extend(mode.get_inference_hooks(validated, model_info))

                # Tensor checksums: sha256[:8] of each modified tensor
                checksums = compute_tensor_checksums(original_tensors, bent)

                # Copy bent weights into pipeline's transformer (bf16 on GPU)
                patch_pipeline(pipeline, bent, model_info.key_mapping)

                # Memory management: the pipeline now holds its own copy of the
                # bent weights. Free the CPU clone and clear any leftover GPU
                # allocations from the previous job.
                del bent
                torch.cuda.empty_cache()

                # Install inference hooks (e.g. rope_warp)
                cleanups = [hook(pipeline) for hook in all_hooks]

                if is_i2i:
                    img = generate_image_i2i(
                        pipeline,
                        source_img,
                        config.prompt,
                        seed=config.seed,
                        steps=config.inference["steps"],
                        guidance_scale=config.inference["guidance_scale"],
                        width=config.inference["width"],
                        height=config.inference["height"],
                    )
                else:
                    img = generate_image(
                        pipeline,
                        config.prompt,
                        seed=config.seed,
                        steps=config.inference["steps"],
                        guidance_scale=config.inference["guidance_scale"],
                        width=config.inference["width"],
                        height=config.inference["height"],
                    )

                # Cleanup hooks after inference
                for cleanup in cleanups:
                    cleanup()

                # Save individual image with PNG tEXt metadata
                if job.stages:
                    img_filename = make_pipeline_image_filename(
                        job.stages, is_i2i=is_i2i,
                    )
                    meta_mode = job.mode_name  # "+"-joined names
                    meta_params = [
                        {"mode": sn, "params": {
                            k: v for k, v in sorted(sv.items())
                            if not k.startswith("_")
                        }}
                        for sn, sv in validated_stages
                    ]
                else:
                    img_filename = make_image_filename(
                        job.mode_name, validated, is_i2i=is_i2i,
                    )
                    meta_mode = job.mode_name
                    meta_params = {
                        k: v for k, v in sorted(validated.items())
                        if not k.startswith("_")
                    }
                save_image_with_metadata(
                    img, out_dir / img_filename,
                    model=config.model,
                    mode=meta_mode,
                    params=meta_params,
                    prompt=config.prompt,
                    seed=config.seed,
                    steps=config.inference["steps"],
                    guidance_scale=config.inference["guidance_scale"],
                    width=config.inference["width"],
                    height=config.inference["height"],
                    tensor_checksums=checksums,
                    source_image_hash=source_image_hash,
                )
                images.append(img)

                elapsed = time.time() - t0

                # Record job in manifest
                job_manifest: dict[str, Any] = {
                    "job_index": job.job_index,
                    "image": img_filename,
                    "grid_position": {
                        "row": job.grid_row_value,
                        "col": job.grid_col_value,
                    },
                    "wall_time_seconds": round(elapsed, 2),
                    "tensor_checksums": checksums,
                }
                if job.stages:
                    manifest_stages = []
                    for stage_name, stage_validated in validated_stages:
                        stage_params = {
                            k: v for k, v in sorted(stage_validated.items())
                            if not k.startswith("_")
                        }
                        manifest_stages.append({
                            "mode": stage_name,
                            "params": stage_params,
                        })
                    job_manifest["pipeline"] = True
                    job_manifest["stages"] = manifest_stages
                else:
                    job_manifest["params"] = {
                        k: v for k, v in sorted(validated.items())
                        if not k.startswith("_")
                    }
                grid_jobs_manifest.append(job_manifest)
                if job.stages:
                    stage_summary = " + ".join(
                        s.mode_name for s in job.stages
                    )
                    print(f"  [{global_job_num}/{total_jobs}] {stage_summary} "
                          f"-- {elapsed:.1f}s")
                else:
                    params_short = ", ".join(
                        f"{k}={v}" for k, v in sorted(validated.items())
                        if not k.startswith("_")
                    )
                    print(f"  [{global_job_num}/{total_jobs}] {job.mode_name} "
                          f"{params_short} -- {elapsed:.1f}s")
                bar()

        # Assemble grid
        if images:
            prompt_short = config.prompt[:60]
            title = (
                f"{config.model} | {group.jobs[0].mode_name} | "
                f'"{prompt_short}" | seed={config.seed}'
            )
            if group.fixed_params:
                fixed_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(group.fixed_params.items())
                )
                title += f" | {fixed_str}"

            grid_img = make_grid(
                images=images,
                row_labels=group.row_labels,
                col_labels=group.col_labels,
                row_param_name=group.row_param_name,
                col_param_name=group.col_param_name,
                title=title,
            )
            grid_filename = f"grid_{gi}_{group.jobs[0].mode_name}"
            if group.group_label:
                grid_filename += f"_{group.group_label.replace(', ', '_').replace('=', '-')}"
            grid_filename += ".png"
            grid_img.save(out_dir / grid_filename)
            print(f"Saved grid: {out_dir / grid_filename}")

        # Record grid in manifest
        manifest["grids"].append({
            "grid_index": gi,
            "mode": group.jobs[0].mode_name if group.jobs else "",
            "group_label": group.group_label,
            "row_param": group.row_param_name,
            "col_param": group.col_param_name,
            "row_labels": group.row_labels,
            "col_labels": group.col_labels,
            "grid_image": grid_filename,
            "jobs": grid_jobs_manifest,
        })

    # Record total sweep timing
    manifest["total_wall_time_seconds"] = round(time.time() - sweep_t0, 2)

    # Save sweep manifest
    manifest_path = out_dir / "sweep_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nSaved manifest: {manifest_path}")

    print("Sweep complete.")


def cmd_audit(args: argparse.Namespace) -> None:
    """Dump tensor information from a model."""
    from flux_bend.loader import load_transformer_tensors, resolve_model_path

    model_dir = resolve_model_path(args.model)
    tensors = load_transformer_tensors(model_dir)

    print(f"\n{'Key':<80s}  {'Shape':<30s}  Dtype")
    print("-" * 120)
    total_params = 0
    for key in sorted(tensors.keys()):
        t = tensors[key]
        shape = tuple(t.shape)
        params = 1
        for s in shape:
            params *= s
        total_params += params
        print(f"{key:<80s}  {str(shape):<30s}  {t.dtype}")

    print(f"\nTotal: {len(tensors)} tensors, {total_params:,} parameters ({total_params / 1e9:.2f}B)")


def cmd_list_modes(args: argparse.Namespace) -> None:
    """List available bending modes and their parameter schemas."""
    from flux_bend.registry import discover_modes

    modes = discover_modes()
    if not modes:
        print("No bending modes found.")
        return

    for name in sorted(modes.keys()):
        mode_cls = modes[name]
        print(f"\n{name}: {mode_cls.description}")
        params = mode_cls.parameters()
        if params:
            for pname, spec in params.items():
                req = "REQUIRED" if spec.required else f"default={spec.default}"
                bounds = ""
                if spec.min is not None or spec.max is not None:
                    parts = []
                    if spec.min is not None:
                        parts.append(f"min={spec.min}")
                    if spec.max is not None:
                        parts.append(f"max={spec.max}")
                    if spec.step is not None:
                        parts.append(f"step={spec.step}")
                    bounds = f" [{', '.join(parts)}]"
                print(f"  {pname} ({spec.type}, {req}){bounds}")
                print(f"    {spec.description}")
        else:
            print("  (no parameters)")


def cmd_test(args: argparse.Namespace) -> None:
    """Smoke test: load pipeline, generate one image, verify no OOM."""
    from flux_bend.inference import (
        generate_image, generate_image_i2i, load_pipeline, load_source_image,
    )

    print("=== flux-bend smoke test ===")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Seed: {args.seed}")
    if args.image:
        print(f"Source image: {args.image}")

    pipeline = load_pipeline(args.model)

    # Load source image for img2img if provided
    source_img = None
    if args.image:
        source_img = load_source_image(Path(args.image), 1024, 1024)

    print("Generating test image...")

    t0 = time.time()
    if source_img is not None:
        image = generate_image_i2i(
            pipeline,
            source_img,
            prompt=args.prompt,
            seed=args.seed,
            steps=50,
            guidance_scale=4.0,
            width=1024,
            height=1024,
        )
    else:
        image = generate_image(
            pipeline,
            prompt=args.prompt,
            seed=args.seed,
            steps=50,
            guidance_scale=4.0,
            width=1024,
            height=1024,
        )
    elapsed = time.time() - t0

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    if source_img is not None:
        out_path = out_dir / f"test_i2i_seed{args.seed}.png"
    else:
        out_path = out_dir / f"test_seed{args.seed}.png"
    image.save(out_path)

    print(f"Image saved: {out_path}")
    print(f"Generation time: {elapsed:.1f}s")
    print(f"Image size: {image.size}")
    print("Smoke test PASSED.")


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Reproduce a single job from a sweep manifest.

    Reads sweep_manifest.json, extracts the job at --job-index,
    reconstructs its params, re-runs bending + inference, and
    optionally verifies output against the original.
    """
    import torch

    from flux_bend.inference import (
        generate_image, generate_image_i2i, load_pipeline, load_source_image,
        patch_pipeline,
    )
    from flux_bend.loader import (
        clone_tensors,
        load_transformer_tensors,
        resolve_model_path,
    )
    from flux_bend.metadata import compute_tensor_checksums, save_image_with_metadata
    from flux_bend.model_info import ModelInfo
    from flux_bend.registry import discover_modes

    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    job_index = args.job_index
    model_info = ModelInfo.klein_4b()
    modes = discover_modes()

    # Find the job across all grids
    target_job: dict | None = None
    target_grid: dict | None = None
    for grid in manifest["grids"]:
        for job in grid["jobs"]:
            if job["job_index"] == job_index:
                target_job = job
                target_grid = grid
                break
        if target_job:
            break

    if target_job is None:
        print(f"Job index {job_index} not found in manifest.")
        print(f"Available indices: ", end="")
        all_indices = [
            j["job_index"] for g in manifest["grids"] for j in g["jobs"]
        ]
        print(sorted(all_indices))
        sys.exit(1)

    model = manifest["model"]
    prompt = manifest["prompt"]
    seed = manifest["seed"]
    inference = manifest.get("inference", {})
    steps = inference.get("steps", 50)
    guidance_scale = inference.get("guidance_scale", 4.0)
    width = inference.get("width", 1024)
    height = inference.get("height", 1024)

    is_pipeline = target_job.get("pipeline", False)

    # Detect img2img from manifest and load source image if provided
    is_i2i = "source_image_hash" in manifest
    source_img = None
    source_image_hash: str | None = manifest.get("source_image_hash")
    if is_i2i:
        if args.image is None:
            print("This manifest was produced with img2img "
                  f"(source_image_hash={source_image_hash}).")
            print("Provide --image to reproduce accurately.")
            sys.exit(1)
        source_img = load_source_image(Path(args.image), width, height)

    print(f"=== Reproducing job {job_index} ===")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    if is_i2i:
        print(f"Source image: {args.image} (hash={source_image_hash})")

    if is_pipeline:
        print(f"Pipeline: {len(target_job['stages'])} stages")
        for si, stage in enumerate(target_job["stages"]):
            print(f"  Stage {si}: {stage['mode']} {stage['params']}")
    else:
        mode_name = target_grid["mode"]
        print(f"Mode: {mode_name}")
        print(f"Params: {target_job['params']}")

    # Load tensors and pipeline
    print("\nLoading model tensors...")
    model_dir = resolve_model_path(model)
    original_tensors = load_transformer_tensors(model_dir)
    bent = clone_tensors(original_tensors)

    all_hooks: list = []

    if is_pipeline:
        for stage in target_job["stages"]:
            smode = stage["mode"]
            sparams = stage["params"]
            if smode not in modes:
                print(f"Unknown mode: {smode}")
                sys.exit(1)
            mode_cls = modes[smode]
            mode = mode_cls()
            validated = mode_cls.validate(sparams, model_info)
            mode.apply(bent, validated, model_info)
            all_hooks.extend(mode.get_inference_hooks(validated, model_info))
    else:
        mode_name = target_grid["mode"]
        if mode_name not in modes:
            print(f"Unknown mode: {mode_name}")
            sys.exit(1)
        mode_cls = modes[mode_name]
        mode = mode_cls()
        validated = mode_cls.validate(target_job["params"], model_info)
        mode.apply(bent, validated, model_info)
        all_hooks.extend(mode.get_inference_hooks(validated, model_info))

    # Tensor checksums for verification
    checksums = compute_tensor_checksums(original_tensors, bent)
    original_checksums = target_job.get("tensor_checksums", {})

    if original_checksums and checksums:
        match = checksums == original_checksums
        print(f"\nTensor checksums: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            for key in sorted(set(checksums.keys()) | set(original_checksums.keys())):
                orig = original_checksums.get(key, "N/A")
                repro = checksums.get(key, "N/A")
                status = "OK" if orig == repro else "DIFF"
                print(f"  {key}: original={orig} reproduced={repro} [{status}]")
    elif checksums:
        print(f"\nTensor checksums ({len(checksums)} modified):")
        for k, v in sorted(checksums.items()):
            print(f"  {k}: {v}")

    # Generate image
    pipeline = load_pipeline(model)
    patch_pipeline(pipeline, bent, model_info.key_mapping)
    del bent
    torch.cuda.empty_cache()

    cleanups = [hook(pipeline) for hook in all_hooks]

    print("\nGenerating image...")
    t0 = time.time()
    if is_i2i:
        image = generate_image_i2i(
            pipeline, source_img, prompt, seed=seed,
            steps=steps, guidance_scale=guidance_scale,
            width=width, height=height,
        )
    else:
        image = generate_image(
            pipeline, prompt, seed=seed,
            steps=steps, guidance_scale=guidance_scale,
            width=width, height=height,
        )
    elapsed = time.time() - t0

    for cleanup in cleanups:
        cleanup()

    # Save reproduced image
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_pipeline:
        meta_mode = "+".join(s["mode"] for s in target_job["stages"])
        meta_params = target_job["stages"]
    else:
        meta_mode = target_grid["mode"]
        meta_params = target_job["params"]

    out_path = out_dir / f"reproduce_job{job_index}.png"
    save_image_with_metadata(
        image, out_path,
        model=model,
        mode=meta_mode,
        params=meta_params,
        prompt=prompt,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        tensor_checksums=checksums,
        source_image_hash=source_image_hash,
    )

    print(f"\nSaved: {out_path}")
    print(f"Generation time: {elapsed:.1f}s")
    print("Reproduce complete.")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare images against a baseline using LPIPS, SSIM, and MSE."""
    from flux_bend.analysis import compare_images

    baseline = Path(args.baseline)
    image_paths = [Path(p) for p in args.images]

    if not baseline.exists():
        print(f"Baseline not found: {baseline}")
        sys.exit(1)

    missing = [p for p in image_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Image not found: {p}")
        sys.exit(1)

    print(f"Baseline: {baseline}")
    print(f"Comparing {len(image_paths)} image(s)...\n")

    results = compare_images(baseline, image_paths)

    # Print table
    header = f"{'Image':<60s}  {'LPIPS':>8s}  {'SSIM':>8s}  {'MSE':>12s}"
    print(header)
    print("-" * len(header))
    for r in results:
        name = Path(r["path"]).name
        if len(name) > 58:
            name = name[:55] + "..."
        print(f"{name:<60s}  {r['lpips']:8.4f}  {r['ssim']:8.4f}  {r['mse']:12.6f}")

    # Summary stats
    if len(results) > 1:
        avg_lpips = sum(r["lpips"] for r in results) / len(results)
        avg_ssim = sum(r["ssim"] for r in results) / len(results)
        avg_mse = sum(r["mse"] for r in results) / len(results)
        print("-" * len(header))
        print(f"{'AVERAGE':<60s}  {avg_lpips:8.4f}  {avg_ssim:8.4f}  {avg_mse:12.6f}")


def cmd_weight_diff(args: argparse.Namespace) -> None:
    """Per-tensor diff metrics between original and bent model weights."""
    from safetensors.torch import load_file

    from flux_bend.analysis import weight_diff_metrics
    from flux_bend.loader import load_transformer_tensors, resolve_model_path

    print("Loading original model tensors...")
    model_dir = resolve_model_path(args.original)
    original_tensors = load_transformer_tensors(model_dir)

    print(f"Loading bent tensors from {args.bent}...")
    bent_path = Path(args.bent)
    if not bent_path.exists():
        print(f"Bent file not found: {bent_path}")
        sys.exit(1)
    bent_tensors = load_file(str(bent_path), device="cpu")

    print(f"\nOriginal: {len(original_tensors)} tensors")
    print(f"Bent:     {len(bent_tensors)} tensors\n")

    results = weight_diff_metrics(original_tensors, bent_tensors)

    # Filter to only modified tensors
    modified = [r for r in results if r["frobenius_ratio"] > 0]

    if not modified:
        print("No tensors were modified.")
        return

    header = (
        f"{'Key':<70s}  {'Shape':<20s}  "
        f"{'Frob Ratio':>10s}  {'Diff Norm':>12s}  "
        f"{'Max Abs':>10s}  {'% Changed':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in modified:
        key = r["key"]
        if len(key) > 68:
            key = key[:65] + "..."
        shape_str = str(r["shape"])
        print(
            f"{key:<70s}  {shape_str:<20s}  "
            f"{r['frobenius_ratio']:10.6f}  {r['diff_norm']:12.4f}  "
            f"{r['max_abs_diff']:10.6f}  {r['pct_changed']:8.2f}%"
        )

    print(f"\n{len(modified)}/{len(results)} tensor(s) modified")


def cmd_block_importance(args: argparse.Namespace) -> None:
    """Compute block importance from a block_dropout sweep via LPIPS."""
    from flux_bend.analysis import block_importance_from_manifest, draw_bar_chart

    sweep_dir = Path(args.sweep_dir)
    manifest_path = sweep_dir / "sweep_manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Sweep dir: {sweep_dir}")
    results = block_importance_from_manifest(sweep_dir)

    if not results:
        print("No block dropout jobs found in manifest.")
        return

    # Print table
    header = f"{'Block Indices':<30s}  {'LPIPS':>8s}  {'Image':<50s}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        idx_str = str(r["block_indices"])
        img_name = Path(r["image"]).name if r["image"] else ""
        if len(img_name) > 48:
            img_name = img_name[:45] + "..."
        print(f"{idx_str:<30s}  {r['lpips']:8.4f}  {img_name:<50s}")

    # Draw bar chart
    labels = [str(r["block_indices"]) for r in results]
    values = [r["lpips"] for r in results]

    chart = draw_bar_chart(
        labels=labels,
        values=values,
        title="Block Importance (LPIPS vs Baseline)",
        x_label="LPIPS (higher = more important)",
    )
    chart_path = sweep_dir / "block_importance.png"
    chart.save(chart_path)
    print(f"\nSaved chart: {chart_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="flux-bend",
        description="Deterministic model bending tool for FLUX.2 Klein Base 4B",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bend
    p_bend = subparsers.add_parser("bend", help="Apply a single bending mode")
    p_bend.add_argument("--model", required=True, help="HF model ID or local path")
    p_bend.add_argument("--mode", required=True, help="Bending mode name")
    p_bend.add_argument("--params", default="", help="Parameters as k=v,k=v")
    p_bend.add_argument("--prompt", default="A cat holding a sign that says hello world", help="Prompt for image generation")
    p_bend.add_argument("--seed", type=int, default=None, help="Inference seed (overrides param seed)")
    p_bend.add_argument("--image", default=None, help="Source image for img2img editing")
    p_bend.add_argument("--out", default="outputs", help="Output directory")
    p_bend.set_defaults(func=cmd_bend)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run a parameter sweep from YAML config")
    p_sweep.add_argument("--config", required=True, help="Path to sweep YAML config")
    p_sweep.add_argument("--out-dir", default="outputs", help="Output directory")
    p_sweep.add_argument("--dry-run", action="store_true", help="Print jobs without executing")
    p_sweep.set_defaults(func=cmd_sweep)

    # audit
    p_audit = subparsers.add_parser("audit", help="Dump tensor info from model")
    p_audit.add_argument("--model", required=True, help="HF model ID or local path")
    p_audit.set_defaults(func=cmd_audit)

    # list-modes
    p_list = subparsers.add_parser("list-modes", help="List available bending modes")
    p_list.set_defaults(func=cmd_list_modes)

    # reproduce
    p_repro = subparsers.add_parser(
        "reproduce", help="Reproduce a job from sweep_manifest.json"
    )
    p_repro.add_argument(
        "manifest", help="Path to sweep_manifest.json"
    )
    p_repro.add_argument(
        "--job-index", type=int, required=True,
        help="Job index to reproduce (from manifest)"
    )
    p_repro.add_argument(
        "--out-dir", default="outputs/reproduce",
        help="Output directory for reproduced image"
    )
    p_repro.add_argument(
        "--image", default=None,
        help="Source image for img2img reproduction (required if manifest has source_image_hash)"
    )
    p_repro.set_defaults(func=cmd_reproduce)

    # compare
    p_compare = subparsers.add_parser(
        "compare", help="Compare images against baseline (LPIPS, SSIM, MSE)"
    )
    p_compare.add_argument(
        "--baseline", required=True, help="Path to baseline image"
    )
    p_compare.add_argument(
        "--images", nargs="+", required=True, help="Paths to images to compare"
    )
    p_compare.set_defaults(func=cmd_compare)

    # weight-diff
    p_wdiff = subparsers.add_parser(
        "weight-diff", help="Per-tensor diff metrics between original and bent weights"
    )
    p_wdiff.add_argument(
        "--original", required=True, help="HF model ID or local path to original model"
    )
    p_wdiff.add_argument(
        "--bent", required=True, help="Path to bent safetensors file"
    )
    p_wdiff.set_defaults(func=cmd_weight_diff)

    # block-importance
    p_bimp = subparsers.add_parser(
        "block-importance",
        help="Compute block importance from block_dropout sweep (LPIPS bar chart)",
    )
    p_bimp.add_argument(
        "--sweep-dir", required=True,
        help="Directory containing sweep_manifest.json and images",
    )
    p_bimp.set_defaults(func=cmd_block_importance)

    # test
    p_test = subparsers.add_parser("test", help="Smoke test: load model, generate one image")
    p_test.add_argument("--model", required=True, help="HF model ID or local path")
    p_test.add_argument("--prompt", default="A cat holding a sign that says hello world")
    p_test.add_argument("--seed", type=int, default=42)
    p_test.add_argument("--image", default=None, help="Source image for img2img smoke test")
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()
    args.func(args)
