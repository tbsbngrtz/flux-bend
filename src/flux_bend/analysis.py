"""Analysis utilities: image comparison, weight diffs, block importance.

Requires [analysis] extras: lpips, torchmetrics.
Install with: uv pip install -e ".[analysis]"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from flux_bend.grid import _get_font


def _require_analysis_extras() -> None:
    """Check that [analysis] extras are installed, raise helpful error if not."""
    try:
        import lpips  # noqa: F401
        import torchmetrics  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Analysis extras not installed ({e}). "
            f'Install with: uv pip install -e ".[analysis]"'
        ) from e


def _load_image_tensor(path: Path) -> torch.Tensor:
    """Load a PNG image as a float32 tensor in [0, 1] range, shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB")
    import numpy as np

    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def compare_images(
    baseline_path: Path,
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    """Compute LPIPS, SSIM, and MSE between baseline and each image.

    Returns list of dicts: {path, lpips, ssim, mse}.
    """
    _require_analysis_extras()
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        StructuralSimilarityIndexMeasure,
    )

    from alive_progress import alive_bar

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    baseline = _load_image_tensor(baseline_path).to(device)

    results: list[dict[str, Any]] = []

    with alive_bar(len(image_paths), title="Comparing images") as bar:
        for img_path in image_paths:
            img = _load_image_tensor(img_path).to(device)

            # Resize if dimensions don't match baseline
            if img.shape != baseline.shape:
                img = torch.nn.functional.interpolate(
                    img, size=baseline.shape[2:], mode="bilinear", align_corners=False
                )

            # LPIPS expects inputs in [-1, 1]
            lpips_val = lpips_metric(baseline * 2 - 1, img * 2 - 1).item()

            ssim_val = ssim_metric(img, baseline).item()

            mse_val = torch.nn.functional.mse_loss(img, baseline).item()

            results.append({
                "path": str(img_path),
                "lpips": lpips_val,
                "ssim": ssim_val,
                "mse": mse_val,
            })

            # Reset stateful metrics
            ssim_metric.reset()

            bar()

    return results


def weight_diff_metrics(
    original_tensors: dict[str, torch.Tensor],
    bent_tensors: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Compute per-tensor diff metrics between original and bent weights.

    Returns list sorted by Frobenius ratio descending:
    {key, shape, diff_norm, orig_norm, frobenius_ratio, max_abs_diff, pct_changed}
    """
    from alive_progress import alive_bar

    all_keys = sorted(set(original_tensors.keys()) | set(bent_tensors.keys()))
    results: list[dict[str, Any]] = []

    with alive_bar(len(all_keys), title="Computing weight diffs") as bar:
        for key in all_keys:
            if key not in original_tensors or key not in bent_tensors:
                bar()
                continue

            orig = original_tensors[key].float()
            bent = bent_tensors[key].float()

            if orig.shape != bent.shape:
                bar()
                continue

            diff = bent - orig
            diff_norm = diff.norm().item()
            orig_norm = orig.norm().item()
            frobenius_ratio = diff_norm / orig_norm if orig_norm > 0 else 0.0
            max_abs = diff.abs().max().item()
            num_changed = (diff.abs() > 0).sum().item()
            total_elements = orig.numel()
            pct_changed = 100.0 * num_changed / total_elements if total_elements > 0 else 0.0

            results.append({
                "key": key,
                "shape": tuple(orig.shape),
                "diff_norm": diff_norm,
                "orig_norm": orig_norm,
                "frobenius_ratio": frobenius_ratio,
                "max_abs_diff": max_abs,
                "pct_changed": pct_changed,
            })

            bar()

    # Sort by Frobenius ratio descending (most modified first)
    results.sort(key=lambda r: r["frobenius_ratio"], reverse=True)
    return results


def block_importance_from_manifest(
    sweep_dir: Path,
) -> list[dict[str, Any]]:
    """Read a block_dropout sweep manifest and compute per-block LPIPS.

    Expects a sweep run with block_dropout mode where each job drops
    a single block (or set of blocks). Computes LPIPS between the
    baseline and each job image.

    Returns list of {block_indices, lpips, image} sorted by block index.
    """
    _require_analysis_extras()
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    from alive_progress import alive_bar

    manifest_path = sweep_dir / "sweep_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    baseline_path = sweep_dir / manifest["baseline"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    baseline = _load_image_tensor(baseline_path).to(device)

    results: list[dict[str, Any]] = []

    # Collect all jobs across grids
    all_jobs: list[dict[str, Any]] = []
    for grid in manifest["grids"]:
        for job in grid["jobs"]:
            all_jobs.append(job)

    with alive_bar(len(all_jobs), title="Computing block importance") as bar:
        for job in all_jobs:
            params = job.get("params", {})
            block_indices = params.get("block_indices", [])
            img_file = job.get("image", "")

            img_path = sweep_dir / img_file
            if not img_path.exists():
                print(f"  WARNING: image not found: {img_path}")
                bar()
                continue

            img = _load_image_tensor(img_path).to(device)
            if img.shape != baseline.shape:
                img = torch.nn.functional.interpolate(
                    img, size=baseline.shape[2:], mode="bilinear", align_corners=False
                )

            lpips_val = lpips_metric(baseline * 2 - 1, img * 2 - 1).item()

            results.append({
                "block_indices": block_indices,
                "lpips": lpips_val,
                "image": img_file,
                "wall_time_seconds": job.get("wall_time_seconds"),
            })

            bar()

    # Sort by first block index
    results.sort(key=lambda r: r["block_indices"][0] if r["block_indices"] else 999)
    return results


def draw_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    width: int = 800,
    bar_height: int = 30,
    margin: int = 40,
    color: tuple[int, int, int] = (66, 133, 244),
) -> Image.Image:
    """Draw a horizontal bar chart using Pillow (no matplotlib).

    Args:
        labels: Bar labels (one per bar).
        values: Numeric values (one per bar).
        title: Chart title.
        x_label: Label for the value axis.
        y_label: Label for the category axis.
        width: Image width in pixels.
        bar_height: Height of each bar in pixels.
        margin: Margin around the chart area.
        color: RGB color for bars.
    """
    font = _get_font(14)
    title_font = _get_font(16)
    small_font = _get_font(12)

    n_bars = len(labels)
    bar_spacing = 8
    label_area_width = 160
    title_area = 50 if title else 0
    x_label_area = 30 if x_label else 0

    chart_height = (
        title_area + margin
        + n_bars * (bar_height + bar_spacing)
        + margin + x_label_area
    )
    img = Image.new("RGB", (width, chart_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) // 2, 10), title, fill=(0, 0, 0), font=title_font)

    # Chart area
    chart_left = margin + label_area_width
    chart_right = width - margin - 60  # leave room for value text
    chart_top = title_area + margin
    max_val = max(values) if values else 1.0
    if max_val == 0:
        max_val = 1.0

    # Draw bars
    for i, (label, val) in enumerate(zip(labels, values)):
        y = chart_top + i * (bar_height + bar_spacing)

        # Label
        bbox = draw.textbbox((0, 0), label, font=font)
        text_h = bbox[3] - bbox[1]
        draw.text(
            (margin, y + (bar_height - text_h) // 2),
            label, fill=(0, 0, 0), font=font,
        )

        # Bar
        bar_width = int((val / max_val) * (chart_right - chart_left))
        bar_width = max(bar_width, 2)  # minimum visible bar
        draw.rectangle(
            [chart_left, y, chart_left + bar_width, y + bar_height],
            fill=color,
        )

        # Value text
        val_str = f"{val:.4f}"
        draw.text(
            (chart_left + bar_width + 5, y + (bar_height - text_h) // 2),
            val_str, fill=(80, 80, 80), font=small_font,
        )

    # X axis label
    if x_label:
        bbox = draw.textbbox((0, 0), x_label, font=font)
        tw = bbox[2] - bbox[0]
        y = chart_top + n_bars * (bar_height + bar_spacing) + 10
        draw.text(((chart_left + chart_right - tw) // 2, y), x_label, fill=(0, 0, 0), font=font)

    return img
