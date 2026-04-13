"""Deterministic output filename generation."""

from __future__ import annotations

import hashlib
from typing import Any

MAX_FILENAME_LENGTH = 180


def _format_value(value: Any) -> str:
    """Format a parameter value for use in filenames."""
    if isinstance(value, float):
        return f"{value:.2f}"
    elif isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    else:
        return str(value)


def make_filename(mode_name: str, params: dict[str, Any], *, is_i2i: bool = False) -> str:
    """Generate a deterministic filename stem from mode name and parameters.

    Format: klein4b_{mode}_{param1}-{val1}_{param2}-{val2}
    When is_i2i is True, prefixes with i2i_: i2i_klein4b_{mode}_{params}
    Falls back to a hash if the name exceeds MAX_FILENAME_LENGTH chars.
    """
    prefix = "i2i_klein4b" if is_i2i else "klein4b"
    parts = [f"{prefix}_{mode_name}"]
    for key in sorted(params.keys()):
        value = _format_value(params[key])
        parts.append(f"{key}-{value}")
    stem = "_".join(parts)

    if len(stem) > MAX_FILENAME_LENGTH:
        param_str = "_".join(
            f"{k}-{_format_value(params[k])}" for k in sorted(params.keys())
        )
        h = hashlib.sha256(param_str.encode()).hexdigest()[:12]
        stem = f"{prefix}_{mode_name}_{h}"

    return stem


def make_image_filename(mode_name: str, params: dict[str, Any], *, is_i2i: bool = False) -> str:
    """Generate filename for a single bent image."""
    return make_filename(mode_name, params, is_i2i=is_i2i) + ".png"


def make_grid_filename(mode_name: str, params: dict[str, Any], *, is_i2i: bool = False) -> str:
    """Generate filename for a grid image."""
    return make_filename(mode_name, params, is_i2i=is_i2i) + "_grid.png"


def make_pipeline_image_filename(
    stages: list[Any],
    *,
    is_i2i: bool = False,
) -> str:
    """Generate filename for a pipeline (multi-mode) job.

    Concatenates per-stage filenames with ``+``.
    Falls back to hash if the combined name exceeds MAX_FILENAME_LENGTH.

    Args:
        stages: list of PipelineStage (or any object with .mode_name and .params).
        is_i2i: If True, prefix each stage filename with i2i_.
    """
    parts: list[str] = []
    for stage in stages:
        filtered = {
            k: v for k, v in sorted(stage.params.items())
            if not k.startswith("_")
        }
        # Per-stage filenames never get the i2i_ prefix — it's added once below
        parts.append(make_filename(stage.mode_name, filtered))
    stem = "+".join(parts)
    if is_i2i:
        stem = f"i2i_{stem}"

    if len(stem) > MAX_FILENAME_LENGTH:
        h = hashlib.sha256(stem.encode()).hexdigest()[:12]
        mode_names = "+".join(s.mode_name for s in stages)
        prefix = "i2i_klein4b" if is_i2i else "klein4b"
        stem = f"{prefix}_{mode_names}_{h}"

    return stem + ".png"
