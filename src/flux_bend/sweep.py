"""YAML sweep config parsing and cartesian product expansion."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineStage:
    """One stage in a pipeline job — a mode with resolved parameters."""

    mode_name: str
    params: dict[str, Any]


@dataclass
class SweepJob:
    """A single job in a sweep — one set of mode parameters to apply.

    For single-mode jobs: mode_name and params are used directly, stages is empty.
    For pipeline jobs: stages holds the ordered list of modes to apply,
    mode_name is the "+"-joined names (for display/naming), and params is empty.
    """

    mode_name: str
    params: dict[str, Any]
    grid_row_value: Any
    grid_col_value: Any
    job_index: int
    stages: list[PipelineStage] = field(default_factory=list)


@dataclass
class SweepGroup:
    """A group of jobs that form one grid image."""

    jobs: list[SweepJob]
    row_param_name: str | None
    col_param_name: str | None
    row_labels: list[str]
    col_labels: list[str]
    fixed_params: dict[str, Any] = field(default_factory=dict)
    group_label: str = ""


@dataclass
class SweepConfig:
    """Top-level sweep configuration parsed from YAML."""

    model: str
    prompt: str
    seed: int
    inference: dict[str, Any]
    groups: list[SweepGroup]
    image: Path | None = None


def _expand_range(spec: dict[str, float]) -> list[float]:
    """Expand a range spec like {min: 0.1, max: 1.0, step: 0.3} into a list."""
    lo = float(spec["min"])
    hi = float(spec["max"])
    step = float(spec["step"])
    values: list[float] = []
    current = lo
    while current <= hi + step * 0.01:  # small epsilon for float precision
        values.append(round(current, 10))
        current += step
    return values


def _is_range_spec(value: Any) -> bool:
    """Check if a value is a range spec dict."""
    return (
        isinstance(value, dict)
        and "min" in value
        and "max" in value
        and "step" in value
    )


def _expand_param_values(value: Any) -> list[Any]:
    """Expand a parameter value into a list of values for sweeping."""
    if _is_range_spec(value):
        return _expand_range(value)
    elif isinstance(value, list):
        return value
    else:
        return [value]


def _format_label(value: Any) -> str:
    """Format a parameter value as a grid label."""
    if isinstance(value, float):
        return f"{value:.2f}"
    elif isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def _make_stage_keys(stages_raw: list[dict[str, Any]]) -> list[str]:
    """Assign unique keys to pipeline stages, suffixing duplicates.

    E.g. two mlp_low_rank stages become ["mlp_low_rank", "mlp_low_rank_2"].
    """
    counts: dict[str, int] = {}
    keys: list[str] = []
    for stage in stages_raw:
        name = stage["mode"]
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            keys.append(name)
        else:
            keys.append(f"{name}_{counts[name]}")
    return keys


def _parse_pipeline_entry(sweep_entry: dict[str, Any]) -> list[SweepGroup]:
    """Parse a pipeline sweep entry into one or more SweepGroups.

    Pipeline entries have a ``pipeline:`` key with a list of stages,
    each with ``mode`` and ``params``. Swept params from ALL stages
    participate in a single cartesian product. Grid param references
    use ``stage_key.param_name`` to disambiguate (e.g. ``mlp_low_rank.alpha``).

    Order matters: stages are applied sequentially, so the second mode
    sees the first mode's weight changes.
    """
    stages_raw = sweep_entry["pipeline"]
    grid_spec = sweep_entry.get("grid", {})
    row_param = grid_spec.get("rows")
    col_param = grid_spec.get("cols")

    # Warn about unrecognized keys at the pipeline entry level
    _KNOWN_PIPELINE_KEYS = {"pipeline", "grid"}
    unknown_keys = set(sweep_entry.keys()) - _KNOWN_PIPELINE_KEYS
    if unknown_keys:
        print(f"WARNING: pipeline sweep entry has unrecognized keys "
              f"at the entry level: {sorted(unknown_keys)}. "
              f"Did you mean to put them inside a stage's 'params:'?")

    stage_keys = _make_stage_keys(stages_raw)
    pipeline_name = "+".join(s["mode"] for s in stages_raw)

    # Separate swept vs fixed params across all stages.
    # Namespaced key format: "stage_key.param_name"
    swept: dict[str, list[Any]] = {}       # namespaced key -> values
    fixed: dict[str, Any] = {}             # namespaced key -> value
    stage_fixed: list[dict[str, Any]] = [] # per-stage fixed params (original names)

    for si, (stage, key) in enumerate(zip(stages_raw, stage_keys)):
        params_raw = stage.get("params", {})
        sf: dict[str, Any] = {}
        for pname, pval in params_raw.items():
            ns_key = f"{key}.{pname}"
            expanded = _expand_param_values(pval)
            if len(expanded) > 1:
                swept[ns_key] = expanded
            else:
                fixed[ns_key] = expanded[0]
                sf[pname] = expanded[0]
        stage_fixed.append(sf)

    num_swept = len(swept)

    def _build_job(
        combo: dict[str, Any],
        row_val: Any,
        col_val: Any,
        idx: int,
    ) -> SweepJob:
        """Build a SweepJob from a namespaced param combination."""
        stages: list[PipelineStage] = []
        for si, (stage_raw, key) in enumerate(zip(stages_raw, stage_keys)):
            params: dict[str, Any] = dict(stage_fixed[si])
            # Add swept params for this stage (strip namespace)
            for ns_key, val in combo.items():
                if ns_key.startswith(key + "."):
                    pname = ns_key[len(key) + 1:]
                    params[pname] = val
            stages.append(PipelineStage(mode_name=stage_raw["mode"], params=params))

        return SweepJob(
            mode_name=pipeline_name,
            params={},
            grid_row_value=row_val,
            grid_col_value=col_val,
            job_index=idx,
            stages=stages,
        )

    groups: list[SweepGroup] = []

    if num_swept == 0:
        job = _build_job({}, None, None, 0)
        groups.append(SweepGroup(
            jobs=[job],
            row_param_name=None,
            col_param_name=None,
            row_labels=[],
            col_labels=[],
        ))

    elif num_swept == 1:
        ns_key = list(swept.keys())[0]
        values = swept[ns_key]
        col_p = col_param or ns_key

        jobs: list[SweepJob] = []
        for idx, val in enumerate(values):
            jobs.append(_build_job({ns_key: val}, None, val, idx))

        groups.append(SweepGroup(
            jobs=jobs,
            row_param_name=None,
            col_param_name=col_p,
            row_labels=[""],
            col_labels=[_format_label(v) for v in values],
        ))

    elif num_swept == 2:
        swept_names = list(swept.keys())
        if row_param and row_param in swept_names:
            r_param = row_param
            c_param = [n for n in swept_names if n != row_param][0]
        elif col_param and col_param in swept_names:
            c_param = col_param
            r_param = [n for n in swept_names if n != col_param][0]
        else:
            r_param, c_param = swept_names[0], swept_names[1]

        row_values = swept[r_param]
        col_values = swept[c_param]

        jobs = []
        idx = 0
        for rv in row_values:
            for cv in col_values:
                combo = {r_param: rv, c_param: cv}
                jobs.append(_build_job(combo, rv, cv, idx))
                idx += 1

        groups.append(SweepGroup(
            jobs=jobs,
            row_param_name=r_param,
            col_param_name=c_param,
            row_labels=[_format_label(v) for v in row_values],
            col_labels=[_format_label(v) for v in col_values],
        ))

    else:
        # 3+ swept params -> one grid per extra param combo
        swept_names = list(swept.keys())
        if row_param and col_param:
            r_param = row_param
            c_param = col_param
        else:
            r_param = swept_names[0]
            c_param = swept_names[1]

        extra_params = {
            n: swept[n] for n in swept_names if n not in (r_param, c_param)
        }
        extra_names = sorted(extra_params.keys())
        extra_combos = list(itertools.product(
            *(extra_params[n] for n in extra_names)
        ))

        row_values = swept[r_param]
        col_values = swept[c_param]

        for extra_vals in extra_combos:
            extra_fixed = dict(zip(extra_names, extra_vals))
            label_parts = [f"{n}={_format_label(v)}" for n, v in extra_fixed.items()]
            group_label = ", ".join(label_parts)

            jobs = []
            idx = 0
            for rv in row_values:
                for cv in col_values:
                    combo = {r_param: rv, c_param: cv}
                    combo.update(extra_fixed)
                    jobs.append(_build_job(combo, rv, cv, idx))
                    idx += 1

            groups.append(SweepGroup(
                jobs=jobs,
                row_param_name=r_param,
                col_param_name=c_param,
                row_labels=[_format_label(v) for v in row_values],
                col_labels=[_format_label(v) for v in col_values],
                group_label=group_label,
            ))

    return groups


def parse_sweep_config(config_path: Path) -> SweepConfig:
    """Parse a YAML sweep config file into a SweepConfig."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model = raw["model"]
    prompt = raw["prompt"]
    seed = int(raw["seed"])
    inference = raw.get("inference", {})
    inference.setdefault("steps", 50)
    inference.setdefault("guidance_scale", 4.0)
    inference.setdefault("width", 1024)
    inference.setdefault("height", 1024)

    # Optional source image for img2img — resolve relative to config directory
    image_path: Path | None = None
    if "image" in raw:
        raw_image = Path(raw["image"])
        if raw_image.is_absolute():
            image_path = raw_image
        else:
            image_path = (config_path.parent / raw_image).resolve()

    all_groups: list[SweepGroup] = []

    for sweep_entry in raw.get("sweeps", []):
        if "pipeline" in sweep_entry:
            groups = _parse_pipeline_entry(sweep_entry)
            all_groups.extend(groups)
            continue

        mode_name = sweep_entry["mode"]
        params_raw = sweep_entry.get("params", {})
        grid_spec = sweep_entry.get("grid", {})
        row_param = grid_spec.get("rows")
        col_param = grid_spec.get("cols")

        # Warn about unrecognized keys — often a sign of misplaced params
        _KNOWN_ENTRY_KEYS = {"mode", "params", "grid"}
        unknown_keys = set(sweep_entry.keys()) - _KNOWN_ENTRY_KEYS
        if unknown_keys:
            print(f"WARNING: sweep entry for '{mode_name}' has unrecognized keys "
                  f"at the entry level: {sorted(unknown_keys)}. "
                  f"Did you mean to put them inside 'params:'?")

        # Identify swept vs fixed params
        swept: dict[str, list[Any]] = {}
        fixed: dict[str, Any] = {}
        for pname, pval in params_raw.items():
            expanded = _expand_param_values(pval)
            if len(expanded) > 1:
                swept[pname] = expanded
            else:
                fixed[pname] = expanded[0]

        num_swept = len(swept)

        if num_swept == 0:
            # No sweep — single job
            job = SweepJob(
                mode_name=mode_name,
                params=dict(fixed),
                grid_row_value=None,
                grid_col_value=None,
                job_index=0,
            )
            group = SweepGroup(
                jobs=[job],
                row_param_name=None,
                col_param_name=None,
                row_labels=[],
                col_labels=[],
                fixed_params=fixed,
            )
            all_groups.append(group)

        elif num_swept == 1:
            # Single swept param -> single row, param on cols
            param_name = list(swept.keys())[0]
            values = swept[param_name]
            col_param = col_param or param_name

            jobs: list[SweepJob] = []
            for idx, val in enumerate(values):
                p = dict(fixed)
                p[param_name] = val
                jobs.append(SweepJob(
                    mode_name=mode_name,
                    params=p,
                    grid_row_value=None,
                    grid_col_value=val,
                    job_index=idx,
                ))

            group = SweepGroup(
                jobs=jobs,
                row_param_name=None,
                col_param_name=col_param,
                row_labels=[""],
                col_labels=[_format_label(v) for v in values],
                fixed_params=fixed,
            )
            all_groups.append(group)

        elif num_swept == 2:
            # Two swept params -> rows × cols grid
            swept_names = list(swept.keys())
            if row_param and row_param in swept_names:
                r_param = row_param
                c_param = [n for n in swept_names if n != row_param][0]
            elif col_param and col_param in swept_names:
                c_param = col_param
                r_param = [n for n in swept_names if n != col_param][0]
            else:
                r_param, c_param = swept_names[0], swept_names[1]

            row_values = swept[r_param]
            col_values = swept[c_param]

            jobs = []
            idx = 0
            for rv in row_values:
                for cv in col_values:
                    p = dict(fixed)
                    p[r_param] = rv
                    p[c_param] = cv
                    jobs.append(SweepJob(
                        mode_name=mode_name,
                        params=p,
                        grid_row_value=rv,
                        grid_col_value=cv,
                        job_index=idx,
                    ))
                    idx += 1

            group = SweepGroup(
                jobs=jobs,
                row_param_name=r_param,
                col_param_name=c_param,
                row_labels=[_format_label(v) for v in row_values],
                col_labels=[_format_label(v) for v in col_values],
                fixed_params=fixed,
            )
            all_groups.append(group)

        else:
            # 3+ swept params -> one grid per value of extra params
            swept_names = list(swept.keys())
            # Use grid spec to determine row/col, rest are "extra"
            if row_param and col_param:
                r_param = row_param
                c_param = col_param
            else:
                r_param = swept_names[0]
                c_param = swept_names[1]

            extra_params = {
                n: swept[n] for n in swept_names if n not in (r_param, c_param)
            }
            extra_names = sorted(extra_params.keys())
            extra_combos = list(itertools.product(
                *(extra_params[n] for n in extra_names)
            ))

            row_values = swept[r_param]
            col_values = swept[c_param]

            for extra_vals in extra_combos:
                extra_fixed = dict(zip(extra_names, extra_vals))
                label_parts = [f"{n}={_format_label(v)}" for n, v in extra_fixed.items()]
                group_label = ", ".join(label_parts)

                jobs = []
                idx = 0
                for rv in row_values:
                    for cv in col_values:
                        p = dict(fixed)
                        p.update(extra_fixed)
                        p[r_param] = rv
                        p[c_param] = cv
                        jobs.append(SweepJob(
                            mode_name=mode_name,
                            params=p,
                            grid_row_value=rv,
                            grid_col_value=cv,
                            job_index=idx,
                        ))
                        idx += 1

                all_fixed = dict(fixed)
                all_fixed.update(extra_fixed)
                group = SweepGroup(
                    jobs=jobs,
                    row_param_name=r_param,
                    col_param_name=c_param,
                    row_labels=[_format_label(v) for v in row_values],
                    col_labels=[_format_label(v) for v in col_values],
                    fixed_params=all_fixed,
                    group_label=group_label,
                )
                all_groups.append(group)

    total_jobs = sum(len(g.jobs) for g in all_groups)
    print(f"Parsed {len(all_groups)} grid(s), {total_jobs} total job(s)")

    return SweepConfig(
        model=model,
        prompt=prompt,
        seed=seed,
        inference=inference,
        groups=all_groups,
        image=image_path,
    )
