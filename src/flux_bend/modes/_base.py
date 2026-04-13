"""Base class for all bending modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch

from flux_bend.model_info import ModelInfo


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single mode parameter."""

    name: str
    type: Literal["int", "float", "str", "list[int]"]
    required: bool
    description: str
    default: Any = None
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None


class BendingMode(ABC):
    """Abstract base class for all bending modes.

    Each mode modifies transformer weight tensors in-place to produce
    deterministic glitch effects. Subclasses must define:
    - name: str matching the filename (without .py)
    - description: str
    - parameters() classmethod returning param schema
    - apply() method performing the actual tensor modification
    """

    name: str
    description: str

    @classmethod
    @abstractmethod
    def parameters(cls) -> dict[str, ParamSpec]:
        """Return parameter schema for this mode."""
        ...

    @classmethod
    def validate(cls, params: dict[str, Any], model_info: ModelInfo) -> dict[str, Any]:
        """Validate params against schema, fill defaults, raise on errors."""
        specs = cls.parameters()
        validated: dict[str, Any] = {}

        for spec_name, spec in specs.items():
            if spec_name in params:
                value = params[spec_name]
                value = _coerce_type(value, spec)
                _check_bounds(value, spec)
                validated[spec_name] = value
            elif spec.required:
                raise ValueError(
                    f"Mode '{cls.name}': required parameter '{spec_name}' not provided"
                )
            else:
                validated[spec_name] = spec.default

        unknown = set(params.keys()) - set(specs.keys())
        if unknown:
            raise ValueError(
                f"Mode '{cls.name}': unknown parameters: {sorted(unknown)}"
            )

        return validated

    @abstractmethod
    def apply(
        self,
        tensors: dict[str, torch.Tensor],
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> None:
        """Modify tensors in-place. Must be deterministic given the same params."""
        ...

    def get_inference_hooks(
        self,
        params: dict[str, Any],
        model_info: ModelInfo,
    ) -> list[Any]:
        """Return runtime hooks for modifications that cannot be done via weights.

        Each hook is a callable: install_fn(pipeline) -> cleanup_fn().
        The CLI calls install_fn before inference and cleanup_fn after.

        Default: empty list (no hooks needed — most modes are weights-only).
        Override this for modes like rope_warp where the target computation
        happens at runtime rather than being stored in weight tensors.
        """
        return []

    @staticmethod
    def check_bf16_survival(
        key: str, original: torch.Tensor, bent_f32: torch.Tensor
    ) -> None:
        """Check if a modification survives bf16 casting.

        Compares the bent tensor (in f32) cast to bf16 against the original bf16.
        Logs a warning if max abs diff is zero (intervention below precision floor).
        """
        bent_bf16 = bent_f32.to(torch.bfloat16)
        original_bf16 = original.to(torch.bfloat16)
        max_diff = (bent_bf16.float() - original_bf16.float()).abs().max().item()
        if max_diff == 0.0:
            print(
                f"  WARNING: modification to '{key}' did not survive bf16 casting "
                f"(max abs diff = 0). Intervention is below bf16 precision floor."
            )
        else:
            print(f"  {key}: max abs diff after bf16 round-trip = {max_diff:.6e}")


def _coerce_type(value: Any, spec: ParamSpec) -> Any:
    """Coerce a parameter value to the expected type."""
    if spec.type == "int":
        if not isinstance(value, int):
            value = int(value)
        return value
    elif spec.type == "float":
        if not isinstance(value, float):
            value = float(value)
        return value
    elif spec.type == "str":
        return str(value)
    elif spec.type == "list[int]":
        if isinstance(value, str):
            value = [int(x.strip()) for x in value.split(",")]
        elif isinstance(value, (list, tuple)):
            value = [int(x) for x in value]
        else:
            raise TypeError(
                f"Parameter '{spec.name}': expected list[int], got {type(value).__name__}"
            )
        return value
    else:
        raise ValueError(f"Unknown param type: {spec.type}")


def _check_bounds(value: Any, spec: ParamSpec) -> None:
    """Check min/max bounds for numeric parameters."""
    if spec.type in ("int", "float") and isinstance(value, (int, float)):
        if spec.min is not None and value < spec.min:
            raise ValueError(
                f"Parameter '{spec.name}': {value} < min {spec.min}"
            )
        if spec.max is not None and value > spec.max:
            raise ValueError(
                f"Parameter '{spec.name}': {value} > max {spec.max}"
            )
    elif spec.type == "list[int]" and isinstance(value, list):
        for item in value:
            if spec.min is not None and item < spec.min:
                raise ValueError(
                    f"Parameter '{spec.name}': item {item} < min {spec.min}"
                )
            if spec.max is not None and item > spec.max:
                raise ValueError(
                    f"Parameter '{spec.name}': item {item} > max {spec.max}"
                )
