"""Dynamic mode discovery and registration."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flux_bend.modes._base import BendingMode


def discover_modes() -> dict[str, type[BendingMode]]:
    """Scan the modes/ package for BendingMode subclasses.

    Returns a dict mapping mode name -> mode class.
    Raises on duplicate names.
    """
    from flux_bend.modes._base import BendingMode

    modes_dir = Path(__file__).parent / "modes"
    registry: dict[str, type[BendingMode]] = {}

    for module_info in pkgutil.iter_modules([str(modes_dir)]):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"flux_bend.modes.{module_info.name}")

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BendingMode)
                and attr is not BendingMode
                and hasattr(attr, "name")
            ):
                mode_name = attr.name
                if mode_name in registry:
                    raise ValueError(
                        f"Duplicate mode name '{mode_name}': "
                        f"{registry[mode_name].__module__} and {attr.__module__}"
                    )
                registry[mode_name] = attr

    return registry
