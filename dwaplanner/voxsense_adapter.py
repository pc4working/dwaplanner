"""Helpers for importing the sibling voxsense repository."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("VOXSENSE_REPO")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append((_repo_root().parent / "voxsense").resolve())
    return candidates


def _ensure_voxsense_on_path() -> Path:
    for root in _candidate_roots():
        if (root / "test_directional_traversability_vis.py").is_file():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return root
    raise ModuleNotFoundError(
        "Could not locate the sibling voxsense repository. "
        "Set VOXSENSE_REPO to its path if it is not next to this project."
    )


@lru_cache(maxsize=1)
def directional_module() -> ModuleType:
    _ensure_voxsense_on_path()
    return importlib.import_module("test_directional_traversability_vis")


@lru_cache(maxsize=1)
def single_frame_module() -> ModuleType:
    _ensure_voxsense_on_path()
    return importlib.import_module("test_single_frame_voxel_vis")


def build_traversability_run(*args: Any, **kwargs: Any) -> Any:
    if args and isinstance(args[0], str):
        args = (Path(args[0]).expanduser(), *args[1:])
    if "input_path" in kwargs and isinstance(kwargs["input_path"], str):
        kwargs["input_path"] = Path(kwargs["input_path"]).expanduser()
    return directional_module().build_traversability_run(*args, **kwargs)


def build_grid_mesh(*args: Any, **kwargs: Any) -> Any:
    return directional_module().build_grid_mesh(*args, **kwargs)


def build_direction_lines(*args: Any, **kwargs: Any) -> Any:
    return directional_module().build_direction_lines(*args, **kwargs)


def build_render_point_cloud(*args: Any, **kwargs: Any) -> Any:
    return directional_module().build_render_point_cloud(*args, **kwargs)


def print_run_stats(*args: Any, **kwargs: Any) -> Any:
    return directional_module().print_run_stats(*args, **kwargs)


def resolve_input_path(*args: Any, **kwargs: Any) -> Any:
    if args and isinstance(args[0], str):
        args = (Path(args[0]).expanduser(), *args[1:])
    if "input_path" in kwargs and isinstance(kwargs["input_path"], str):
        kwargs["input_path"] = Path(kwargs["input_path"]).expanduser()
    return single_frame_module().resolve_input_path(*args, **kwargs)


def exports() -> dict[str, Any]:
    module = directional_module()
    return {
        "CellState": module.CellState,
        "DIRECTION_NAMES": module.DIRECTION_NAMES,
        "DIRECTION_OFFSETS": module.DIRECTION_OFFSETS,
        "DIRECTION_UNIT_VECTORS": module.DIRECTION_UNIT_VECTORS,
        "TraversabilityGrid": module.TraversabilityGrid,
        "TraversabilityRun": module.TraversabilityRun,
    }
