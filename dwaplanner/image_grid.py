"""Image-backed traversability grid utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class ImageGrid:
    voxel_size: float
    min_ix: int
    min_iy: int
    height_index: np.ndarray
    state: np.ndarray
    passable_mask: np.ndarray
    map_rgb_top_down: np.ndarray
    source_path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.state.shape


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        path = input_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        return path

    candidates: list[Path] = []
    for pattern in ("*.png", "*.pgm", "*.ppm", "*.jpg", "*.jpeg", "*.bmp"):
        candidates.extend(sorted((Path.cwd() / "maps").glob(pattern)))
    if not candidates:
        raise FileNotFoundError("No input image provided and no files were found under ./maps.")
    return candidates[0].resolve()


def _ensure_grayscale(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim == 2:
        return image_array.astype(np.uint8)
    if image_array.ndim == 3 and image_array.shape[2] >= 3:
        rgb = image_array[:, :, :3].astype(np.float64)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        return np.clip(gray, 0.0, 255.0).astype(np.uint8)
    raise ValueError("Unsupported image array shape; expected HxW or HxWx3.")


def load_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _inflate_obstacles(free_mask: np.ndarray, inflation_cells: int) -> np.ndarray:
    if inflation_cells <= 0:
        return free_mask.copy()

    inflated_free = free_mask.copy()
    obstacle_rows, obstacle_cols = np.where(~free_mask)
    for row, col in zip(obstacle_rows, obstacle_cols):
        row_min = max(0, row - inflation_cells)
        row_max = min(free_mask.shape[0], row + inflation_cells + 1)
        col_min = max(0, col - inflation_cells)
        col_max = min(free_mask.shape[1], col + inflation_cells + 1)
        for neighbor_row in range(row_min, row_max):
            for neighbor_col in range(col_min, col_max):
                if (neighbor_row - row) ** 2 + (neighbor_col - col) ** 2 <= inflation_cells**2:
                    inflated_free[neighbor_row, neighbor_col] = False
    return inflated_free


def _build_passable_mask(state: np.ndarray) -> np.ndarray:
    rows, cols = state.shape
    passable_mask = np.zeros((rows, cols, 8), dtype=bool)
    offsets = (
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    )
    for row in range(rows):
        for col in range(cols):
            if state[row, col] == 0:
                continue
            for direction_index, (dx, dy) in enumerate(offsets):
                neighbor_row = row + dy
                neighbor_col = col + dx
                if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                    passable_mask[row, col, direction_index] = bool(state[neighbor_row, neighbor_col] != 0)
    return passable_mask


def build_grid_from_image_array(
    image_array: np.ndarray,
    meters_per_pixel: float,
    obstacle_threshold: int = 127,
    obstacle_inflation_cells: int = 0,
    source_path: Path | None = None,
) -> ImageGrid:
    if meters_per_pixel <= 0.0:
        raise ValueError("meters_per_pixel must be positive.")
    if obstacle_threshold < 0 or obstacle_threshold > 255:
        raise ValueError("obstacle_threshold must be in [0, 255].")
    if obstacle_inflation_cells < 0:
        raise ValueError("obstacle_inflation_cells cannot be negative.")

    grayscale = _ensure_grayscale(image_array)
    free_mask_top_down = grayscale > obstacle_threshold
    free_mask_top_down = _inflate_obstacles(free_mask_top_down, obstacle_inflation_cells)

    rows, cols = free_mask_top_down.shape
    state = np.flipud(free_mask_top_down).astype(np.uint8)
    height_index = np.zeros((rows, cols), dtype=np.int32)
    passable_mask = _build_passable_mask(state)

    base_rgb_top_down = np.zeros((rows, cols, 3), dtype=np.uint8)
    base_rgb_top_down[free_mask_top_down] = np.asarray([244, 244, 244], dtype=np.uint8)
    base_rgb_top_down[~free_mask_top_down] = np.asarray([28, 28, 28], dtype=np.uint8)

    return ImageGrid(
        voxel_size=meters_per_pixel,
        min_ix=-(cols // 2),
        min_iy=0,
        height_index=height_index,
        state=state,
        passable_mask=passable_mask,
        map_rgb_top_down=base_rgb_top_down,
        source_path=source_path,
    )


def load_image_grid(
    input_path: Path | None,
    meters_per_pixel: float,
    obstacle_threshold: int = 127,
    obstacle_inflation_cells: int = 0,
) -> ImageGrid:
    resolved_path = resolve_input_path(input_path)
    image_array = load_image_array(resolved_path)
    return build_grid_from_image_array(
        image_array=image_array,
        meters_per_pixel=meters_per_pixel,
        obstacle_threshold=obstacle_threshold,
        obstacle_inflation_cells=obstacle_inflation_cells,
        source_path=resolved_path,
    )


def print_grid_stats(grid: ImageGrid) -> None:
    occupied = int(np.count_nonzero(grid.state))
    total = int(grid.state.size)
    print(f"input_image: {grid.source_path if grid.source_path is not None else '<array>'}")
    print(f"grid_shape_rows_cols: {grid.state.shape[0]} x {grid.state.shape[1]}")
    print(f"meters_per_pixel: {grid.voxel_size:.3f}")
    print(f"traversable_cells: {occupied}")
    print(f"blocked_cells: {total - occupied}")
