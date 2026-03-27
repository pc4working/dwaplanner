"""2D image visualization helpers for the DWA planner."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from .dwa_planner import DWAResult, RobotState


def _trajectory_score_bounds(result: DWAResult) -> tuple[float, float]:
    valid_scores = [candidate.score for candidate in result.candidates if candidate.valid]
    if not valid_scores:
        return 0.0, 1.0
    return min(valid_scores), max(valid_scores)


def _score_to_color(score: float, min_score: float, max_score: float) -> tuple[int, int, int]:
    if not math.isfinite(score):
        return (150, 150, 150)
    if math.isclose(min_score, max_score):
        return (32, 210, 64)
    ratio = float(np.clip((score - min_score) / (max_score - min_score), 0.0, 1.0))
    if ratio < 0.5:
        blend = ratio / 0.5
        return (255, int(64 + 176 * blend), 18)
    blend = (ratio - 0.5) / 0.5
    return (int(242 - 180 * blend), 230, int(24 + 64 * blend))


def _world_to_canvas(point_xy: np.ndarray, grid: object, cell_pixels: int) -> tuple[float, float]:
    col_float = float(point_xy[0] / grid.voxel_size - grid.min_ix)
    row_float = float(point_xy[1] / grid.voxel_size - grid.min_iy)
    x = col_float * cell_pixels
    y = (grid.state.shape[0] - row_float) * cell_pixels
    return (x, y)


def _make_base_image(grid: object, cell_pixels: int) -> Image.Image:
    image = Image.fromarray(grid.map_rgb_top_down, mode="RGB")
    width = grid.state.shape[1] * cell_pixels
    height = grid.state.shape[0] * cell_pixels
    image = image.resize((width, height), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    grid_color = (210, 210, 210)
    for col in range(grid.state.shape[1] + 1):
        x = col * cell_pixels
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for row in range(grid.state.shape[0] + 1):
        y = row * cell_pixels
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    return image


def _render_overlay(
    base_image: Image.Image,
    result: DWAResult,
    goal_xy: tuple[float, float] | np.ndarray,
    state: RobotState,
    project_xy: Callable[[np.ndarray], tuple[float, float]],
    scale_px: int,
    draw_invalid: bool = False,
) -> Image.Image:
    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    min_score, max_score = _trajectory_score_bounds(result)
    best_signature = (
        round(result.best_linear_velocity, 6),
        round(result.best_angular_velocity, 6),
    )

    for candidate in result.candidates:
        if not candidate.valid and not draw_invalid:
            continue
        color = _score_to_color(candidate.score, min_score, max_score)
        if candidate.valid:
            signature = (
                round(candidate.linear_velocity, 6),
                round(candidate.angular_velocity, 6),
            )
            if signature == best_signature:
                color = (0, 255, 255)

        polyline = [project_xy(pose[:2]) for pose in candidate.trajectory]
        draw.line(polyline, fill=color, width=max(2, scale_px // 4))

    start_xy = np.asarray([state.x, state.y], dtype=np.float64)
    goal_xy = np.asarray(goal_xy, dtype=np.float64)
    start_canvas = project_xy(start_xy)
    goal_canvas = project_xy(goal_xy)
    marker_radius = max(3, scale_px // 3)

    draw.ellipse(
        [
            (start_canvas[0] - marker_radius, start_canvas[1] - marker_radius),
            (start_canvas[0] + marker_radius, start_canvas[1] + marker_radius),
        ],
        fill=(24, 144, 255),
        outline=(255, 255, 255),
        width=2,
    )
    draw.ellipse(
        [
            (goal_canvas[0] - marker_radius, goal_canvas[1] - marker_radius),
            (goal_canvas[0] + marker_radius, goal_canvas[1] + marker_radius),
        ],
        fill=(255, 32, 192),
        outline=(255, 255, 255),
        width=2,
    )

    arrow_length_m = max(0.25, abs(result.best_linear_velocity) * 1.2)
    arrow_heading = float(state.theta + result.best_angular_velocity * 0.5)
    arrow_tip_xy = start_xy + arrow_length_m * np.asarray(
        [math.cos(arrow_heading), math.sin(arrow_heading)],
        dtype=np.float64,
    )
    tip_canvas = project_xy(arrow_tip_xy)
    draw.line([start_canvas, tip_canvas], fill=(255, 145, 0), width=max(2, scale_px // 4))

    arrow_head_length = max(6, scale_px)
    left = (
        tip_canvas[0] - arrow_head_length * math.cos(arrow_heading - math.pi / 6.0),
        tip_canvas[1] + arrow_head_length * math.sin(arrow_heading - math.pi / 6.0),
    )
    right = (
        tip_canvas[0] - arrow_head_length * math.cos(arrow_heading + math.pi / 6.0),
        tip_canvas[1] + arrow_head_length * math.sin(arrow_heading + math.pi / 6.0),
    )
    draw.polygon([tip_canvas, left, right], fill=(255, 145, 0))

    text_lines = [
        f"v={result.best_linear_velocity:.3f} m/s",
        f"w={result.best_angular_velocity:.3f} rad/s",
        f"score={result.best_score:.3f}" if math.isfinite(result.best_score) else "score=-inf",
        f"valid={result.valid_candidate_count}/{len(result.candidates)}",
    ]
    text_box_height = 8 + 16 * len(text_lines)
    draw.rectangle([(6, 6), (170, text_box_height)], fill=(0, 0, 0))
    for index, line in enumerate(text_lines):
        draw.text((12, 10 + index * 16), line, fill=(255, 255, 255))

    return image


def render_dwa_result_image(
    grid: object,
    result: DWAResult,
    goal_xy: tuple[float, float] | np.ndarray,
    state: RobotState,
    cell_pixels: int = 12,
    draw_invalid: bool = False,
) -> Image.Image:
    if cell_pixels <= 1:
        raise ValueError("cell_pixels must be greater than 1.")

    return _render_overlay(
        base_image=_make_base_image(grid, cell_pixels),
        result=result,
        goal_xy=goal_xy,
        state=state,
        project_xy=lambda point_xy: _world_to_canvas(point_xy, grid, cell_pixels),
        scale_px=cell_pixels,
        draw_invalid=draw_invalid,
    )


def render_dwa_on_base_image(
    base_image_rgb: np.ndarray,
    result: DWAResult,
    goal_xy: tuple[float, float] | np.ndarray,
    state: RobotState,
    project_xy: Callable[[np.ndarray], tuple[float, float]],
    scale_px: int = 12,
    draw_invalid: bool = False,
) -> Image.Image:
    return _render_overlay(
        base_image=Image.fromarray(base_image_rgb, mode="RGB"),
        result=result,
        goal_xy=goal_xy,
        state=state,
        project_xy=project_xy,
        scale_px=scale_px,
        draw_invalid=draw_invalid,
    )


def visualize_dwa_result(
    grid: object,
    result: DWAResult,
    goal_xy: tuple[float, float] | np.ndarray,
    state: RobotState,
    output_path: str | Path | None = None,
    cell_pixels: int = 12,
    draw_invalid: bool = False,
) -> Image.Image:
    image = render_dwa_result_image(
        grid=grid,
        result=result,
        goal_xy=goal_xy,
        state=state,
        cell_pixels=cell_pixels,
        draw_invalid=draw_invalid,
    )
    if output_path is not None:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    return image
