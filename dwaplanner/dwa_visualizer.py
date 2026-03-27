"""Open3D helpers for DWA planner visualization."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import open3d as o3d

from .dwa_planner import DWAResult, RobotState, grid_cell_height, is_traversable_cell, world_to_grid
from .voxsense_adapter import (
    build_direction_lines,
    build_grid_mesh,
    build_render_point_cloud,
)


def _trajectory_score_bounds(result: DWAResult) -> tuple[float, float]:
    valid_scores = [candidate.score for candidate in result.candidates if candidate.valid]
    if not valid_scores:
        return 0.0, 1.0
    return min(valid_scores), max(valid_scores)


def _score_to_color(score: float, min_score: float, max_score: float) -> list[float]:
    if not math.isfinite(score):
        return [0.55, 0.55, 0.55]
    if math.isclose(min_score, max_score):
        return [0.20, 0.85, 0.20]
    ratio = float(np.clip((score - min_score) / (max_score - min_score), 0.0, 1.0))
    if ratio < 0.5:
        blend = ratio / 0.5
        return [1.0, 0.25 + 0.65 * blend, 0.10]
    blend = (ratio - 0.5) / 0.5
    return [0.95 - 0.75 * blend, 0.90, 0.10 + 0.15 * blend]


def _pose_to_xyz(pose: np.ndarray, grid: object, z_lift: float) -> np.ndarray:
    row, col = world_to_grid(float(pose[0]), float(pose[1]), grid)
    z = grid_cell_height(row, col, grid) if is_traversable_cell(row, col, grid) else 0.0
    return np.asarray([pose[0], pose[1], z + z_lift], dtype=np.float64)


def build_trajectory_lines(
    grid: object,
    result: DWAResult,
    z_lift: float | None = None,
) -> o3d.geometry.LineSet:
    z_lift = z_lift if z_lift is not None else float(grid.voxel_size) * 0.20
    min_score, max_score = _trajectory_score_bounds(result)
    best_signature = (
        round(result.best_linear_velocity, 6),
        round(result.best_angular_velocity, 6),
    )

    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    for candidate in result.candidates:
        if candidate.trajectory.shape[0] < 2:
            continue

        signature = (
            round(candidate.linear_velocity, 6),
            round(candidate.angular_velocity, 6),
        )
        color = [0.0, 1.0, 1.0] if candidate.valid and signature == best_signature else _score_to_color(
            candidate.score,
            min_score,
            max_score,
        )
        for start_pose, end_pose in zip(candidate.trajectory[:-1], candidate.trajectory[1:]):
            start_point = _pose_to_xyz(start_pose, grid, z_lift)
            end_point = _pose_to_xyz(end_pose, grid, z_lift)
            point_index = len(points)
            points.append(start_point.tolist())
            points.append(end_point.tolist())
            lines.append([point_index, point_index + 1])
            colors.append(color)

    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def _rotation_from_z_axis(target: np.ndarray) -> np.ndarray:
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-9:
        if dot > 0.0:
            return np.eye(3)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1.0, 0.0, 0.0]) * math.pi)

    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - dot) / (cross_norm**2))


def build_velocity_arrow(
    grid: object,
    state: RobotState,
    result: DWAResult,
) -> o3d.geometry.TriangleMesh:
    base_length = max(float(grid.voxel_size) * 1.2, abs(result.best_linear_velocity) * 1.2)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=float(grid.voxel_size) * 0.06,
        cone_radius=float(grid.voxel_size) * 0.10,
        cylinder_height=base_length * 0.65,
        cone_height=base_length * 0.35,
    )
    target_heading = float(state.theta + result.best_angular_velocity * 0.5)
    direction = np.asarray(
        [math.cos(target_heading), math.sin(target_heading), 0.0],
        dtype=np.float64,
    )
    arrow.rotate(_rotation_from_z_axis(direction), center=np.zeros(3, dtype=np.float64))

    row, col = world_to_grid(state.x, state.y, grid)
    z = grid_cell_height(row, col, grid) if is_traversable_cell(row, col, grid) else 0.0
    arrow.translate(np.asarray([state.x, state.y, z + float(grid.voxel_size) * 0.22], dtype=np.float64))
    arrow.paint_uniform_color([1.0, 0.55, 0.10])
    arrow.compute_vertex_normals()
    return arrow


def build_goal_marker(
    goal_xy: tuple[float, float] | np.ndarray,
    grid: object,
) -> o3d.geometry.TriangleMesh:
    goal = np.asarray(goal_xy, dtype=np.float64)
    row, col = world_to_grid(float(goal[0]), float(goal[1]), grid)
    z = grid_cell_height(row, col, grid) if is_traversable_cell(row, col, grid) else 0.0
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(grid.voxel_size) * 0.22)
    sphere.translate(np.asarray([goal[0], goal[1], z + float(grid.voxel_size) * 0.35], dtype=np.float64))
    sphere.paint_uniform_color([0.95, 0.15, 0.85])
    sphere.compute_vertex_normals()
    return sphere


def build_robot_marker(state: RobotState, grid: object) -> o3d.geometry.TriangleMesh:
    row, col = world_to_grid(state.x, state.y, grid)
    z = grid_cell_height(row, col, grid) if is_traversable_cell(row, col, grid) else 0.0
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(grid.voxel_size) * 0.18)
    mesh.translate(np.asarray([state.x, state.y, z + float(grid.voxel_size) * 0.18], dtype=np.float64))
    mesh.paint_uniform_color([0.15, 0.90, 0.95])
    mesh.compute_vertex_normals()
    return mesh


def build_visualization_geometries(
    grid: object,
    result: DWAResult,
    goal_xy: tuple[float, float] | np.ndarray,
    state: RobotState,
    show_blocked_directions: bool = False,
    context_points: np.ndarray | None = None,
    max_render_points: int = 20000,
) -> list[o3d.geometry.Geometry]:
    geometries: list[o3d.geometry.Geometry] = [
        build_grid_mesh(grid),
        build_direction_lines(grid=grid, show_blocked=show_blocked_directions),
        build_trajectory_lines(grid=grid, result=result),
        build_goal_marker(goal_xy=goal_xy, grid=grid),
        build_velocity_arrow(grid=grid, state=state, result=result),
        build_robot_marker(state=state, grid=grid),
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(float(grid.voxel_size) * 4.0, 0.5),
            origin=[0.0, 0.0, 0.0],
        ),
    ]
    if context_points is not None and context_points.size > 0:
        geometries.append(build_render_point_cloud(context_points, max_render_points))
    return geometries


def visualize_dwa_result(
    grid: object,
    result: DWAResult,
    goal_xy: tuple[float, float] | np.ndarray,
    state: RobotState,
    show_blocked_directions: bool = False,
    context_points: np.ndarray | None = None,
    max_render_points: int = 20000,
    window_name: str = "DWA Planner",
) -> list[o3d.geometry.Geometry]:
    geometries = build_visualization_geometries(
        grid=grid,
        result=result,
        goal_xy=goal_xy,
        state=state,
        show_blocked_directions=show_blocked_directions,
        context_points=context_points,
        max_render_points=max_render_points,
    )
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
    return geometries
