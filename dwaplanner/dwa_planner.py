"""Core Dynamic Window Approach implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.ndimage import distance_transform_edt

EMPTY_STATE_VALUE = 0
_DIRECTION_TO_INDEX = {
    (0, 1): 0,
    (1, 1): 1,
    (1, 0): 2,
    (1, -1): 3,
    (0, -1): 4,
    (-1, -1): 5,
    (-1, 0): 6,
    (-1, 1): 7,
}


class TraversabilityGridLike(Protocol):
    voxel_size: float
    min_ix: int
    min_iy: int
    state: np.ndarray
    passable_mask: np.ndarray
    height_index: np.ndarray


@dataclass(slots=True)
class PlanningContext:
    passable_cell_mask: np.ndarray
    safe_passable_cell_mask: np.ndarray
    clearance_map: np.ndarray


@dataclass(slots=True)
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = math.pi / 2.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


@dataclass(slots=True)
class DWAConfig:
    min_linear_velocity: float = 0.5
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 0.35
    max_linear_acceleration: float = 0.5
    max_angular_acceleration: float = 1.0
    control_interval: float = 0.5
    prediction_horizon: float = 2.0
    simulation_dt: float = 0.1
    linear_velocity_samples: int = 7
    angular_velocity_samples: int = 11
    heading_weight: float = 1.0
    goal_progress_weight: float = 0.8
    clearance_weight: float = 0.5
    velocity_weight: float = 0.2
    clearance_search_radius_cells: int = 4
    goal_tolerance: float = 0.25
    robot_radius: float = 0.25

    def __post_init__(self) -> None:
        if self.min_linear_velocity < 0.0:
            raise ValueError("min_linear_velocity cannot be negative.")
        if self.max_linear_velocity <= self.min_linear_velocity:
            raise ValueError("max_linear_velocity must be greater than min_linear_velocity.")
        if self.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be positive.")
        if self.max_linear_acceleration <= 0.0:
            raise ValueError("max_linear_acceleration must be positive.")
        if self.max_angular_acceleration <= 0.0:
            raise ValueError("max_angular_acceleration must be positive.")
        if self.control_interval <= 0.0:
            raise ValueError("control_interval must be positive.")
        if self.prediction_horizon <= 0.0:
            raise ValueError("prediction_horizon must be positive.")
        if self.simulation_dt <= 0.0:
            raise ValueError("simulation_dt must be positive.")
        if self.linear_velocity_samples <= 0:
            raise ValueError("linear_velocity_samples must be positive.")
        if self.angular_velocity_samples <= 0:
            raise ValueError("angular_velocity_samples must be positive.")
        if self.clearance_search_radius_cells <= 0:
            raise ValueError("clearance_search_radius_cells must be positive.")
        if self.goal_tolerance < 0.0:
            raise ValueError("goal_tolerance cannot be negative.")
        if self.robot_radius < 0.0:
            raise ValueError("robot_radius cannot be negative.")


@dataclass(slots=True)
class TrajectoryCandidate:
    linear_velocity: float
    angular_velocity: float
    trajectory: np.ndarray
    score: float
    heading_score: float
    goal_progress_score: float
    clearance_score: float
    velocity_score: float
    min_clearance: float
    final_goal_distance: float
    valid: bool


@dataclass(slots=True)
class DWAResult:
    best_linear_velocity: float
    best_angular_velocity: float
    best_trajectory: np.ndarray
    best_score: float
    dynamic_window: tuple[float, float, float, float]
    candidates: list[TrajectoryCandidate] = field(default_factory=list)
    used_emergency_stop: bool = False

    @property
    def valid_candidate_count(self) -> int:
        return sum(1 for candidate in self.candidates if candidate.valid)


def wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def world_to_grid(x: float, y: float, grid: TraversabilityGridLike) -> tuple[int, int]:
    ix = int(math.floor(x / grid.voxel_size))
    iy = int(math.floor(y / grid.voxel_size))
    row = iy - int(grid.min_iy)
    col = ix - int(grid.min_ix)
    return row, col


def is_traversable_cell(row: int, col: int, grid: TraversabilityGridLike) -> bool:
    rows, cols = grid.state.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return False
    return bool(build_passable_cell_mask(grid)[row, col])


def build_passable_cell_mask(grid: TraversabilityGridLike) -> np.ndarray:
    occupied_mask = np.asarray(grid.state != EMPTY_STATE_VALUE, dtype=bool)
    directional_mask = np.asarray(grid.passable_mask.any(axis=2), dtype=bool)
    return occupied_mask & directional_mask


def grid_cell_height(row: int, col: int, grid: TraversabilityGridLike) -> float:
    if not is_traversable_cell(row, col, grid):
        return 0.0
    return (float(grid.height_index[row, col]) + 0.5) * float(grid.voxel_size)


class DWAPlanner:
    def __init__(self, config: DWAConfig | None = None) -> None:
        self.config = config or DWAConfig()

    def plan(
        self,
        grid: TraversabilityGridLike,
        goal_xy: tuple[float, float] | np.ndarray,
        state: RobotState | None = None,
    ) -> DWAResult:
        state = state or RobotState()
        goal = np.asarray(goal_xy, dtype=np.float64)
        if goal.shape != (2,):
            raise ValueError("goal_xy must be a 2D position.")

        planning_context = self._build_planning_context(grid)
        self._ensure_pose_on_grid(state.x, state.y, grid, planning_context)
        current_position = np.asarray([state.x, state.y], dtype=np.float64)
        initial_goal_distance = float(np.linalg.norm(goal - current_position))
        if float(np.linalg.norm(goal - current_position)) <= self.config.goal_tolerance:
            stop_trajectory = self._simulate_trajectory(state, 0.0, 0.0)
            return DWAResult(
                best_linear_velocity=0.0,
                best_angular_velocity=0.0,
                best_trajectory=stop_trajectory,
                best_score=1.0,
                dynamic_window=self._compute_dynamic_window(state),
                candidates=[
                    TrajectoryCandidate(
                        linear_velocity=0.0,
                        angular_velocity=0.0,
                        trajectory=stop_trajectory,
                        score=1.0,
                        heading_score=1.0,
                        goal_progress_score=0.0,
                        clearance_score=1.0,
                        velocity_score=0.0,
                        min_clearance=self._compute_cell_clearance(
                            *world_to_grid(state.x, state.y, grid),
                            planning_context,
                        ),
                        final_goal_distance=float(np.linalg.norm(goal - current_position)),
                        valid=True,
                    )
                ],
            )

        dynamic_window = self._compute_dynamic_window(state)
        candidates: list[TrajectoryCandidate] = []
        best_candidate: TrajectoryCandidate | None = None

        for linear_velocity, angular_velocity in self._sample_velocities(dynamic_window):
            trajectory = self._simulate_trajectory(state, linear_velocity, angular_velocity)
            valid, min_clearance = self._check_collision(trajectory, grid, planning_context)
            final_goal_distance = float(np.linalg.norm(goal - trajectory[-1, :2]))
            if valid:
                heading_score = self._compute_heading_score(trajectory, goal)
                goal_progress_score = self._compute_goal_progress_score(
                    initial_goal_distance=initial_goal_distance,
                    final_goal_distance=final_goal_distance,
                    voxel_size=grid.voxel_size,
                )
                clearance_score = self._normalize_clearance(min_clearance, grid.voxel_size)
                velocity_score = self._compute_velocity_score(linear_velocity)
                score = (
                    self.config.heading_weight * heading_score
                    + self.config.goal_progress_weight * goal_progress_score
                    + self.config.clearance_weight * clearance_score
                    + self.config.velocity_weight * velocity_score
                )
            else:
                heading_score = 0.0
                goal_progress_score = 0.0
                clearance_score = 0.0
                velocity_score = 0.0
                score = float("-inf")

            candidate = TrajectoryCandidate(
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                trajectory=trajectory,
                score=score,
                heading_score=heading_score,
                goal_progress_score=goal_progress_score,
                clearance_score=clearance_score,
                velocity_score=velocity_score,
                min_clearance=min_clearance,
                final_goal_distance=final_goal_distance,
                valid=valid,
            )
            candidates.append(candidate)
            if valid and (best_candidate is None or candidate.score > best_candidate.score):
                best_candidate = candidate

        if best_candidate is None:
            stop_trajectory = self._simulate_trajectory(state, 0.0, 0.0)
            return DWAResult(
                best_linear_velocity=0.0,
                best_angular_velocity=0.0,
                best_trajectory=stop_trajectory,
                best_score=float("-inf"),
                dynamic_window=dynamic_window,
                candidates=candidates,
                used_emergency_stop=True,
            )

        return DWAResult(
            best_linear_velocity=best_candidate.linear_velocity,
            best_angular_velocity=best_candidate.angular_velocity,
            best_trajectory=best_candidate.trajectory,
            best_score=best_candidate.score,
            dynamic_window=dynamic_window,
            candidates=candidates,
        )

    def _compute_dynamic_window(self, state: RobotState) -> tuple[float, float, float, float]:
        dv = self.config.max_linear_acceleration * self.config.control_interval
        dw = self.config.max_angular_acceleration * self.config.control_interval
        min_v = max(0.0, state.linear_velocity - dv)
        max_v = min(self.config.max_linear_velocity, state.linear_velocity + dv)
        if max_v > 0.0:
            min_v = max(self.config.min_linear_velocity, min_v)
            max_v = max(min_v, max_v)
        min_w = max(-self.config.max_angular_velocity, state.angular_velocity - dw)
        max_w = min(self.config.max_angular_velocity, state.angular_velocity + dw)
        if min_v > max_v:
            min_v = max_v
        if min_w > max_w:
            min_w = max_w
        return (min_v, max_v, min_w, max_w)

    def _sample_velocities(
        self,
        dynamic_window: tuple[float, float, float, float],
    ) -> list[tuple[float, float]]:
        min_v, max_v, min_w, max_w = dynamic_window
        linear_values = np.linspace(
            min_v,
            max_v,
            self.config.linear_velocity_samples,
            dtype=np.float64,
        )
        angular_values = np.linspace(
            min_w,
            max_w,
            self.config.angular_velocity_samples,
            dtype=np.float64,
        )
        return [
            (float(linear_velocity), float(angular_velocity))
            for linear_velocity in linear_values
            for angular_velocity in angular_values
        ]

    def _simulate_trajectory(
        self,
        state: RobotState,
        linear_velocity: float,
        angular_velocity: float,
    ) -> np.ndarray:
        step_count = max(1, int(math.ceil(self.config.prediction_horizon / self.config.simulation_dt)))
        trajectory = np.zeros((step_count + 1, 3), dtype=np.float64)
        trajectory[0] = np.asarray([state.x, state.y, state.theta], dtype=np.float64)

        x = float(state.x)
        y = float(state.y)
        theta = float(state.theta)
        for index in range(1, step_count + 1):
            theta_mid = theta + angular_velocity * self.config.simulation_dt * 0.5
            x += linear_velocity * math.cos(theta_mid) * self.config.simulation_dt
            y += linear_velocity * math.sin(theta_mid) * self.config.simulation_dt
            theta = wrap_angle(theta + angular_velocity * self.config.simulation_dt)
            trajectory[index] = np.asarray([x, y, theta], dtype=np.float64)
        return trajectory

    def _check_collision(
        self,
        trajectory: np.ndarray,
        grid: TraversabilityGridLike,
        planning_context: PlanningContext,
    ) -> tuple[bool, float]:
        prev_point = trajectory[0, :2]
        prev_row, prev_col = world_to_grid(float(prev_point[0]), float(prev_point[1]), grid)
        if not self._is_safe_passable_cell(prev_row, prev_col, planning_context):
            return False, 0.0

        min_clearance = self._compute_cell_clearance(prev_row, prev_col, planning_context)
        for point in trajectory[1:, :2]:
            valid, segment_min_clearance, prev_row, prev_col = self._check_segment(
                prev_point,
                point,
                prev_row,
                prev_col,
                grid,
                planning_context,
            )
            min_clearance = min(min_clearance, segment_min_clearance)
            if not valid:
                return False, min_clearance
            prev_point = point

        return True, min_clearance

    def _check_segment(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        start_row: int,
        start_col: int,
        grid: TraversabilityGridLike,
        planning_context: PlanningContext,
    ) -> tuple[bool, float, int, int]:
        segment = end_xy - start_xy
        distance = float(np.linalg.norm(segment))
        if distance == 0.0:
            clearance = self._compute_cell_clearance(start_row, start_col, planning_context)
            return True, clearance, start_row, start_col

        steps = max(1, int(math.ceil(distance / max(grid.voxel_size * 0.5, 1e-6))))
        prev_row = start_row
        prev_col = start_col
        min_clearance = self._compute_cell_clearance(start_row, start_col, planning_context)
        for index in range(1, steps + 1):
            alpha = index / steps
            sample_xy = start_xy + alpha * segment
            row, col = world_to_grid(float(sample_xy[0]), float(sample_xy[1]), grid)
            if not self._is_safe_passable_cell(row, col, planning_context):
                return False, 0.0, row, col

            if row != prev_row or col != prev_col:
                delta_col = int(np.sign(col - prev_col))
                delta_row = int(np.sign(row - prev_row))
                direction_index = _DIRECTION_TO_INDEX.get((delta_col, delta_row))
                if direction_index is None:
                    return False, 0.0, row, col
                if not bool(grid.passable_mask[prev_row, prev_col, direction_index]):
                    return False, 0.0, row, col

            min_clearance = min(min_clearance, self._compute_cell_clearance(row, col, planning_context))
            prev_row = row
            prev_col = col

        return True, min_clearance, prev_row, prev_col

    def _compute_heading_score(self, trajectory: np.ndarray, goal_xy: np.ndarray) -> float:
        final_pose = trajectory[-1]
        goal_vector = goal_xy - final_pose[:2]
        goal_distance = float(np.linalg.norm(goal_vector))
        if goal_distance <= self.config.goal_tolerance:
            return 1.0

        heading_vector = np.asarray(
            [math.cos(float(final_pose[2])), math.sin(float(final_pose[2]))],
            dtype=np.float64,
        )
        goal_direction = goal_vector / goal_distance
        alignment = float(np.clip(np.dot(heading_vector, goal_direction), -1.0, 1.0))
        return 0.5 * (alignment + 1.0)

    def _normalize_clearance(self, clearance: float, voxel_size: float) -> float:
        max_clearance = self.config.clearance_search_radius_cells * voxel_size
        if max_clearance <= 0.0:
            return 0.0
        return float(np.clip(clearance / max_clearance, 0.0, 1.0))

    def _compute_goal_progress_score(
        self,
        initial_goal_distance: float,
        final_goal_distance: float,
        voxel_size: float,
    ) -> float:
        normalizer = max(initial_goal_distance, voxel_size)
        if normalizer <= 0.0:
            return 0.0
        progress = max(0.0, initial_goal_distance - final_goal_distance)
        return float(np.clip(progress / normalizer, 0.0, 1.0))

    def _compute_velocity_score(self, linear_velocity: float) -> float:
        if self.config.max_linear_velocity <= 0.0:
            return 0.0
        return float(np.clip(linear_velocity / self.config.max_linear_velocity, 0.0, 1.0))

    def _compute_cell_clearance(
        self,
        row: int,
        col: int,
        planning_context: PlanningContext,
    ) -> float:
        rows, cols = planning_context.clearance_map.shape
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return 0.0
        return float(planning_context.clearance_map[row, col])

    def _build_planning_context(self, grid: TraversabilityGridLike) -> PlanningContext:
        passable_cell_mask = build_passable_cell_mask(grid)
        padded_mask = np.pad(passable_cell_mask, 1, mode="constant", constant_values=False)
        clearance_map = distance_transform_edt(
            padded_mask,
            sampling=float(grid.voxel_size),
        )[1:-1, 1:-1]
        # Convert center-to-center EDT into a conservative boundary clearance estimate.
        clearance_map = np.maximum(clearance_map - 0.5 * float(grid.voxel_size), 0.0)
        clearance_map[~passable_cell_mask] = 0.0
        safe_passable_cell_mask = passable_cell_mask & (clearance_map >= self.config.robot_radius)
        return PlanningContext(
            passable_cell_mask=passable_cell_mask,
            safe_passable_cell_mask=safe_passable_cell_mask,
            clearance_map=clearance_map,
        )

    def _is_safe_passable_cell(
        self,
        row: int,
        col: int,
        planning_context: PlanningContext,
    ) -> bool:
        rows, cols = planning_context.safe_passable_cell_mask.shape
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return False
        return bool(planning_context.safe_passable_cell_mask[row, col])

    def _ensure_pose_on_grid(
        self,
        x: float,
        y: float,
        grid: TraversabilityGridLike,
        planning_context: PlanningContext,
    ) -> None:
        row, col = world_to_grid(x, y, grid)
        if not self._is_safe_passable_cell(row, col, planning_context):
            raise RuntimeError(
                f"Robot pose ({x:.3f}, {y:.3f}) is outside the safe PASSABLE region."
            )
