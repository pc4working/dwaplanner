from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from dwaplanner.dwa_planner import DWAConfig, DWAPlanner, RobotState, world_to_grid
from dwaplanner.dwa_visualizer import render_dwa_result_image
from dwaplanner.unitree_b2 import (
    B2CommandLimits,
    B2MotionState,
    B2VelocityCommand,
    DEFAULT_B2_MAX_ANGULAR_VELOCITY,
    build_robot_state_from_motion_state,
    clamp_b2_velocity_command,
)


@dataclass(slots=True)
class FakeGrid:
    voxel_size: float
    min_ix: int
    min_iy: int
    height_index: np.ndarray
    state: np.ndarray
    passable_mask: np.ndarray
    map_rgb_top_down: np.ndarray


def _make_grid(rows: int, cols: int, meters_per_pixel: float, obstacle_mask_top_down: np.ndarray | None = None) -> FakeGrid:
    min_ix = -(cols // 2)
    min_iy = 0
    state = np.ones((rows, cols), dtype=np.uint8)
    if obstacle_mask_top_down is not None:
        state = np.flipud(~obstacle_mask_top_down).astype(np.uint8)

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
                if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols and state[neighbor_row, neighbor_col] != 0:
                    passable_mask[row, col, direction_index] = True

    map_rgb_top_down = np.zeros((rows, cols, 3), dtype=np.uint8)
    map_rgb_top_down[:] = np.asarray([244, 244, 244], dtype=np.uint8)
    if obstacle_mask_top_down is not None:
        map_rgb_top_down[obstacle_mask_top_down] = np.asarray([28, 28, 28], dtype=np.uint8)

    return FakeGrid(
        voxel_size=meters_per_pixel,
        min_ix=min_ix,
        min_iy=min_iy,
        height_index=np.zeros((rows, cols), dtype=np.int32),
        state=state,
        passable_mask=passable_mask,
        map_rgb_top_down=map_rgb_top_down,
    )


def make_flat_grid(rows: int = 12, cols: int = 13, meters_per_pixel: float = 0.25):
    return _make_grid(rows=rows, cols=cols, meters_per_pixel=meters_per_pixel)


def make_blocked_forward_grid(rows: int = 18, cols: int = 17, meters_per_pixel: float = 0.2):
    obstacle_mask_top_down = np.zeros((rows, cols), dtype=bool)
    center = cols // 2
    obstacle_mask_top_down[1:4, center - 1 : center + 2] = True
    return _make_grid(
        rows=rows,
        cols=cols,
        meters_per_pixel=meters_per_pixel,
        obstacle_mask_top_down=obstacle_mask_top_down,
    )


def make_safe_start_state(grid: FakeGrid) -> RobotState:
    return RobotState(y=grid.voxel_size * 1.5)


class DWAPlannerTest(unittest.TestCase):
    def test_default_b2_angular_limit_is_used_by_dwa(self) -> None:
        self.assertAlmostEqual(DWAConfig().max_angular_velocity, DEFAULT_B2_MAX_ANGULAR_VELOCITY)

    def test_world_to_grid_matches_expected_origin_cell(self) -> None:
        grid = make_flat_grid()
        row, col = world_to_grid(0.0, 0.0, grid)
        self.assertEqual((row, col), (0, 6))

    def test_forward_goal_prefers_forward_motion(self) -> None:
        grid = make_flat_grid()
        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.0, 1.5), state=make_safe_start_state(grid))

        self.assertFalse(result.used_emergency_stop)
        self.assertGreater(result.valid_candidate_count, 0)
        self.assertGreaterEqual(result.best_linear_velocity, 0.5)
        self.assertLess(abs(result.best_angular_velocity), 0.11)

    def test_goal_inside_tolerance_returns_stop(self) -> None:
        grid = make_flat_grid()
        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.0, 0.45), state=make_safe_start_state(grid))

        self.assertEqual(result.best_linear_velocity, 0.0)
        self.assertEqual(result.best_angular_velocity, 0.0)
        self.assertFalse(result.used_emergency_stop)

    def test_blocked_forward_direction_causes_turn(self) -> None:
        grid = make_blocked_forward_grid()

        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.2, 1.5), state=make_safe_start_state(grid))

        self.assertFalse(result.used_emergency_stop)
        self.assertGreater(abs(result.best_angular_velocity), 0.09)
        self.assertGreaterEqual(result.best_linear_velocity, 0.5)
        self.assertTrue(math.isfinite(result.best_score))

    def test_occupied_but_not_passable_start_cell_is_rejected(self) -> None:
        grid = make_flat_grid()
        start_row, start_col = world_to_grid(0.0, 0.0, grid)
        grid.passable_mask[start_row, start_col, :] = False

        planner = DWAPlanner(DWAConfig(robot_radius=0.0))
        with self.assertRaises(RuntimeError):
            planner.plan(grid, goal_xy=(0.0, 1.5), state=RobotState())

    def test_robot_radius_rejects_too_narrow_corridor(self) -> None:
        rows = 12
        cols = 11
        obstacle_mask_top_down = np.ones((rows, cols), dtype=bool)
        obstacle_mask_top_down[:, cols // 2] = False
        grid = _make_grid(
            rows=rows,
            cols=cols,
            meters_per_pixel=0.2,
            obstacle_mask_top_down=obstacle_mask_top_down,
        )

        planner = DWAPlanner(DWAConfig(robot_radius=0.25))
        with self.assertRaises(RuntimeError):
            planner.plan(grid, goal_xy=(0.0, 1.5), state=RobotState())

    def test_render_output_has_expected_size(self) -> None:
        grid = make_flat_grid()
        planner = DWAPlanner()
        state = make_safe_start_state(grid)
        result = planner.plan(grid, goal_xy=(0.0, 1.5), state=state)
        image = render_dwa_result_image(grid, result, goal_xy=(0.0, 1.5), state=state, cell_pixels=10)

        self.assertEqual(image.size, (grid.state.shape[1] * 10, grid.state.shape[0] * 10))

    def test_b2_command_clamp_limits_forward_and_yaw_speed(self) -> None:
        command = B2VelocityCommand(linear_x=1.8, angular_z=0.9)
        clipped = clamp_b2_velocity_command(
            command,
            limits=B2CommandLimits(max_forward_velocity=1.0, max_angular_velocity=0.35),
        )

        self.assertAlmostEqual(clipped.linear_x, 1.0)
        self.assertAlmostEqual(clipped.angular_z, 0.35)

    def test_build_robot_state_from_motion_state_keeps_fallback_pose_by_default(self) -> None:
        fallback_state = RobotState(x=1.0, y=2.0, theta=0.5, linear_velocity=0.1, angular_velocity=0.2)
        motion_state = B2MotionState(
            timestamp_seconds=12.0,
            mode=0,
            gait_type=0,
            progress=0.0,
            position_x=9.0,
            position_y=8.0,
            position_z=0.0,
            yaw=1.2,
            linear_velocity_x=0.3,
            linear_velocity_y=0.4,
            linear_velocity_z=0.0,
            yaw_speed=0.25,
            body_height=0.0,
        )

        state = build_robot_state_from_motion_state(motion_state, fallback_state, use_live_pose=False)

        self.assertEqual(state.x, fallback_state.x)
        self.assertEqual(state.y, fallback_state.y)
        self.assertEqual(state.theta, fallback_state.theta)
        self.assertAlmostEqual(state.linear_velocity, 0.5)
        self.assertAlmostEqual(state.angular_velocity, 0.25)


if __name__ == "__main__":
    unittest.main()
