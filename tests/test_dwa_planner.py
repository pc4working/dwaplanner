from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from dwaplanner.dwa_planner import DWAPlanner, RobotState, world_to_grid
from dwaplanner.dwa_visualizer import render_dwa_result_image


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
    obstacle_mask_top_down[rows - 4 : rows - 1, center - 1 : center + 2] = True
    return _make_grid(
        rows=rows,
        cols=cols,
        meters_per_pixel=meters_per_pixel,
        obstacle_mask_top_down=obstacle_mask_top_down,
    )


class DWAPlannerTest(unittest.TestCase):
    def test_world_to_grid_matches_expected_origin_cell(self) -> None:
        grid = make_flat_grid()
        row, col = world_to_grid(0.0, 0.0, grid)
        self.assertEqual((row, col), (0, 6))

    def test_forward_goal_prefers_forward_motion(self) -> None:
        grid = make_flat_grid()
        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.0, 1.5), state=RobotState())

        self.assertFalse(result.used_emergency_stop)
        self.assertGreater(result.valid_candidate_count, 0)
        self.assertGreater(result.best_linear_velocity, 0.05)
        self.assertLess(abs(result.best_angular_velocity), 0.11)

    def test_goal_inside_tolerance_returns_stop(self) -> None:
        grid = make_flat_grid()
        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.0, 0.1), state=RobotState())

        self.assertEqual(result.best_linear_velocity, 0.0)
        self.assertEqual(result.best_angular_velocity, 0.0)
        self.assertFalse(result.used_emergency_stop)

    def test_blocked_forward_direction_causes_turn(self) -> None:
        grid = make_blocked_forward_grid()

        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.2, 1.5), state=RobotState())

        self.assertFalse(result.used_emergency_stop)
        self.assertGreater(abs(result.best_angular_velocity), 0.09)
        self.assertLess(result.best_linear_velocity, 0.2)
        self.assertTrue(math.isfinite(result.best_score))

    def test_render_output_has_expected_size(self) -> None:
        grid = make_flat_grid()
        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.0, 1.5), state=RobotState())
        image = render_dwa_result_image(grid, result, goal_xy=(0.0, 1.5), state=RobotState(), cell_pixels=10)

        self.assertEqual(image.size, (grid.state.shape[1] * 10, grid.state.shape[0] * 10))


if __name__ == "__main__":
    unittest.main()
