from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from dwaplanner.dwa_planner import DWAPlanner, RobotState, world_to_grid


@dataclass(slots=True)
class FakeGrid:
    voxel_size: float
    min_ix: int
    min_iy: int
    height_index: np.ndarray
    state: np.ndarray
    passable_mask: np.ndarray


def make_flat_grid(rows: int = 12, cols: int = 13, voxel_size: float = 0.25) -> FakeGrid:
    min_ix = -(cols // 2)
    min_iy = 0
    height_index = np.zeros((rows, cols), dtype=np.int32)
    state = np.ones((rows, cols), dtype=np.uint8)
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
            for direction_index, (dx, dy) in enumerate(offsets):
                neighbor_row = row + dy
                neighbor_col = col + dx
                if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                    passable_mask[row, col, direction_index] = True

    return FakeGrid(
        voxel_size=voxel_size,
        min_ix=min_ix,
        min_iy=min_iy,
        height_index=height_index,
        state=state,
        passable_mask=passable_mask,
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
        grid = make_flat_grid()
        start_row, start_col = world_to_grid(0.0, 0.0, grid)
        grid.passable_mask[start_row, start_col, 0] = False
        grid.passable_mask[start_row, start_col, 1] = False
        grid.passable_mask[start_row, start_col, 7] = False

        planner = DWAPlanner()
        result = planner.plan(grid, goal_xy=(0.0, 1.5), state=RobotState())

        self.assertFalse(result.used_emergency_stop)
        self.assertGreater(abs(result.best_angular_velocity), 0.1)
        self.assertTrue(math.isfinite(result.best_score))


if __name__ == "__main__":
    unittest.main()
