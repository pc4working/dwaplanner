from __future__ import annotations

import math
import unittest

import numpy as np

from dwaplanner.dwa_planner import DWAPlanner, RobotState, world_to_grid
from dwaplanner.dwa_visualizer import render_dwa_result_image
from dwaplanner.image_grid import build_grid_from_image_array


def make_flat_grid(rows: int = 12, cols: int = 13, meters_per_pixel: float = 0.25):
    image = np.full((rows, cols), 255, dtype=np.uint8)
    return build_grid_from_image_array(image, meters_per_pixel=meters_per_pixel)


def make_blocked_forward_grid(rows: int = 18, cols: int = 17, meters_per_pixel: float = 0.2):
    image = np.full((rows, cols), 255, dtype=np.uint8)
    center = cols // 2
    image[rows - 4 : rows - 1, center - 1 : center + 2] = 0
    return build_grid_from_image_array(image, meters_per_pixel=meters_per_pixel)


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
