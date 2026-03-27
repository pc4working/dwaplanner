#!/usr/bin/env python3
"""Demo script for running DWA on a voxsense traversability grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from dwaplanner.dwa_planner import DWAConfig, DWAPlanner, RobotState
from dwaplanner.dwa_visualizer import visualize_dwa_result
from dwaplanner.voxsense_adapter import build_traversability_run, print_run_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DWA local planning on a voxsense traversability grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to a single-frame point cloud. Defaults to the first file under the sibling voxsense/pcd directory.",
    )
    parser.add_argument("--goal-x", type=float, default=0.0, help="Goal x position in meters.")
    parser.add_argument("--goal-y", type=float, default=2.0, help="Goal y position in meters.")
    parser.add_argument("--start-x", type=float, default=0.0, help="Robot start x position.")
    parser.add_argument("--start-y", type=float, default=0.0, help="Robot start y position.")
    parser.add_argument(
        "--start-theta-deg",
        type=float,
        default=90.0,
        help="Robot heading in degrees. 90 degrees points along +y.",
    )
    parser.add_argument("--current-v", type=float, default=0.0, help="Current linear velocity in m/s.")
    parser.add_argument("--current-w", type=float, default=0.0, help="Current angular velocity in rad/s.")
    parser.add_argument("--voxel-size", type=float, default=0.15, help="Voxel size in meters.")
    parser.add_argument(
        "--min-points-per-voxel",
        type=int,
        default=5,
        help="Discard sparse 3D voxels below this point count.",
    )
    parser.add_argument(
        "--origin-radius",
        type=float,
        default=0.3,
        help="Remove points inside this 3D radius before building observed support columns.",
    )
    parser.add_argument(
        "--fill-radius",
        type=float,
        default=0.45,
        help="Cells inside this xy radius are force-filled as the default traversable zone.",
    )
    parser.add_argument(
        "--blocked-height-diff-voxels",
        type=int,
        default=3,
        help="A direction is blocked when the neighbor height delta exceeds this threshold in z voxels.",
    )
    parser.add_argument("--show-points", action="store_true", help="Render the filtered point cloud.")
    parser.add_argument(
        "--show-blocked-directions",
        action="store_true",
        help="Show red blocked spokes in the grid visualization.",
    )
    parser.add_argument(
        "--max-render-points",
        type=int,
        default=20000,
        help="Maximum number of point cloud points rendered when --show-points is enabled.",
    )
    parser.add_argument("--no-vis", action="store_true", help="Run planning without opening the Open3D window.")
    parser.add_argument("--max-linear-velocity", type=float, default=1.0, help="Max linear velocity.")
    parser.add_argument("--max-angular-velocity", type=float, default=1.0, help="Max angular velocity.")
    parser.add_argument("--max-linear-acceleration", type=float, default=0.5, help="Max linear acceleration.")
    parser.add_argument("--max-angular-acceleration", type=float, default=1.0, help="Max angular acceleration.")
    parser.add_argument("--control-interval", type=float, default=0.5, help="Velocity command interval.")
    parser.add_argument("--prediction-horizon", type=float, default=2.0, help="Trajectory horizon in seconds.")
    parser.add_argument("--simulation-dt", type=float, default=0.1, help="Trajectory simulation timestep.")
    parser.add_argument("--linear-samples", type=int, default=7, help="Number of sampled linear velocities.")
    parser.add_argument("--angular-samples", type=int, default=11, help="Number of sampled angular velocities.")
    parser.add_argument("--heading-weight", type=float, default=1.0, help="Weight for heading alignment.")
    parser.add_argument("--clearance-weight", type=float, default=0.5, help="Weight for clearance.")
    parser.add_argument("--velocity-weight", type=float, default=0.2, help="Weight for speed preference.")
    parser.add_argument(
        "--clearance-search-radius-cells",
        type=int,
        default=4,
        help="Search radius for nearest empty cell clearance.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DWAConfig:
    return DWAConfig(
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        max_linear_acceleration=args.max_linear_acceleration,
        max_angular_acceleration=args.max_angular_acceleration,
        control_interval=args.control_interval,
        prediction_horizon=args.prediction_horizon,
        simulation_dt=args.simulation_dt,
        linear_velocity_samples=args.linear_samples,
        angular_velocity_samples=args.angular_samples,
        heading_weight=args.heading_weight,
        clearance_weight=args.clearance_weight,
        velocity_weight=args.velocity_weight,
        clearance_search_radius_cells=args.clearance_search_radius_cells,
    )


def print_plan_summary(result: object) -> None:
    print(f"sampled_trajectories: {len(result.candidates)}")
    print(f"valid_trajectories: {result.valid_candidate_count}")
    print(
        "dynamic_window:"
        f" v=[{result.dynamic_window[0]:.3f}, {result.dynamic_window[1]:.3f}]"
        f" w=[{result.dynamic_window[2]:.3f}, {result.dynamic_window[3]:.3f}]"
    )
    print(f"selected_linear_velocity_mps: {result.best_linear_velocity:.3f}")
    print(f"selected_angular_velocity_rps: {result.best_angular_velocity:.3f}")
    print(f"best_score: {result.best_score:.3f}")
    print(f"used_emergency_stop: {result.used_emergency_stop}")


def main() -> None:
    args = parse_args()
    run = build_traversability_run(
        input_path=args.input_path,
        voxel_size=args.voxel_size,
        min_points_per_voxel=args.min_points_per_voxel,
        origin_radius=args.origin_radius,
        fill_radius=args.fill_radius,
        blocked_height_diff_voxels=args.blocked_height_diff_voxels,
    )
    print_run_stats(run)

    state = RobotState(
        x=args.start_x,
        y=args.start_y,
        theta=math.radians(args.start_theta_deg),
        linear_velocity=args.current_v,
        angular_velocity=args.current_w,
    )
    goal_xy = (args.goal_x, args.goal_y)
    planner = DWAPlanner(build_config(args))
    result = planner.plan(run.grid, goal_xy=goal_xy, state=state)
    print_plan_summary(result)

    if args.no_vis:
        return

    context_points = run.forward_points if args.show_points else None
    visualize_dwa_result(
        grid=run.grid,
        result=result,
        goal_xy=goal_xy,
        state=state,
        show_blocked_directions=args.show_blocked_directions,
        context_points=context_points,
        max_render_points=args.max_render_points,
    )


if __name__ == "__main__":
    main()
