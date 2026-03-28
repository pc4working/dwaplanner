#!/usr/bin/env python3
"""Demo script for running DWA on a voxsense traversability BEV image."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from dwaplanner.dwa_planner import DWAConfig, DWAPlanner, RobotState
from dwaplanner.dwa_visualizer import render_dwa_on_base_image
from dwaplanner.unitree_b2 import (
    B2CommandLimits,
    B2SportController,
    DEFAULT_B2_MAX_ANGULAR_VELOCITY,
    build_robot_state_from_motion_state,
    command_from_dwa_result,
)
from dwaplanner.voxsense_adapter import (
    bev_world_to_pixel,
    build_bev_image,
    build_traversability_run,
    print_run_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DWA local planning on a voxsense BEV image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to a point cloud file. Defaults to the first file under the sibling voxsense/pcd directory.",
    )
    parser.add_argument("--goal-x", type=float, default=0.0, help="Goal x position in meters.")
    parser.add_argument("--goal-y", type=float, default=2.0, help="Goal y position in meters.")
    parser.add_argument("--start-x", type=float, default=0.0, help="Robot start x position.")
    parser.add_argument(
        "--start-y",
        type=float,
        default=None,
        help="Robot start y position. Defaults to robot_radius + half a cell so the body starts inside known free space.",
    )
    parser.add_argument(
        "--start-theta-deg",
        type=float,
        default=90.0,
        help="Robot heading in degrees. 90 degrees points along +y.",
    )
    parser.add_argument("--current-v", type=float, default=0.5, help="Current linear velocity in m/s.")
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
    parser.add_argument("--show-points", action="store_true", help="Render the filtered point cloud on the BEV base image.")
    parser.add_argument(
        "--show-blocked-directions",
        action="store_true",
        help="Render red spokes for blocked directions in addition to passable spokes.",
    )
    parser.add_argument("--max-render-points", type=int, default=20000, help="Maximum number of rendered context points.")
    parser.add_argument("--x-limit", type=float, default=4.0, help="Render horizontal range [-x-limit, x-limit] in meters.")
    parser.add_argument("--forward-limit", type=float, default=8.0, help="Render forward range [0, forward-limit] in meters.")
    parser.add_argument("--pixels-per-meter", type=float, default=120.0, help="Resolution of the output image.")
    parser.add_argument("--cell-fill-ratio", type=float, default=0.88, help="Fraction of each voxel cell used for the filled tile.")
    parser.add_argument("--point-radius-px", type=int, default=2, help="Radius for rendered context points.")
    parser.add_argument("--line-width-px", type=int, default=2, help="Line width for BEV direction spokes.")
    parser.add_argument("--margin-px", type=int, default=32, help="Outer image margin in pixels.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output image path. Defaults to outputs/<input_stem>_dwa_bev.png.",
    )
    parser.add_argument("--overlay-scale-px", type=int, default=10, help="Reference pixel size for DWA overlay thickness.")
    parser.add_argument("--draw-invalid", action="store_true", help="Also draw invalid candidate trajectories.")
    parser.add_argument("--min-linear-velocity", type=float, default=0.5, help="Minimum commanded cruise speed.")
    parser.add_argument("--max-linear-velocity", type=float, default=1.0, help="Max linear velocity.")
    parser.add_argument(
        "--max-angular-velocity",
        type=float,
        default=DEFAULT_B2_MAX_ANGULAR_VELOCITY,
        help="Max angular velocity. The B2 control path is limited to 0.35 rad/s by default.",
    )
    parser.add_argument("--max-linear-acceleration", type=float, default=0.5, help="Max linear acceleration.")
    parser.add_argument("--max-angular-acceleration", type=float, default=1.0, help="Max angular acceleration.")
    parser.add_argument("--control-interval", type=float, default=0.5, help="Velocity command interval.")
    parser.add_argument("--prediction-horizon", type=float, default=2.0, help="Trajectory horizon in seconds.")
    parser.add_argument("--simulation-dt", type=float, default=0.1, help="Trajectory simulation timestep.")
    parser.add_argument("--linear-samples", type=int, default=7, help="Number of sampled linear velocities.")
    parser.add_argument("--angular-samples", type=int, default=11, help="Number of sampled angular velocities.")
    parser.add_argument("--heading-weight", type=float, default=1.0, help="Weight for heading alignment.")
    parser.add_argument("--goal-progress-weight", type=float, default=0.8, help="Weight for distance reduction toward the goal.")
    parser.add_argument("--clearance-weight", type=float, default=0.5, help="Weight for clearance.")
    parser.add_argument("--velocity-weight", type=float, default=0.2, help="Weight for speed preference.")
    parser.add_argument("--robot-radius", type=float, default=0.25, help="Robot body radius in meters.")
    parser.add_argument(
        "--clearance-search-radius-cells",
        type=int,
        default=4,
        help="Clearance score normalization cap in cell units after distance-transform lookup.",
    )
    parser.add_argument(
        "--unitree-network-interface",
        type=str,
        default=None,
        help="Network interface used to reach the Unitree B2 SDK2 transport.",
    )
    parser.add_argument(
        "--unitree-sdk-root",
        type=Path,
        default=None,
        help="Path to the local unitree_sdk2_python checkout. Defaults to UNITREE_SDK2_PYTHON or ~/unitree_sdk2_python.",
    )
    parser.add_argument(
        "--use-unitree-state",
        action="store_true",
        help="Read the current B2 planar speed and yaw rate from sport mode state before planning.",
    )
    parser.add_argument(
        "--use-unitree-pose",
        action="store_true",
        help="Also use the live B2 x/y/yaw as the planning pose. This assumes the map frame is aligned with sport mode state.",
    )
    parser.add_argument(
        "--unitree-state-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for a B2 sport mode state sample.",
    )
    parser.add_argument(
        "--execute-unitree",
        action="store_true",
        help="Send the selected DWA command to a Unitree B2 through SDK2.",
    )
    parser.add_argument(
        "--unitree-recovery-stand",
        action="store_true",
        help="Call RecoveryStand() before sending the DWA command.",
    )
    parser.add_argument(
        "--unitree-stand-up",
        action="store_true",
        help="Call StandUp() before sending the DWA command.",
    )
    parser.add_argument(
        "--unitree-balance-stand",
        action="store_true",
        help="Call BalanceStand() before sending the DWA command.",
    )
    parser.add_argument(
        "--unitree-classic-walk",
        action="store_true",
        help="Enable ClassicWalk while executing the selected DWA command.",
    )
    parser.add_argument(
        "--unitree-command-duration",
        type=float,
        default=None,
        help="Seconds to resend the selected Move(vx, 0, vyaw). Defaults to control_interval.",
    )
    parser.add_argument(
        "--unitree-command-rate-hz",
        type=float,
        default=10.0,
        help="Rate used to resend Move(vx, 0, vyaw) while executing on the B2.",
    )
    parser.add_argument(
        "--unitree-no-stop-after-command",
        action="store_true",
        help="Do not call StopMove() after the command window ends.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DWAConfig:
    return DWAConfig(
        min_linear_velocity=args.min_linear_velocity,
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
        goal_progress_weight=args.goal_progress_weight,
        clearance_weight=args.clearance_weight,
        velocity_weight=args.velocity_weight,
        clearance_search_radius_cells=args.clearance_search_radius_cells,
        robot_radius=args.robot_radius,
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


def build_start_state(args: argparse.Namespace, run: object) -> RobotState:
    return RobotState(
        x=args.start_x,
        y=args.start_y if args.start_y is not None else args.robot_radius + run.grid.voxel_size * 0.5,
        theta=math.radians(args.start_theta_deg),
        linear_velocity=args.current_v,
        angular_velocity=args.current_w,
    )


def needs_unitree(args: argparse.Namespace) -> bool:
    return bool(args.use_unitree_state or args.use_unitree_pose or args.execute_unitree)


def build_unitree_controller(args: argparse.Namespace) -> B2SportController:
    if not args.unitree_network_interface:
        raise SystemExit(
            "--unitree-network-interface is required when using --use-unitree-state, "
            "--use-unitree-pose or --execute-unitree."
        )
    return B2SportController(
        network_interface=args.unitree_network_interface,
        sdk_root=args.unitree_sdk_root,
        command_limits=B2CommandLimits(
            max_forward_velocity=args.max_linear_velocity,
            max_angular_velocity=args.max_angular_velocity,
        ),
        read_motion_state=bool(args.use_unitree_state or args.use_unitree_pose),
    )


def main() -> None:
    args = parse_args()
    controller: B2SportController | None = None
    try:
        run = build_traversability_run(
            input_path=args.input_path,
            voxel_size=args.voxel_size,
            min_points_per_voxel=args.min_points_per_voxel,
            origin_radius=args.origin_radius,
            fill_radius=args.fill_radius,
            blocked_height_diff_voxels=args.blocked_height_diff_voxels,
        )
        print_run_stats(run)

        base_state = build_start_state(args, run)
        if needs_unitree(args):
            controller = build_unitree_controller(args)

        state = base_state
        if controller is not None and (args.use_unitree_state or args.use_unitree_pose):
            motion_state = controller.read_motion_state(timeout=args.unitree_state_timeout)
            if motion_state is None:
                raise RuntimeError("Timed out waiting for a Unitree B2 sport mode state sample.")
            state = build_robot_state_from_motion_state(
                motion_state,
                fallback_state=base_state,
                use_live_pose=args.use_unitree_pose,
            )
            print(f"unitree_live_speed_mps: {motion_state.planar_speed:.3f}")
            print(f"unitree_live_yaw_speed_rps: {motion_state.yaw_speed:.3f}")
            if args.use_unitree_pose:
                print(
                    "unitree_live_pose:"
                    f" x={motion_state.position_x:.3f}"
                    f" y={motion_state.position_y:.3f}"
                    f" yaw={math.degrees(motion_state.yaw):.1f} deg"
                )

        goal_xy = (args.goal_x, args.goal_y)
        planner = DWAPlanner(build_config(args))
        result = planner.plan(run.grid, goal_xy=goal_xy, state=state)
        print_plan_summary(result)

        base_image_rgb = build_bev_image(run, args)
        if args.output_path is not None:
            output_path = args.output_path.expanduser().resolve()
        else:
            input_name = run.input_path.stem
            output_path = (Path.cwd() / "outputs" / f"{input_name}_dwa_bev.png").resolve()
        image = render_dwa_on_base_image(
            base_image_rgb=base_image_rgb,
            result=result,
            goal_xy=goal_xy,
            state=state,
            project_xy=lambda point_xy: bev_world_to_pixel(
                x=float(point_xy[0]),
                y=float(point_xy[1]),
                x_limit=args.x_limit,
                forward_limit=args.forward_limit,
                pixels_per_meter=args.pixels_per_meter,
                margin_px=args.margin_px,
            ),
            scale_px=args.overlay_scale_px,
            draw_invalid=args.draw_invalid,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"output_image: {output_path}")

        if controller is not None and args.execute_unitree:
            if args.unitree_recovery_stand:
                print(f"unitree_recovery_stand_return_code: {controller.recovery_stand()}")
                print("unitree_wait_s: 1.0")
                time.sleep(1.0)
            if args.unitree_stand_up:
                print(f"unitree_stand_up_return_code: {controller.stand_up()}")
                print("unitree_wait_s: 1.0")
                time.sleep(1.0)
            if args.unitree_balance_stand:
                print(f"unitree_balance_stand_return_code: {controller.balance_stand()}")
                print("unitree_wait_s: 1.0")
                time.sleep(1.0)
            if args.unitree_classic_walk:
                print(f"unitree_classic_walk_enable_return_code: {controller.classic_walk(True)}")
                print("unitree_wait_s: 0.5")
                time.sleep(0.5)

            b2_command = command_from_dwa_result(
                result,
                limits=B2CommandLimits(
                    max_forward_velocity=args.max_linear_velocity,
                    max_angular_velocity=args.max_angular_velocity,
                ),
            )
            command_duration = (
                args.unitree_command_duration if args.unitree_command_duration is not None else args.control_interval
            )
            print(f"unitree_command_linear_velocity_mps: {b2_command.linear_x:.3f}")
            print(f"unitree_command_angular_velocity_rps: {b2_command.angular_z:.3f}")
            print(f"unitree_command_duration_s: {command_duration:.3f}")
            print(
                "unitree_stop_after_command:"
                f" {not args.unitree_no_stop_after_command}"
            )
            try:
                execute_code = controller.execute_velocity_command(
                    b2_command,
                    duration_s=command_duration,
                    rate_hz=args.unitree_command_rate_hz,
                    stop_after=not args.unitree_no_stop_after_command,
                )
                print(f"unitree_execute_return_code: {execute_code}")
            finally:
                if args.unitree_classic_walk:
                    print(f"unitree_classic_walk_disable_return_code: {controller.classic_walk(False)}")
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    main()
