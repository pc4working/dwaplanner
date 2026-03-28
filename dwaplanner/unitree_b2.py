"""Helpers for sending DWA velocity commands to a Unitree B2."""

from __future__ import annotations

import importlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dwa_planner import DWAResult, RobotState

DEFAULT_B2_MAX_ANGULAR_VELOCITY = 0.35
DEFAULT_B2_STATE_TOPICS = ("sportmodestate", "lf/sportmodestate", "rt/sportmodestate")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_sdk_roots(explicit_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root.expanduser().resolve())

    for env_name in ("UNITREE_SDK2_PYTHON", "UNITREE_SDK2_REPO"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser().resolve())

    candidates.extend(
        [
            (Path.home() / "unitree_sdk2_python").resolve(),
            (_repo_root().parent / "unitree_sdk2_python").resolve(),
            (_repo_root().parent.parent / "unitree_sdk2_python").resolve(),
        ]
    )
    return candidates


def ensure_unitree_sdk_on_path(sdk_root: Path | None = None) -> Path:
    for root in _candidate_sdk_roots(sdk_root):
        if (root / "unitree_sdk2py").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return root

    raise ModuleNotFoundError(
        "Could not locate the local unitree_sdk2_python repository. "
        "Pass --unitree-sdk-root or set UNITREE_SDK2_PYTHON."
    )


@dataclass(slots=True, frozen=True)
class B2CommandLimits:
    max_forward_velocity: float = 1.0
    max_lateral_velocity: float = 0.0
    max_angular_velocity: float = DEFAULT_B2_MAX_ANGULAR_VELOCITY

    def __post_init__(self) -> None:
        if self.max_forward_velocity < 0.0:
            raise ValueError("max_forward_velocity cannot be negative.")
        if self.max_lateral_velocity < 0.0:
            raise ValueError("max_lateral_velocity cannot be negative.")
        if self.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be positive.")


@dataclass(slots=True, frozen=True)
class B2VelocityCommand:
    linear_x: float
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass(slots=True, frozen=True)
class B2MotionState:
    timestamp_seconds: float
    mode: int
    gait_type: int
    progress: float
    position_x: float
    position_y: float
    position_z: float
    yaw: float
    linear_velocity_x: float
    linear_velocity_y: float
    linear_velocity_z: float
    yaw_speed: float
    body_height: float

    @property
    def planar_speed(self) -> float:
        return math.hypot(self.linear_velocity_x, self.linear_velocity_y)


@dataclass(slots=True, frozen=True)
class _UnitreeSDKModules:
    ChannelFactoryInitialize: Any
    ChannelSubscriber: Any
    SportClient: Any
    SportModeState: Any


def load_unitree_sdk_modules(sdk_root: Path | None = None) -> _UnitreeSDKModules:
    ensure_unitree_sdk_on_path(sdk_root)
    channel_module = importlib.import_module("unitree_sdk2py.core.channel")
    sport_module = importlib.import_module("unitree_sdk2py.b2.sport.sport_client")
    sport_state_module = importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_")
    return _UnitreeSDKModules(
        ChannelFactoryInitialize=channel_module.ChannelFactoryInitialize,
        ChannelSubscriber=channel_module.ChannelSubscriber,
        SportClient=sport_module.SportClient,
        SportModeState=sport_state_module.SportModeState_,
    )


def clamp_b2_velocity_command(
    command: B2VelocityCommand,
    limits: B2CommandLimits | None = None,
) -> B2VelocityCommand:
    limits = limits or B2CommandLimits()
    if not math.isfinite(command.linear_x):
        raise ValueError("linear_x must be finite.")
    if not math.isfinite(command.linear_y):
        raise ValueError("linear_y must be finite.")
    if not math.isfinite(command.angular_z):
        raise ValueError("angular_z must be finite.")

    return B2VelocityCommand(
        linear_x=float(np_clip(command.linear_x, -limits.max_forward_velocity, limits.max_forward_velocity)),
        linear_y=float(np_clip(command.linear_y, -limits.max_lateral_velocity, limits.max_lateral_velocity)),
        angular_z=float(np_clip(command.angular_z, -limits.max_angular_velocity, limits.max_angular_velocity)),
    )


def command_from_dwa_result(
    result: DWAResult,
    limits: B2CommandLimits | None = None,
) -> B2VelocityCommand:
    return clamp_b2_velocity_command(
        B2VelocityCommand(
            linear_x=float(result.best_linear_velocity),
            linear_y=0.0,
            angular_z=float(result.best_angular_velocity),
        ),
        limits=limits,
    )


def build_robot_state_from_motion_state(
    motion_state: B2MotionState,
    fallback_state: RobotState,
    *,
    use_live_pose: bool = False,
) -> RobotState:
    return RobotState(
        x=motion_state.position_x if use_live_pose else fallback_state.x,
        y=motion_state.position_y if use_live_pose else fallback_state.y,
        theta=motion_state.yaw if use_live_pose else fallback_state.theta,
        linear_velocity=motion_state.planar_speed,
        angular_velocity=motion_state.yaw_speed,
    )


def motion_state_from_dds(message: Any) -> B2MotionState:
    stamp = getattr(message, "stamp", None)
    imu_state = getattr(message, "imu_state", None)
    position = getattr(message, "position", (0.0, 0.0, 0.0))
    velocity = getattr(message, "velocity", (0.0, 0.0, 0.0))
    rpy = getattr(imu_state, "rpy", (0.0, 0.0, 0.0))
    timestamp_seconds = 0.0
    if stamp is not None:
        timestamp_seconds = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9

    return B2MotionState(
        timestamp_seconds=timestamp_seconds,
        mode=int(getattr(message, "mode", 0)),
        gait_type=int(getattr(message, "gait_type", 0)),
        progress=float(getattr(message, "progress", 0.0)),
        position_x=float(position[0]),
        position_y=float(position[1]),
        position_z=float(position[2]),
        yaw=float(rpy[2]),
        linear_velocity_x=float(velocity[0]),
        linear_velocity_y=float(velocity[1]),
        linear_velocity_z=float(velocity[2]),
        yaw_speed=float(getattr(message, "yaw_speed", 0.0)),
        body_height=float(getattr(message, "body_height", 0.0)),
    )


class B2MotionStateReader:
    def __init__(
        self,
        sdk_modules: _UnitreeSDKModules,
        *,
        topic_candidates: tuple[str, ...] = DEFAULT_B2_STATE_TOPICS,
    ) -> None:
        self._topic_candidates = tuple(topic_candidates)
        self._subscribers: list[Any] = []
        for topic_name in self._topic_candidates:
            subscriber = sdk_modules.ChannelSubscriber(topic_name, sdk_modules.SportModeState)
            subscriber.Init()
            self._subscribers.append(subscriber)

    def read(self, timeout: float = 1.0) -> B2MotionState | None:
        if not self._subscribers:
            return None

        per_topic_timeout = max(float(timeout), 0.0) / float(len(self._subscribers))
        for subscriber in self._subscribers:
            message = subscriber.Read(per_topic_timeout)
            if message is not None:
                return motion_state_from_dds(message)
        return None

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.Close()
        self._subscribers.clear()


class B2SportController:
    def __init__(
        self,
        *,
        network_interface: str,
        sdk_root: Path | None = None,
        command_limits: B2CommandLimits | None = None,
        read_motion_state: bool = False,
        state_topics: tuple[str, ...] = DEFAULT_B2_STATE_TOPICS,
        timeout: float = 10.0,
    ) -> None:
        self.command_limits = command_limits or B2CommandLimits()
        self._sdk_modules = load_unitree_sdk_modules(sdk_root)
        self._sdk_modules.ChannelFactoryInitialize(0, network_interface)

        self._sport_client = self._sdk_modules.SportClient()
        self._sport_client.SetTimeout(timeout)
        self._sport_client.Init()

        self._motion_state_reader: B2MotionStateReader | None = None
        if read_motion_state:
            self._motion_state_reader = B2MotionStateReader(
                self._sdk_modules,
                topic_candidates=state_topics,
            )

    def read_motion_state(self, timeout: float = 1.0) -> B2MotionState | None:
        if self._motion_state_reader is None:
            raise RuntimeError("Motion state reader is not enabled for this controller.")
        return self._motion_state_reader.read(timeout=timeout)

    def recovery_stand(self) -> int:
        return int(self._sport_client.RecoveryStand())

    def stand_up(self) -> int:
        return int(self._sport_client.StandUp())

    def balance_stand(self) -> int:
        return int(self._sport_client.BalanceStand())

    def classic_walk(self, enabled: bool) -> int:
        return int(self._sport_client.ClassicWalk(bool(enabled)))

    def stop(self) -> int:
        return int(self._sport_client.StopMove())

    def send_velocity_command(self, command: B2VelocityCommand) -> int:
        clipped = clamp_b2_velocity_command(command, limits=self.command_limits)
        return int(self._sport_client.Move(clipped.linear_x, clipped.linear_y, clipped.angular_z))

    def send_dwa_result(self, result: DWAResult) -> int:
        return self.send_velocity_command(command_from_dwa_result(result, limits=self.command_limits))

    def execute_velocity_command(
        self,
        command: B2VelocityCommand,
        *,
        duration_s: float,
        rate_hz: float = 10.0,
        stop_after: bool = True,
    ) -> int:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive.")
        if duration_s < 0.0:
            raise ValueError("duration_s cannot be negative.")

        last_code = self.send_velocity_command(command)
        if duration_s > 0.0:
            deadline = time.monotonic() + duration_s
            period = 1.0 / rate_hz
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(period, remaining))
                last_code = self.send_velocity_command(command)

        if stop_after:
            self.stop()
        return last_code

    def close(self) -> None:
        if self._motion_state_reader is not None:
            self._motion_state_reader.close()
            self._motion_state_reader = None


def np_clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
