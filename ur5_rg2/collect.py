import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from data_tools.deletion import confirm_and_rmtree
from hardware.zmq_core.camera_node import ZMQClientCamera
from hardware.force.robotiq_fts300 import (
    RobotiqFTS300ModbusSerial,
    RobotiqFTS300ModbusTCP,
)
from hardware.zmq_core.robot_node import ZMQClientRobot
from hardware.zmq_core.tactile_node import ZMQClientTactile


TACTILE_FIELD_KEYS = (
    "observation.tactile_left.force",
    "observation.tactile_right.force",
)
TACTILE_DATASET_KEYS = TACTILE_FIELD_KEYS + (
    "observation.tactile_left.force6d",
    "observation.tactile_right.force6d",
    "observation.tactile_left.depth",
    "observation.tactile_right.depth",
    "observation.tactile_left.rectify",
    "observation.tactile_right.rectify",
    "observation.tactile.timestamp",
    "observation.tactile.host_timestamp",
)
XENSE_LEFT_SERIAL = "OG001783"
XENSE_RIGHT_SERIAL = "OG001782"
XENSE_FORCE_HEIGHT = 35
XENSE_FORCE_WIDTH = 20
TACTILE_FORCE_SAFETY_LIMIT_N = np.asarray([20 ,20 , 20], dtype=np.float32)
TACTILE_TORQUE_SAFETY_LIMIT_NMM = np.asarray([100.0, 100.0, 100.0], dtype=np.float32)
FORCE_TORQUE_KEY = "observation.force_torque"
FORCE_TARE_SECONDS = 2.0
DEFAULT_FORCE_SAFETY_THRESHOLD_N = 5.0
DEFAULT_FORCE_SAFETY_HARD_STOP_N = 18.0
DEFAULT_TORQUE_SAFETY_THRESHOLD_NM = 0.5
DEFAULT_TORQUE_SAFETY_HARD_STOP_NM = 1.5
DEFAULT_FORCE_TORQUE_TOOL_OFFSET_M = 0.042 - 0.0034
# RESET_TCP_POSITION = np.asarray([ 0.107776706, -0.696389763,  0.199836926], dtype=np.float64)

# 垂直插入
RESET_TCP_POSITION = np.asarray([-0.0134, -0.6171,  0.2355], dtype=np.float64)

# 0913 标定
RESET_ORIENTATION_TCP_XY = np.asarray([0.0375347175, -0.5471017194], dtype=np.float64)

RESET_TCP_POSITION_RANDOM_RANGE_M = 0.05
# Initial reset posture [pi, 0, 0], then rotate +180 deg around base Z.
RESET_TCP_ROTATION_VECTOR = np.asarray(
    [0.0, np.pi, 0.0],
    dtype=np.float64,
)

_DEFAULT_COMPUTE_EPISODE_STATS = lerobot_dataset_module.compute_episode_stats
_DEFAULT_AGGREGATE_STATS = lerobot_dataset_module.aggregate_stats


def _channelwise_tactile_stats(feature_stats: dict) -> dict:
    if not feature_stats:
        return feature_stats

    mean = np.asarray(feature_stats.get("mean"))
    if mean.ndim < 2 or mean.shape[-1] != 3:
        return feature_stats

    spatial_axes = tuple(range(mean.ndim - 1))
    channel_mean = np.mean(mean, axis=spatial_axes)
    channel_std = None
    if "std" in feature_stats:
        std = np.asarray(feature_stats["std"])
        if std.shape == mean.shape:
            variance = np.mean(std**2 + (mean - channel_mean) ** 2, axis=0)
            if variance.ndim > 1:
                variance = np.mean(variance, axis=tuple(range(variance.ndim - 1)))
            channel_std = np.sqrt(variance)

    reduced_stats = dict(feature_stats)
    if "min" in feature_stats:
        reduced_stats["min"] = np.min(
            np.asarray(feature_stats["min"]),
            axis=tuple(range(np.asarray(feature_stats["min"]).ndim - 1)),
        )
    if "max" in feature_stats:
        reduced_stats["max"] = np.max(
            np.asarray(feature_stats["max"]),
            axis=tuple(range(np.asarray(feature_stats["max"]).ndim - 1)),
        )
    reduced_stats["mean"] = channel_mean
    if channel_std is not None:
        reduced_stats["std"] = channel_std
    return reduced_stats


def aggregate_stats_with_channelwise_tactile(
    stats_list: list[dict[str, dict]],
) -> dict[str, dict[str, np.ndarray]]:
    normalized_stats = []
    for stats in stats_list:
        normalized = dict(stats)
        for key in TACTILE_FIELD_KEYS:
            if key in normalized:
                normalized[key] = _channelwise_tactile_stats(normalized[key])
        normalized_stats.append(normalized)

    return _DEFAULT_AGGREGATE_STATS(normalized_stats)


def compute_episode_stats_with_channelwise_tactile(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
) -> dict:
    ep_stats = _DEFAULT_COMPUTE_EPISODE_STATS(episode_data, features)

    for key in TACTILE_FIELD_KEYS:
        if key not in episode_data:
            continue

        tactile_array = np.asarray(episode_data[key], dtype=np.float32)
        if tactile_array.ndim < 3 or tactile_array.shape[-1] != 3:
            continue

        reduce_axes = tuple(range(tactile_array.ndim - 1))
        ep_stats[key] = {
            "min": np.min(tactile_array, axis=reduce_axes),
            "max": np.max(tactile_array, axis=reduce_axes),
            "mean": np.mean(tactile_array, axis=reduce_axes),
            "std": np.std(tactile_array, axis=reduce_axes),
            "count": np.array([len(tactile_array)]),
        }

    return ep_stats


lerobot_dataset_module.compute_episode_stats = compute_episode_stats_with_channelwise_tactile
lerobot_dataset_module.aggregate_stats = aggregate_stats_with_channelwise_tactile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect LeRobot episodes on a UR5 + RG2 with "
            "dual Xense tactile sensors and keyboard teleop."
        )
    )
    parser.add_argument("--repo-id", default="ur5_rg2_xense_insert")
    parser.add_argument(
        "--root",
        default="./data/ur5_rg2_xense_insert",
    )
    parser.add_argument("--task", default="Insert the black cylinder vertically into the hole.")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument(
        "--robot-timeout-ms",
        type=int,
        default=10000,
        help="ZMQ timeout for robot requests; first RG2 command can take several seconds.",
    )
    parser.add_argument("--wrist-camera-port", type=int, default=5000)
    parser.add_argument("--base-camera-port", type=int, default=5001)
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument(
        "--collect-tactile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect Xense tactile fields. Use --no-collect-tactile to disable.",
    )
    parser.add_argument(
        "--xense-left-serial",
        default=XENSE_LEFT_SERIAL,
        help="Compatibility/logging option; Xense devices are opened by ur5_rg2/launch.py.",
    )
    parser.add_argument(
        "--xense-right-serial",
        default=XENSE_RIGHT_SERIAL,
        help="Compatibility/logging option; Xense devices are opened by ur5_rg2/launch.py.",
    )
    parser.add_argument("--xense-force-height", type=int, default=XENSE_FORCE_HEIGHT)
    parser.add_argument("--xense-force-width", type=int, default=XENSE_FORCE_WIDTH)
    parser.add_argument(
        "--xense-depth-height",
        type=int,
        default=None,
        help="Saved Xense Depth height. Defaults to --xense-force-height.",
    )
    parser.add_argument(
        "--xense-depth-width",
        type=int,
        default=None,
        help="Saved Xense Depth width. Defaults to --xense-force-width.",
    )
    parser.add_argument(
        "--xense-rectify-height",
        type=int,
        default=None,
        help="Saved Xense Rectify image height. Defaults to --image-size.",
    )
    parser.add_argument(
        "--xense-rectify-width",
        type=int,
        default=None,
        help="Saved Xense Rectify image width. Defaults to --image-size.",
    )
    parser.add_argument(
        "--require-valid-tactile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When collecting tactile data, skip frames whose Xense source reports valid=False.",
    )
    parser.add_argument(
        "--collect-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect force/torque wrench as observation.force_torque. Use --no-collect-force to disable.",
    )
    parser.add_argument(
        "--force-source",
        choices=("modbus-serial", "modbus-tcp", "ur-rtde"),
        default="modbus-serial",
        help="Read force from Robotiq FTS-300-S Modbus RTU/TCP or UR RTDE TCP force.",
    )
    parser.add_argument("--force-serial-port", default=None)
    parser.add_argument("--force-serial-slave-address", type=int, default=9)
    parser.add_argument("--force-serial-register-address", type=int, default=180)
    parser.add_argument("--force-serial-function-code", type=int, default=3)
    parser.add_argument("--force-serial-baudrate", type=int, default=19200)
    parser.add_argument("--force-serial-timeout-s", type=float, default=0.02)
    parser.add_argument("--force-serial-retries", type=int, default=3)
    parser.add_argument(
        "--force-serial-wakeup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send 50 bytes of 0xff before opening the Modbus RTU instrument.",
    )
    parser.add_argument("--force-modbus-host", default="192.168.1.1")
    parser.add_argument("--force-modbus-port", type=int, default=502)
    parser.add_argument("--force-modbus-unit-id", type=int, default=9)
    parser.add_argument("--force-modbus-address", type=int, default=0)
    parser.add_argument(
        "--force-modbus-function",
        choices=("holding", "input"),
        default="input",
    )
    parser.add_argument(
        "--force-modbus-format",
        choices=("int16", "int32", "float32"),
        default="int32",
    )
    parser.add_argument(
        "--force-modbus-word-order",
        choices=("big", "little"),
        default="big",
    )
    parser.add_argument("--force-modbus-scale", type=float, default=1.0)
    parser.add_argument("--force-modbus-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--force-print-every",
        type=int,
        default=1,
        help="Print force/torque every N collected frames when force collection is enabled.",
    )
    parser.add_argument(
        "--force-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When force collection is enabled, apply a task-agnostic Cartesian "
            "admittance correction to reduce excessive x/y/z force and torque."
        ),
    )
    parser.add_argument(
        "--force-safety-threshold-n",
        default=f"{DEFAULT_FORCE_SAFETY_THRESHOLD_N},{DEFAULT_FORCE_SAFETY_THRESHOLD_N},{DEFAULT_FORCE_SAFETY_THRESHOLD_N}",
        help="Per-axis x,y,z force threshold in N before correction starts.",
    )
    parser.add_argument(
        "--force-safety-hard-stop-n",
        default=f"{DEFAULT_FORCE_SAFETY_HARD_STOP_N},{DEFAULT_FORCE_SAFETY_HARD_STOP_N},{DEFAULT_FORCE_SAFETY_HARD_STOP_N}",
        help="Per-axis x,y,z hard force limit in N. Above this, teleop motion is suppressed and only retreat is commanded.",
    )
    parser.add_argument(
        "--torque-safety-threshold-nm",
        default=f"{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM},{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM},{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM}",
        help="Per-axis rx,ry,rz torque threshold in Nm before posture correction starts.",
    )
    parser.add_argument(
        "--torque-safety-hard-stop-nm",
        default=f"{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM},{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM},{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM}",
        help="Per-axis rx,ry,rz hard torque limit in Nm. Above this, teleop motion is suppressed and only retreat is commanded.",
    )
    parser.add_argument(
        "--force-torque-tool-offset-m",
        type=float,
        default=DEFAULT_FORCE_TORQUE_TOOL_OFFSET_M,
        help=(
            "Lever arm used by the fts300-tool torque mapping, matching "
            "ur5moverealrobot3.py's d = 0.042 - 0.0034."
        ),
    )
    parser.add_argument(
        "--force-safety-axis-signs",
        default=None,
        help=(
            "Manual per-axis sign correction from the force sensor frame to the "
            "selected force safety frame. Use -1 to flip an axis if the retreat "
            "direction is wrong. By default this is selected from --force-safety-axis-preset."
        ),
    )
    parser.add_argument(
        "--force-safety-axis-order",
        default=None,
        help=(
            "Manual axis reorder from raw force sensor xyz to force safety xyz, "
            "e.g. 0,2,1 maps [Fx,Fy,Fz] to [Fx,Fz,Fy]. By default this is "
            "selected from --force-safety-axis-preset."
        ),
    )
    parser.add_argument(
        "--force-safety-axis-preset",
        choices=("auto", "identity", "fts300-tool"),
        default="auto",
        help=(
            "Force axis mapping preset. auto uses fts300-tool for Modbus FTS-300 "
            "and identity for UR RTDE. fts300-tool follows ur5moverealrobot3.py: "
            "[Fx,Fy,Fz] -> [-Fx,-Fz,-Fy]."
        ),
    )
    parser.add_argument(
        "--force-safety-frame",
        choices=("auto", "tool", "base"),
        default="auto",
        help=(
            "Frame of the signed x/y/z force used by force safety. auto uses "
            "tool for Modbus FTS-300 and base for UR RTDE."
        ),
    )
    parser.add_argument(
        "--force-safety-inv-mass",
        type=float,
        default=0.40,
        help="Inverse mass term for the translational admittance correction.",
    )
    parser.add_argument(
        "--force-safety-damping",
        type=float,
        default=18.0,
        help="Damping term for the translational admittance correction.",
    )
    parser.add_argument(
        "--force-safety-stiffness",
        type=float,
        default=80.0,
        help="Stiffness term pulling the correction back to zero.",
    )
    parser.add_argument(
        "--torque-safety-inv-inertia",
        type=float,
        default=0.08,
        help="Inverse inertia term for the rotational admittance correction.",
    )
    parser.add_argument(
        "--torque-safety-damping",
        type=float,
        default=10.0,
        help="Damping term for the rotational admittance correction.",
    )
    parser.add_argument(
        "--torque-safety-stiffness",
        type=float,
        default=35.0,
        help="Stiffness term pulling the posture correction back to zero.",
    )
    parser.add_argument(
        "--force-safety-max-vel",
        type=float,
        default=0.025,
        help="Max Cartesian retreat velocity in m/s.",
    )
    parser.add_argument(
        "--force-safety-max-correction-m",
        type=float,
        default=0.002,
        help="Max absolute Cartesian correction in m.",
    )
    parser.add_argument(
        "--torque-safety-max-angular-vel",
        type=float,
        default=0.10,
        help="Max rotational retreat velocity in rad/s.",
    )
    parser.add_argument(
        "--torque-safety-max-correction-rad",
        type=float,
        default=0.035,
        help="Max absolute posture correction in rad.",
    )
    parser.add_argument(
        "--force-safety-deadband-n",
        type=float,
        default=0.30,
        help="Force deadband in N to avoid chattering near the threshold.",
    )
    parser.add_argument(
        "--torque-safety-deadband-nm",
        type=float,
        default=0.03,
        help="Torque deadband in Nm to avoid chattering near the threshold.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gripper-force", type=float, default=4.0)
    parser.add_argument("--move-step", type=float, default=0.001)
    parser.add_argument("--rot-step", type=float, default=0.06)
    parser.add_argument("--shift-speed-scale", type=float, default=5)
    parser.add_argument("--screw-pitch", type=float, default=0.0025)
    parser.add_argument(
        "--screw-rot-step",
        type=float,
        default=0.05,
        help="Screw rotation increment in rad per control frame when --screw-rpm is unset.",
    )
    parser.add_argument(
        "--screw-rpm",
        type=float,
        default=None,
        help="Screw speed in revolutions per minute; overrides --screw-rot-step.",
    )
    parser.add_argument(
        "--reset-max-joint-step-rad",
        type=float,
        default=0.6,
        help="Max joint step per control frame when moving to L reset through IK.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-command", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--no-tactile",
        action="store_false",
        dest="collect_tactile",
        help="Alias for --no-collect-tactile.",
    )
    args = parser.parse_args()
    if args.xense_depth_height is None:
        args.xense_depth_height = args.xense_force_height
    if args.xense_depth_width is None:
        args.xense_depth_width = args.xense_force_width
    if args.xense_rectify_height is None:
        args.xense_rectify_height = args.image_size
    if args.xense_rectify_width is None:
        args.xense_rectify_width = args.image_size
    if args.xense_force_height <= 0:
        parser.error("--xense-force-height must be positive.")
    if args.xense_force_width <= 0:
        parser.error("--xense-force-width must be positive.")
    if args.xense_depth_height <= 0:
        parser.error("--xense-depth-height must be positive.")
    if args.xense_depth_width <= 0:
        parser.error("--xense-depth-width must be positive.")
    if args.xense_rectify_height <= 0:
        parser.error("--xense-rectify-height must be positive.")
    if args.xense_rectify_width <= 0:
        parser.error("--xense-rectify-width must be positive.")
    return args


def make_dataset(args: argparse.Namespace) -> LeRobotDataset:
    root = Path(args.root)
    complete_dataset = (root / "meta").exists() and (root / "data").exists()

    if args.overwrite and root.exists():
        confirm_and_rmtree(root, reason="--overwrite")
        complete_dataset = False

    if complete_dataset and args.resume:
        print(f"Load existing dataset from {root}")
        return LeRobotDataset(args.repo_id, root=str(root))

    if root.exists():
        if complete_dataset:
            print(f"Directory {root} already contains a LeRobot dataset.")
            ans = input("Overwrite it, continue saving, or quit? [o/c/q] ").strip().lower()
            if ans in ("c", "continue", "r", "resume"):
                print(f"Load existing dataset from {root}")
                return LeRobotDataset(args.repo_id, root=str(root))
            if ans in ("o", "overwrite", "y", "yes"):
                confirm_and_rmtree(root)
            else:
                raise SystemExit("Quit without changing the dataset.")
        else:
            print(f"Directory {root} exists but is not a complete LeRobot dataset.")
            ans = input("Overwrite it and create a new dataset, or quit? [o/q] ").strip().lower()
            if ans in ("o", "overwrite", "y", "yes"):
                confirm_and_rmtree(root)
            else:
                raise SystemExit("Quit without changing the dataset.")

    print(f"Create new dataset at {root}")
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
        "obj_init": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["obj_init"],
        },
    }

    if args.collect_tactile:
        xense_force_shape = (args.xense_force_height, args.xense_force_width, 3)
        xense_depth_shape = (args.xense_depth_height, args.xense_depth_width, 3)
        xense_rectify_shape = (
            args.xense_rectify_height,
            args.xense_rectify_width,
            3,
        )
        features.update(
            {
                "observation.tactile_left.force": {
                    "dtype": "float32",
                    "shape": xense_force_shape,
                    "names": ["height", "width", "axis"],
                },
                "observation.tactile_left.force6d": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_torque"],
                },
                "observation.tactile_left.depth": {
                    "dtype": "image",
                    "shape": xense_depth_shape,
                    "names": ["height", "width", "channels"],
                },
                "observation.tactile_left.rectify": {
                    "dtype": "image",
                    "shape": xense_rectify_shape,
                    "names": ["height", "width", "channels"],
                },
                "observation.tactile_right.force": {
                    "dtype": "float32",
                    "shape": xense_force_shape,
                    "names": ["height", "width", "axis"],
                },
                "observation.tactile_right.force6d": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_torque"],
                },
                "observation.tactile_right.depth": {
                    "dtype": "image",
                    "shape": xense_depth_shape,
                    "names": ["height", "width", "channels"],
                },
                "observation.tactile_right.rectify": {
                    "dtype": "image",
                    "shape": xense_rectify_shape,
                    "names": ["height", "width", "channels"],
                },
                "observation.tactile.timestamp": {
                    "dtype": "float64",
                    "shape": (2,),
                    "names": ["sensor"],
                },
                "observation.tactile.host_timestamp": {
                    "dtype": "float64",
                    "shape": (2,),
                    "names": ["sensor"],
                },
            }
        )

    if args.collect_force:
        features["observation.force_torque"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": ["force_torque"],
        }

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(root),
        robot_type="ur5_rg2",
        fps=args.fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )


def resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(image.astype(np.uint8))
    image = image.resize((image_size, image_size))
    return np.asarray(image, dtype=np.uint8)


def reset_episode_buffer(dataset: LeRobotDataset) -> None:
    if dataset.episode_buffer is None:
        dataset.episode_buffer = dataset.create_episode_buffer()
    else:
        dataset.clear_episode_buffer()


def dataset_has_tactile_features(dataset: LeRobotDataset) -> bool:
    return all(key in dataset.features for key in TACTILE_DATASET_KEYS)


def dataset_has_force_feature(dataset: LeRobotDataset) -> bool:
    return FORCE_TORQUE_KEY in dataset.features


def validate_requested_sensor_features(
    dataset: LeRobotDataset,
    args: argparse.Namespace,
) -> tuple[bool, bool]:
    store_tactile = dataset_has_tactile_features(dataset)
    store_force = dataset_has_force_feature(dataset)

    if args.collect_tactile and not store_tactile:
        raise RuntimeError(
            "This dataset was created without tactile features, but "
            "--collect-tactile is enabled. Use --overwrite, a new --root, or "
            "run with --no-collect-tactile."
        )
    if args.collect_tactile and store_tactile:
        expected_depth_shape = (
            args.xense_depth_height,
            args.xense_depth_width,
            3,
        )
        for key in (
            "observation.tactile_left.depth",
            "observation.tactile_right.depth",
        ):
            feature = dataset.features.get(key, {})
            if (
                feature.get("dtype") != "image"
                or tuple(feature.get("shape", ())) != expected_depth_shape
            ):
                raise RuntimeError(
                    f"{key} in this dataset uses schema {feature}, but the "
                    "collector now saves Xense depth as an image with schema "
                    f"dtype='image', shape={expected_depth_shape}. Use "
                    "--overwrite or a new --root to create a new dataset."
                )
    if args.collect_force and not store_force:
        raise RuntimeError(
            "This dataset was created without observation.force_torque, but "
            "--collect-force is enabled. Use --overwrite, a new --root, or "
            "run with --no-collect-force."
        )

    return store_tactile, store_force


def format_force_torque(force_torque: np.ndarray) -> np.ndarray:
    array = np.asarray(force_torque, dtype=np.float32).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return array[:6].astype(np.float32)


def print_force_torque(force_torque: np.ndarray) -> None:
    values = [float(v) for v in force_torque]
    # print(
    #     "FTS-300-S force_torque "
    #     f"[Fx,Fy,Fz,Tx,Ty,Tz]=[{values[0]: .2f}, {values[1]: .2f}, "
    #     f"{values[2]: .2f}, {values[3]: .2f}, {values[4]: .2f}, {values[5]: .2f}]"
    # )


def parse_xyz_vector(value: str, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        parts = [float(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be three comma-separated numbers, got {value!r}") from exc
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise ValueError(f"{name} must contain one value or three comma-separated values.")
    array = np.asarray(parts, dtype=np.float64)
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive.")
    return array


def parse_axis_order(value: str) -> np.ndarray:
    try:
        order = [int(part.strip()) for part in str(value).split(",")]
    except ValueError as exc:
        raise ValueError(
            f"--force-safety-axis-order must be three comma-separated integers, got {value!r}"
        ) from exc
    if sorted(order) != [0, 1, 2]:
        raise ValueError("--force-safety-axis-order must be a permutation of 0,1,2.")
    return np.asarray(order, dtype=int)


def resolve_force_safety_axis_mapping(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    preset = args.force_safety_axis_preset
    if preset == "auto":
        if args.force_source in ("modbus-serial", "modbus-tcp"):
            preset = "fts300-tool"
        else:
            preset = "identity"

    if preset == "fts300-tool":
        # Matches getPureForceTorqueToTool() in ur5moverealrobot3.py:
        # raw [Fx,Fy,Fz] -> tool [-Fx,-Fz,-Fy].
        axis_order = np.asarray([0, 1, 2], dtype=int)
        axis_signs = np.asarray([1.0, 1.0, -1.0], dtype=np.float64)
    elif preset == "identity":
        axis_order = np.asarray([0, 1, 2], dtype=int)
        axis_signs = np.ones(3, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported force safety axis preset: {preset}")

    if args.force_safety_axis_order is not None:
        axis_order = parse_axis_order(args.force_safety_axis_order)
    if args.force_safety_axis_signs is not None:
        axis_signs = parse_xyz_vector(args.force_safety_axis_signs, "--force-safety-axis-signs")
        axis_signs = np.where(axis_signs >= 0.0, 1.0, -1.0)

    return axis_order, axis_signs, preset


def resolve_force_safety_frame(args: argparse.Namespace) -> str:
    if args.force_safety_frame != "auto":
        return args.force_safety_frame
    if args.force_source == "ur-rtde":
        return "base"
    return "tool"


class CartesianForceSafetyController:
    """
    Task-agnostic force guard for kinesthetic/teleop collection.

    It follows the same idea as the admittance code in ur5moverealrobot3.py:
    excessive Cartesian force/torque is converted into a small compliant retreat
    before the commanded TCP target is sent. Translation and posture are corrected;
    the operator's gripper command is left untouched unless a hard stop is active.
    """

    def __init__(self, args: argparse.Namespace, dt: float) -> None:
        force_threshold = parse_xyz_vector(
            args.force_safety_threshold_n,
            "--force-safety-threshold-n",
            positive=True,
        )
        torque_threshold = parse_xyz_vector(
            args.torque_safety_threshold_nm,
            "--torque-safety-threshold-nm",
            positive=True,
        )
        force_hard_stop = parse_xyz_vector(
            args.force_safety_hard_stop_n,
            "--force-safety-hard-stop-n",
            positive=True,
        )
        torque_hard_stop = parse_xyz_vector(
            args.torque_safety_hard_stop_nm,
            "--torque-safety-hard-stop-nm",
            positive=True,
        )
        self.threshold = np.concatenate([force_threshold, torque_threshold])
        self.hard_stop = np.concatenate([force_hard_stop, torque_hard_stop])
        self.force_frame = resolve_force_safety_frame(args)
        self.axis_order, self.axis_signs, self.axis_preset = resolve_force_safety_axis_mapping(args)
        self.tool_offset_m = float(args.force_torque_tool_offset_m)
        self.inv_mass = np.concatenate(
            [
                np.full(3, float(args.force_safety_inv_mass), dtype=np.float64),
                np.full(3, float(args.torque_safety_inv_inertia), dtype=np.float64),
            ]
        )
        self.damping = np.concatenate(
            [
                np.full(3, float(args.force_safety_damping), dtype=np.float64),
                np.full(3, float(args.torque_safety_damping), dtype=np.float64),
            ]
        )
        self.stiffness = np.concatenate(
            [
                np.full(3, float(args.force_safety_stiffness), dtype=np.float64),
                np.full(3, float(args.torque_safety_stiffness), dtype=np.float64),
            ]
        )
        self.max_vel = np.concatenate(
            [
                np.full(3, abs(float(args.force_safety_max_vel)), dtype=np.float64),
                np.full(3, abs(float(args.torque_safety_max_angular_vel)), dtype=np.float64),
            ]
        )
        self.max_correction = np.concatenate(
            [
                np.full(3, abs(float(args.force_safety_max_correction_m)), dtype=np.float64),
                np.full(3, abs(float(args.torque_safety_max_correction_rad)), dtype=np.float64),
            ]
        )
        self.deadband = np.concatenate(
            [
                np.full(3, abs(float(args.force_safety_deadband_n)), dtype=np.float64),
                np.full(3, abs(float(args.torque_safety_deadband_nm)), dtype=np.float64),
            ]
        )
        self.dt = float(dt)
        self.offset = np.zeros(6, dtype=np.float64)
        self.velocity = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        self.offset.fill(0.0)
        self.velocity.fill(0.0)

    def _wrench_in_base(self, force_torque: np.ndarray, current_tcp: np.ndarray) -> np.ndarray:
        raw = np.asarray(force_torque, dtype=np.float64).reshape(-1)
        if raw.size < 6:
            raw = np.pad(raw, (0, 6 - raw.size))
        force_raw = raw[:3]
        torque_raw = raw[3:6]
        force = force_raw[self.axis_order] * self.axis_signs
        if self.axis_preset == "fts300-tool":
            torque = np.asarray(
                [
                    (torque_raw[0] - force_raw[1] * self.tool_offset_m),
                    (torque_raw[1] + self.tool_offset_m * force_raw[0]),
                    -torque_raw[2],
                ],
                dtype=np.float64,
            )
        else:
            torque = torque_raw[self.axis_order] * self.axis_signs

        if self.force_frame == "tool":
            rotation = rotvec_to_matrix(np.asarray(current_tcp, dtype=np.float64)[3:6])
            force = rotation @ force
            torque = rotation @ torque
        return np.concatenate([force, torque])

    def update(
        self,
        force_torque: np.ndarray,
        current_tcp: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        wrench = self._wrench_in_base(force_torque, current_tcp)
        abs_wrench = np.abs(wrench)
        active = abs_wrench > (self.threshold + self.deadband)
        excess = np.zeros(6, dtype=np.float64)
        excess[active] = np.sign(wrench[active]) * (abs_wrench[active] - self.threshold[active])

        # Oppose the measured wrench. The spring term smoothly returns the
        # correction to zero after contact force drops.
        acceleration = (
            -self.inv_mass * excess
            - self.damping * self.velocity
            - self.stiffness * self.offset
        )
        self.velocity += acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.max_vel, self.max_vel)
        self.offset += self.velocity * self.dt
        self.offset = np.clip(self.offset, -self.max_correction, self.max_correction)

        hard_stop = bool(np.any(abs_wrench > self.hard_stop))
        return self.offset.copy(), hard_stop, wrench

    def correct_target(
        self,
        current_tcp: np.ndarray,
        target_tcp: np.ndarray,
        force_torque: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
        correction, hard_stop, signed_wrench = self.update(force_torque, current_tcp)
        corrected = np.asarray(target_tcp, dtype=np.float64).copy()
        current_tcp = np.asarray(current_tcp, dtype=np.float64)
        if hard_stop:
            corrected[:3] = current_tcp[:3]
            corrected[3:6] = current_tcp[3:6]

        corrected[:3] += correction[:3]
        correction_rotation = rotvec_to_matrix(correction[3:6])
        target_rotation = rotvec_to_matrix(corrected[3:6])
        corrected[3:6] = matrix_to_rotvec(correction_rotation @ target_rotation)
        return corrected, hard_stop, correction, signed_wrench


class ForceReader:
    def __init__(self, args: argparse.Namespace, robot: ZMQClientRobot) -> None:
        self.source = args.force_source
        self._robot = robot
        self._modbus = None
        if self.source == "modbus-serial":
            self._modbus = RobotiqFTS300ModbusSerial(
                port=args.force_serial_port,
                slave_address=args.force_serial_slave_address,
                register_address=args.force_serial_register_address,
                function_code=args.force_serial_function_code,
                baudrate=args.force_serial_baudrate,
                timeout_s=args.force_serial_timeout_s,
                wakeup=args.force_serial_wakeup,
                retries=args.force_serial_retries,
            )
        elif self.source == "modbus-tcp":
            self._modbus = RobotiqFTS300ModbusTCP(
                host=args.force_modbus_host,
                port=args.force_modbus_port,
                unit_id=args.force_modbus_unit_id,
                register_address=args.force_modbus_address,
                function=args.force_modbus_function,
                value_format=args.force_modbus_format,
                word_order=args.force_modbus_word_order,
                scale=args.force_modbus_scale,
                timeout_s=args.force_modbus_timeout_s,
            )

    def read(self) -> np.ndarray:
        if self.source in ("modbus-serial", "modbus-tcp"):
            if self._modbus is None:
                raise RuntimeError("Modbus force reader is not initialized")
            return format_force_torque(self._modbus.read())
        if self.source == "ur-rtde":
            return format_force_torque(self._robot.get_tcp_force())
        raise RuntimeError(f"Unsupported force source: {self.source}")

    def close(self) -> None:
        close = getattr(self._modbus, "close", None)
        if close is not None:
            close()


def calibrate_force_bias(force_reader: ForceReader, fps: int) -> np.ndarray:
    num_samples = max(10, int(FORCE_TARE_SECONDS * fps))
    samples = []
    print(
        "Calibrating FTS-300-S force bias. Keep the end effector still and "
        "unloaded..."
    )
    for _ in range(num_samples):
        samples.append(force_reader.read())
        time.sleep(1.0 / max(1, fps))

    bias = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    values = [float(v) for v in bias]
    print(
        "FTS-300-S bias "
        f"[Fx,Fy,Fz,Tx,Ty,Tz]=[{values[0]: .3f}, {values[1]: .3f}, "
        f"{values[2]: .3f}, {values[3]: .3f}, {values[4]: .3f}, {values[5]: .3f}]"
    )
    return bias


def empty_tactile_frame(
    force_height: int,
    force_width: int,
    depth_height: int,
    depth_width: int,
    rectify_height: int,
    rectify_width: int,
) -> dict[str, np.ndarray | float | bool]:
    return {
        "force": np.zeros((force_height, force_width, 3), dtype=np.float32),
        "force6d": np.zeros(6, dtype=np.float32),
        "depth": np.zeros((depth_height, depth_width, 3), dtype=np.uint8),
        "rectify": np.zeros((rectify_height, rectify_width, 3), dtype=np.uint8),
        "timestamp": 0.0,
        "host_timestamp": 0.0,
        "valid": False,
    }


def _fit_xense_force_array(
    value: np.ndarray | None,
    force_height: int,
    force_width: int,
) -> tuple[np.ndarray, bool]:
    formatted = np.zeros((force_height, force_width, 3), dtype=np.float32)
    if value is None:
        return formatted, False

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return formatted, False

    if array.size == 0:
        return formatted, False

    if array.ndim == 3 and array.shape[-1] == 3:
        height = min(array.shape[0], force_height)
        width = min(array.shape[1], force_width)
        formatted[:height, :width] = array[:height, :width, :3]
        return formatted, True

    try:
        flattened = array.reshape(-1, 3)
    except ValueError:
        return formatted, False

    count = min(flattened.shape[0], force_height * force_width)
    formatted.reshape(-1, 3)[:count] = flattened[:count]
    return formatted, count > 0


def _resize_float_map(array: np.ndarray, height: int, width: int) -> np.ndarray:
    if array.shape == (height, width):
        return array.astype(np.float32, copy=False)
    image = Image.fromarray(array.astype(np.float32, copy=False))
    image = image.resize((width, height))
    return np.asarray(image, dtype=np.float32)


def _fit_xense_depth_array(
    value: np.ndarray | None,
    depth_height: int,
    depth_width: int,
) -> tuple[np.ndarray, bool]:
    formatted = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
    if value is None:
        return formatted, False

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return formatted, False

    if array.size == 0:
        return formatted, False

    array = np.squeeze(array)
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            array = array[..., 0]
        elif array.shape[0] in (1, 3, 4):
            array = array[0]
    if array.ndim == 2:
        try:
            array = _resize_float_map(array, depth_height, depth_width)
        except (TypeError, ValueError):
            height = min(array.shape[0], depth_height)
            width = min(array.shape[1], depth_width)
            resized = np.zeros((depth_height, depth_width), dtype=np.float32)
            resized[:height, :width] = np.asarray(
                array[:height, :width],
                dtype=np.float32,
            )
            array = resized
        image = _to_uint8_image(array)
        return np.repeat(image[..., None], 3, axis=-1), True

    flattened = np.asarray(array, dtype=np.float32).reshape(-1)
    count = min(flattened.shape[0], depth_height * depth_width)
    map_2d = np.zeros((depth_height, depth_width), dtype=np.float32)
    map_2d.reshape(-1)[:count] = flattened[:count]
    image = _to_uint8_image(map_2d)
    return np.repeat(image[..., None], 3, axis=-1), count > 0


def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array

    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    if finite.size > 0 and float(finite.max()) <= 1.0:
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _fit_xense_rectify_image(
    value: np.ndarray | None,
    rectify_height: int,
    rectify_width: int,
) -> tuple[np.ndarray, bool]:
    formatted = np.zeros((rectify_height, rectify_width, 3), dtype=np.uint8)
    if value is None:
        return formatted, False

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return formatted, False

    if array.size == 0:
        return formatted, False

    array = np.squeeze(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3:
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array[:3], 0, -1)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] >= 3:
            array = array[..., :3]
        elif array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[0] >= 3:
            array = np.moveaxis(array[:3], 0, -1)
        elif array.shape[0] == 1:
            array = np.repeat(array[0][..., None], 3, axis=-1)
        else:
            return formatted, False
    else:
        return formatted, False

    array = _to_uint8_image(array)
    if array.shape[:2] != (rectify_height, rectify_width):
        image = Image.fromarray(array)
        image = image.resize((rectify_width, rectify_height))
        array = np.asarray(image, dtype=np.uint8)
    return array.astype(np.uint8, copy=False), True


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return 0.0


def format_tactile_frame(
    frame: dict[str, np.ndarray | float | bool],
    force_height: int,
    force_width: int,
    depth_height: int,
    depth_width: int,
    rectify_height: int,
    rectify_width: int,
) -> dict[str, np.ndarray | float | bool]:
    formatted = empty_tactile_frame(
        force_height,
        force_width,
        depth_height,
        depth_width,
        rectify_height,
        rectify_width,
    )
    force, force_ok = _fit_xense_force_array(
        frame.get("force"),
        force_height,
        force_width,
    )
    depth, _ = _fit_xense_depth_array(
        frame.get("depth"),
        depth_height,
        depth_width,
    )
    rectify, _ = _fit_xense_rectify_image(
        frame.get("rectify"),
        rectify_height,
        rectify_width,
    )
    force6d_value = frame.get("force6d")
    if force6d_value is None:
        force6d_value = np.zeros(6, dtype=np.float32)
    formatted["force"] = force
    formatted["force6d"] = format_force_torque(force6d_value)
    formatted["depth"] = depth
    formatted["rectify"] = rectify
    formatted["valid"] = bool(frame.get("valid", force_ok)) and force_ok
    formatted["timestamp"] = (
        _float_or_zero(frame.get("timestamp")) if formatted["valid"] else 0.0
    )
    formatted["host_timestamp"] = (
        _float_or_zero(frame.get("host_timestamp")) if formatted["valid"] else 0.0
    )
    return formatted


def empty_tactile_frame_from_args(
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float | bool]:
    return empty_tactile_frame(
        args.xense_force_height,
        args.xense_force_width,
        args.xense_depth_height,
        args.xense_depth_width,
        args.xense_rectify_height,
        args.xense_rectify_width,
    )


def format_tactile_frame_from_args(
    frame: dict[str, np.ndarray | float | bool],
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float | bool]:
    return format_tactile_frame(
        frame,
        args.xense_force_height,
        args.xense_force_width,
        args.xense_depth_height,
        args.xense_depth_width,
        args.xense_rectify_height,
        args.xense_rectify_width,
    )


def tactile_wrench_is_safe(frame: dict[str, np.ndarray | float | bool]) -> bool:
    wrench = format_force_torque(frame["force6d"])
    return bool(
        np.all(np.abs(wrench[:3]) < TACTILE_FORCE_SAFETY_LIMIT_N)
        and np.all(np.abs(wrench[3:6]) < TACTILE_TORQUE_SAFETY_LIMIT_NMM)
    )


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    axis_skew = skew(axis)
    return (
        np.eye(3)
        + np.sin(theta) * axis_skew
        + (1.0 - np.cos(theta)) * (axis_skew @ axis_skew)
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cos_theta = (np.trace(rotation) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        diag = np.diag(rotation)
        axis_index = int(np.argmax(diag))
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = np.sqrt(max(0.0, (diag[axis_index] + 1.0) / 2.0))
        if axis[axis_index] < 1e-8:
            axis[axis_index] = 1.0
        for idx in range(3):
            if idx == axis_index:
                continue
            axis[idx] = (
                rotation[idx, axis_index] + rotation[axis_index, idx]
            ) / (4.0 * axis[axis_index])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3)
        return axis / norm * theta
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(theta)
    return axis * theta


def rotation_matrix(angle: float, direction: tuple[float, float, float]) -> np.ndarray:
    return rotvec_to_matrix(np.asarray(direction, dtype=np.float64) * angle)


def _project_unit_to_plane(
    vector: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray | None:
    projected = vector - np.dot(vector, normal) * normal
    norm = float(np.linalg.norm(projected))
    if norm < 1e-9:
        return None
    return projected / norm


def vertical_reset_rotvec_preserving_heading(
    current_rotvec: np.ndarray,
    reset_rotvec: np.ndarray,
) -> np.ndarray:
    reset_rot = rotvec_to_matrix(np.asarray(reset_rotvec, dtype=np.float64))
    current_rot = rotvec_to_matrix(np.asarray(current_rotvec, dtype=np.float64))
    reset_axis = reset_rot[:, 2].copy()
    reset_axis /= np.linalg.norm(reset_axis)

    reset_ref = _project_unit_to_plane(reset_rot[:, 0], reset_axis)
    current_ref = _project_unit_to_plane(current_rot[:, 0], reset_axis)
    if reset_ref is None or current_ref is None:
        reset_ref = _project_unit_to_plane(reset_rot[:, 1], reset_axis)
        current_ref = _project_unit_to_plane(current_rot[:, 1], reset_axis)
    if reset_ref is None or current_ref is None:
        return np.asarray(reset_rotvec, dtype=np.float64).copy()

    heading_delta = np.arctan2(
        np.dot(reset_axis, np.cross(reset_ref, current_ref)),
        np.dot(reset_ref, current_ref),
    )
    target_rot = rotvec_to_matrix(reset_axis * heading_delta) @ reset_rot
    return matrix_to_rotvec(target_rot)



class KeyboardTeleop:
    def __init__(self) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((520, 160))
        pygame.display.set_caption("UR5 RG2 keyboard collection")
        self.gripper_closed = False
        self.screw_direction = 0
        self._paint((80, 80, 80))

    def _paint(self, color: tuple[int, int, int]) -> None:
        self.screen.fill(color)
        self.pygame.display.flip()

    def close(self) -> None:
        self.pygame.quit()

    def poll(self, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, bool, str | None]:
        self.pygame.event.pump()
        event = None
        for pygame_event in self.pygame.event.get():
            if pygame_event.type != self.pygame.KEYDOWN:
                continue
            if pygame_event.key == self.pygame.K_z:
                self.screw_direction = 0
                event = "reset"
            elif pygame_event.key == self.pygame.K_RETURN:
                self.screw_direction = 0
                event = "done"
            elif pygame_event.key == self.pygame.K_ESCAPE:
                self.screw_direction = 0
                event = "quit"
            elif pygame_event.key == self.pygame.K_SPACE:
                self.gripper_closed = not self.gripper_closed
            elif pygame_event.key == self.pygame.K_p:
                event = "start"
            elif pygame_event.key == self.pygame.K_l:
                event = "random_reset_pose"
            elif pygame_event.key == self.pygame.K_o:
                event = "capture_reset_orientation"
            elif pygame_event.key == self.pygame.K_m:
                event = "reset_tcp_orientation"
            elif pygame_event.key == self.pygame.K_t:
                self.screw_direction = 0 if self.screw_direction == 1 else 1
                print("Screw down enabled" if self.screw_direction else "Screw motion stopped")
            elif pygame_event.key == self.pygame.K_g:
                self.screw_direction = 0 if self.screw_direction == -1 else -1
                print("Screw up enabled" if self.screw_direction else "Screw motion stopped")

        keys = self.pygame.key.get_pressed()
        move_step = args.move_step
        shift_pressed = keys[self.pygame.K_LSHIFT] or keys[self.pygame.K_RSHIFT]
        speed_scale = args.shift_speed_scale if shift_pressed else 1.0
        move_step = args.move_step * speed_scale

        dpos = np.zeros(3, dtype=np.float64)
        drot = np.eye(3)

        if keys[self.pygame.K_s]:
            dpos += np.array([move_step, 0.0, 0.0])
        if keys[self.pygame.K_w]:
            dpos += np.array([-move_step, 0.0, 0.0])
        if keys[self.pygame.K_a]:
            dpos += np.array([0.0, -move_step, 0.0])
        if keys[self.pygame.K_d]:
            dpos += np.array([0.0, move_step, 0.0])
        if keys[self.pygame.K_r]:
            dpos += np.array([0.0, 0.0, move_step])
        if keys[self.pygame.K_f]:
            dpos += np.array([0.0, 0.0, -move_step])

        if keys[self.pygame.K_LEFT]:
            drot = rotation_matrix(args.rot_step, (0.0, 1.0, 0.0))
        if keys[self.pygame.K_RIGHT]:
            drot = rotation_matrix(-args.rot_step, (0.0, 1.0, 0.0))
        if keys[self.pygame.K_DOWN]:
            drot = rotation_matrix(args.rot_step, (1.0, 0.0, 0.0))
        if keys[self.pygame.K_UP]:
            drot = rotation_matrix(-args.rot_step, (1.0, 0.0, 0.0))
        if keys[self.pygame.K_q]:
            drot = rotation_matrix(args.rot_step, (0.0, 0.0, 1.0))
        if keys[self.pygame.K_e]:
            drot = rotation_matrix(-args.rot_step, (0.0, 0.0, 1.0))

        screw_rot_step = args.screw_rot_step
        if args.screw_rpm is not None:
            screw_rot_step = 2.0 * np.pi * float(args.screw_rpm) / (60.0 * args.fps)
        screw_linear_step = args.screw_pitch * screw_rot_step / (2.0 * np.pi)
        screw_direction = self.screw_direction
        if keys[self.pygame.K_y]:
            screw_direction = 1
        elif keys[self.pygame.K_h]:
            screw_direction = -1
        if screw_direction == 1:
            dpos += np.array([0.0, 0.0, -screw_linear_step])
            drot = rotation_matrix(screw_rot_step, (0.0, 0.0, 1.0))
        elif screw_direction == -1:
            dpos += np.array([0.0, 0.0, screw_linear_step])
            drot = rotation_matrix(-screw_rot_step, (0.0, 0.0, 1.0))

        has_motion = bool(np.any(dpos) or not np.allclose(drot, np.eye(3)))
        self._paint((0, 140, 0) if has_motion else (80, 80, 80))
        gripper = 1.0 if self.gripper_closed else 0.0
        return dpos, drot, gripper, has_motion, event



def apply_tcp_delta(tcp_pose: np.ndarray, dpos: np.ndarray, drot: np.ndarray) -> np.ndarray:
    target = np.asarray(tcp_pose, dtype=np.float64).copy()
    target[:3] += dpos
    current_rot = rotvec_to_matrix(target[3:6])
    target[3:6] = matrix_to_rotvec(current_rot @ drot)
    return target


def sample_random_reset_tcp(reset_rotation_vector: np.ndarray) -> np.ndarray:
    target = np.zeros(6, dtype=np.float64)
    target[:3] = RESET_TCP_POSITION + np.random.uniform(
        -RESET_TCP_POSITION_RANDOM_RANGE_M,
        RESET_TCP_POSITION_RANDOM_RANGE_M,
        size=3,
    )
    target[3:6] = np.asarray(reset_rotation_vector, dtype=np.float64)
    return target


def step_toward_joint_state(
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    max_joint_step_rad: float,
) -> tuple[np.ndarray, bool]:
    current = np.asarray(current_joints, dtype=np.float64).copy()
    target = np.asarray(target_joints, dtype=np.float64).reshape(-1)
    next_joints = current.copy()

    arm_delta = target[:6] - current[:6]
    max_delta = float(np.max(np.abs(arm_delta)))
    if max_delta > max_joint_step_rad > 0.0:
        next_joints[:6] = current[:6] + arm_delta / max_delta * max_joint_step_rad
    else:
        next_joints[:6] = target[:6]

    reached = np.max(np.abs(target[:6] - next_joints[:6])) < 1e-3
    return next_joints, reached


class PreviewWindow:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._cv2 = None
        self._warned = False
        if self.enabled:
            import cv2

            self._cv2 = cv2

    def show(self, base_image: np.ndarray, wrist_image: np.ndarray) -> None:
        if not self.enabled or self._cv2 is None:
            return

        preview = np.concatenate([base_image, wrist_image], axis=1)
        try:
            self._cv2.imshow("base | wrist", preview[:, :, ::-1])
            self._cv2.waitKey(1)
        except self._cv2.error as exc:
            self.enabled = False
            if not self._warned:
                self._warned = True
                print(
                    "OpenCV preview is unavailable in this environment; "
                    "continuing collection without preview. "
                    "You can also pass --no-preview."
                )
                print(exc)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    robot = ZMQClientRobot(
        port=args.robot_port,
        host=args.host,
        timeout_ms=args.robot_timeout_ms,
    )
    base_camera = ZMQClientCamera(port=args.base_camera_port, host=args.host)
    wrist_camera = ZMQClientCamera(port=args.wrist_camera_port, host=args.host)
    force_reader = None
    left_tactile = None
    right_tactile = None
    if args.collect_tactile:
        left_tactile = ZMQClientTactile(
            port=args.left_tactile_port,
            host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )
        right_tactile = ZMQClientTactile(
            port=args.right_tactile_port,
            host=args.host,
            timeout_ms=args.tactile_timeout_ms,
        )
        print(
            "Using Xense tactile through ZMQ. Start ur5_rg2/launch.py with "
            "--tactile-source xense and the two Xense serial numbers."
        )

    try:
        if args.collect_force:
            force_reader = ForceReader(args, robot)

        robot_dofs = robot.num_dofs()
        if robot_dofs != 7:
            raise RuntimeError(f"Expected UR5 + RG2 robot with 7 DoF, got {robot_dofs}.")
        base_camera.read((args.image_size, args.image_size))
        wrist_camera.read((args.image_size, args.image_size))
        if left_tactile is not None and right_tactile is not None:
            format_tactile_frame_from_args(left_tactile.read(), args)
            format_tactile_frame_from_args(right_tactile.read(), args)
        if force_reader is not None:
            force_reader.read()
    except (ModuleNotFoundError, RuntimeError) as exc:
        if force_reader is not None:
            force_reader.close()
        robot.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()
        raise RuntimeError(
            "Failed to connect to or warm up the robot/camera/tactile/force sources. "
            "Start them first with:\n"
            "  python ur5_rg2/launch.py --robot ur "
            "--camera-width 424 --camera-height 240 --camera-fps 15 "
            "--tactile-source xense --xense-left-serial OG001783 "
            "--xense-right-serial OG001782\n"
            "If the servers are already running, check the camera serial numbers "
            "and RealSense USB/frame status. If tactile collection is enabled, "
            "also check xensesdk, USB permissions, Xense ZMQ ports, and the "
            "Xense serial numbers. "
            "If force collection is enabled with --force-source modbus-serial, "
            "check the FTS-300-S USB serial port, slave address, register address, "
            "baudrate, and minimalmodbus/pyserial installation. If --force-source "
            "modbus-tcp is used, check IP, port, unit id, register address, "
            "function code, data format, and scale. If --force-source ur-rtde "
            "is used, check that the UR TCP force signal is available through RTDE.\n"
            f"Original error: {exc}"
        ) from exc

    dataset = make_dataset(args)
    store_tactile, store_force = validate_requested_sensor_features(dataset, args)
    if store_tactile:
        print(
            "Xense tactile will be saved as "
            "observation.tactile_{left,right}.force/force6d/depth/rectify "
            f"with force_shape=({args.xense_force_height}, {args.xense_force_width}, 3), "
            f"depth_shape=({args.xense_depth_height}, {args.xense_depth_width}, 3), "
            f"rectify_shape=({args.xense_rectify_height}, {args.xense_rectify_width}, 3), "
            f"left_zmq={args.host}:{args.left_tactile_port}, "
            f"right_zmq={args.host}:{args.right_tactile_port}"
        )
    if store_force:
        print(f"Force/torque will be saved as {FORCE_TORQUE_KEY}")
    period = 1.0 / args.fps
    force_bias = np.zeros(6, dtype=np.float32)
    if store_force and force_reader is not None:
        force_bias = calibrate_force_bias(force_reader, args.fps)
    force_safety = None
    if args.force_safety and store_force and force_reader is not None:
        force_safety = CartesianForceSafetyController(args, dt=period)
        print(
            "Force safety enabled: "
            f"force_threshold={force_safety.threshold[:3].tolist()} N, "
            f"torque_threshold={force_safety.threshold[3:].tolist()} Nm, "
            f"force_hard_stop={force_safety.hard_stop[:3].tolist()} N, "
            f"torque_hard_stop={force_safety.hard_stop[3:].tolist()} Nm, "
            f"frame={force_safety.force_frame}, "
            f"axis_preset={force_safety.axis_preset}, "
            f"axis_order={force_safety.axis_order.tolist()}, "
            f"axis_signs={force_safety.axis_signs.tolist()}, "
            f"max_xyz_correction={force_safety.max_correction[:3].tolist()} m, "
            f"max_rot_correction={force_safety.max_correction[3:].tolist()} rad"
        )
    teleop = KeyboardTeleop()
    preview = PreviewWindow(enabled=not args.no_preview)

    obj_init = np.zeros(9, dtype=np.float32)
    episode_id = 0
    recording = False
    frames_in_episode = 0
    force_print_frame = 0
    next_tick = time.time()
    reset_target_joints = None
    reset_rotation_vector = RESET_TCP_ROTATION_VECTOR.copy()
    last_tactile_invalid_print = 0.0
    last_tactile_safety_print = 0.0
    last_tactile_wrench_print = 0.0

    print("Keyboard teleop controls:")
    print("  W/S/A/D/R/F: translate TCP")
    print("  Arrow keys + Q/E: rotate TCP")
    print("  T/G: toggle screw down/up; press the active key again to stop")
    print("  Y/H: hold screw down/up; release to stop")
    print("  P: start recording")
    print("  L: move to randomized reset pose through IK")
    print("  O: capture current TCP orientation for L reset")
    print("  M: set TCP x/y to 0.0378/-0.5468, keep z, set vertical reset orientation")
    print("  SPACE: toggle RG2, Z: clear current episode, ENTER: save episode, ESC: exit")

    try:
        while episode_id < args.num_episodes:
            now = time.time()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue
            next_tick = now + period

            tcp_pose = robot.get_tcp_pose()
            # print(
            #     "TCP position "
            #     f"[x,y,z]=[{tcp_pose[0]: .4f}, {tcp_pose[1]: .4f}, {tcp_pose[2]: .4f}]"
            # )
            obs_before = robot.get_observations()
            current_joints = np.asarray(obs_before["joint_positions"], dtype=np.float32)

            tactile_read_error = None
            if not store_tactile or left_tactile is None:
                left_tactile_frame = empty_tactile_frame_from_args(args)
                right_tactile_frame = empty_tactile_frame_from_args(args)
            else:
                try:
                    left_tactile_frame = format_tactile_frame_from_args(
                        left_tactile.read(),
                        args,
                    )
                    if right_tactile is None:
                        right_tactile_frame = empty_tactile_frame_from_args(args)
                    else:
                        right_tactile_frame = format_tactile_frame_from_args(
                            right_tactile.read(),
                            args,
                        )
                except RuntimeError as exc:
                    tactile_read_error = exc
                    left_tactile_frame = empty_tactile_frame_from_args(args)
                    right_tactile_frame = empty_tactile_frame_from_args(args)

            left_tactile_valid = bool(left_tactile_frame["valid"])
            right_tactile_valid = bool(right_tactile_frame["valid"])
            left_wrench = format_force_torque(left_tactile_frame["force6d"])
            right_wrench = format_force_torque(right_tactile_frame["force6d"])

            tactile_safety_active = bool(
                args.collect_tactile
                and store_tactile
                and (
                    tactile_read_error is not None
                    or left_tactile is None
                    or right_tactile is None
                    or not left_tactile_valid
                    or not right_tactile_valid
                    or not tactile_wrench_is_safe(left_tactile_frame)
                    or not tactile_wrench_is_safe(right_tactile_frame)
                )
            )

            # Print both tactile wrenches once per second, even when they are safe.
            # This makes it clear whether the right tactile node is actually updating.
            now = time.time()
            if (
                args.collect_tactile
                and store_tactile
                and now - last_tactile_wrench_print > 1.0
            ):
                # print(
                #     "Tactile wrench "
                #     f"left_valid={left_tactile_valid} "
                #     f"left={np.array2string(left_wrench, precision=3)}; "
                #     f"right_valid={right_tactile_valid} "
                #     f"right={np.array2string(right_wrench, precision=3)}"
                # )
                print(
                        "Tactile : "
                        f"left={np.array2string(left_wrench, precision=1)}, "
                        f"right={np.array2string(right_wrench, precision=2)}"
                    )
                # print(tcp_pose)

                last_tactile_wrench_print = now

            dpos, drot, gripper, has_motion, event = teleop.poll(args)

            if tactile_safety_active:
                now = time.time()
                if now - last_tactile_safety_print > 1.0:
                    reasons = []
                    if tactile_read_error is not None:
                        reasons.append(f"read_error={tactile_read_error}")
                    if not left_tactile_valid:
                        reasons.append("left_valid=False")
                    if not right_tactile_valid:
                        reasons.append("right_valid=False")
                    if left_tactile_valid and not tactile_wrench_is_safe(left_tactile_frame):
                        reasons.append("left_wrench_over_limit")
                    if right_tactile_valid and not tactile_wrench_is_safe(right_tactile_frame):
                        reasons.append("right_wrench_over_limit")
                    print(
                        "Tactile safety triggered: keyboard input disabled and gripper opened. "
                        f"reason={','.join(reasons) or 'xense_source_unavailable'}; "
                        f"left={np.array2string(left_wrench, precision=3)}, "
                        f"right={np.array2string(right_wrench, precision=3)}"
                    )
                    last_tactile_safety_print = now

                teleop.gripper_closed = False
                teleop.screw_direction = 0
                reset_target_joints = None
                dpos = np.zeros(3, dtype=np.float64)
                drot = np.eye(3)
                gripper = 0.0
                has_motion = False
                if event != "quit":
                    event = None
                if not args.no_command:
                    robot.command_gripper(0.0, force=args.gripper_force)

            if event == "reset":
                reset_target_joints = None
                reset_episode_buffer(dataset)
                if force_safety is not None:
                    force_safety.reset()
                recording = False
                frames_in_episode = 0
                print("Reset current episode buffer")
                continue
            if event == "done":
                reset_target_joints = None
                if recording and frames_in_episode > 0:
                    dataset.save_episode()
                    print(f"Saved episode {episode_id} ({frames_in_episode} frames)")
                    episode_id += 1
                if force_safety is not None:
                    force_safety.reset()
                recording = False
                frames_in_episode = 0
                continue
            # Start recording immediately when P is pressed, instead of waiting for motion. This allows collecting "zero" episodes with no motion, which can be useful for certain applications.
            if event == "start":
                if not recording:
                    recording = True
                    frames_in_episode = 0
                    reset_episode_buffer(dataset)
                    if force_safety is not None:
                        force_safety.reset()
                    print(f"Start recording episode {episode_id}")
            if event == "quit":
                reset_target_joints = None
                if recording and frames_in_episode > 0:
                    dataset.save_episode()
                    print(f"Saved episode {episode_id} before exit ({frames_in_episode} frames)")
                break

            if event == "capture_reset_orientation":
                reset_rotation_vector = np.asarray(tcp_pose[3:6], dtype=np.float64).copy()
                print(
                    "Captured reset TCP orientation "
                    f"[rx,ry,rz]=[{reset_rotation_vector[0]: .6f}, "
                    f"{reset_rotation_vector[1]: .6f}, {reset_rotation_vector[2]: .6f}]"
                )
                continue

            if event == "random_reset_pose":
                reset_tcp = sample_random_reset_tcp(reset_rotation_vector)
                try:
                    reset_target_joints = np.asarray(
                        robot.inverse_kinematics(reset_tcp, qnear=current_joints[:6]),
                        dtype=np.float64,
                    )
                    reset_target_joints[5] -= 0.5 * np.pi
                except RuntimeError as exc:
                    print(
                        "Reset IK failed. Restart ur5_rg2/launch.py so the robot "
                        f"server exposes inverse_kinematics. Error: {exc}"
                    )
                    continue
                if force_safety is not None:
                    force_safety.reset()
                print(
                    "Move to randomized reset pose through IK "
                    f"tcp=[{reset_tcp[0]: .4f}, {reset_tcp[1]: .4f}, "
                    f"{reset_tcp[2]: .4f}, {reset_tcp[3]: .3f}, "
                    f"{reset_tcp[4]: .3f}, {reset_tcp[5]: .3f}] "
                    f"q={np.array2string(reset_target_joints, precision=4)}"
                )

            reset_orientation_requested = event == "reset_tcp_orientation"
            reset_orientation_rotvec = None
            if reset_orientation_requested:
                reset_target_joints = None
                if force_safety is not None:
                    force_safety.reset()
                reset_orientation_rotvec = vertical_reset_rotvec_preserving_heading(
                    tcp_pose[3:6],
                    RESET_TCP_ROTATION_VECTOR,
                )
                print(
                    "Set TCP x/y with current z, vertical RESET orientation, "
                    f"and current heading [rx,ry,rz]=[{reset_orientation_rotvec[0]: .6f}, "
                    f"{reset_orientation_rotvec[1]: .6f}, "
                    f"{reset_orientation_rotvec[2]: .6f}]"
                )

            reset_pose_requested = reset_target_joints is not None
            if reset_pose_requested:
                target_joints, reset_reached = step_toward_joint_state(
                    current_joints,
                    reset_target_joints,
                    abs(float(args.reset_max_joint_step_rad)),
                )
                target_joints[-1] = gripper
                if reset_reached:
                    reset_target_joints = None
                    actual_tcp = robot.get_tcp_pose()
                    print(
                        "Reached randomized reset IK joint target; actual TCP "
                        f"[x,y,z,rx,ry,rz]=[{actual_tcp[0]: .4f}, {actual_tcp[1]: .4f}, "
                        f"{actual_tcp[2]: .4f}, {actual_tcp[3]: .3f}, "
                        f"{actual_tcp[4]: .3f}, {actual_tcp[5]: .3f}]"
                    )
            else:
                if reset_orientation_requested:
                    target_tcp = np.asarray(tcp_pose, dtype=np.float64).copy()
                    target_tcp[:2] = RESET_ORIENTATION_TCP_XY
                    target_tcp[3:6] = reset_orientation_rotvec
                    has_motion = True
                else:
                    target_tcp = apply_tcp_delta(tcp_pose, dpos, drot)

            force_torque = np.zeros(6, dtype=np.float32)
            if store_force and force_reader is not None:
                try:
                    raw_force_torque = force_reader.read()
                    force_torque = (raw_force_torque - force_bias).astype(np.float32)
                except RuntimeError as exc:
                    print(f"Force/torque frame skipped: {exc}")
                    continue
                print_every = max(1, int(args.force_print_every))
                if force_print_frame % print_every == 0:
                    print_force_torque(force_torque)
                force_print_frame += 1

            safety_active = False
            safety_hard_stop = False
            safety_correction = np.zeros(6, dtype=np.float64)
            if force_safety is not None and not reset_pose_requested:
                target_tcp, safety_hard_stop, safety_correction, signed_wrench = (
                    force_safety.correct_target(tcp_pose, target_tcp, force_torque)
                )
                safety_active = bool(np.any(np.abs(safety_correction) > 1e-6))
                if safety_active or safety_hard_stop:
                    print(
                        "Force safety "
                        f"{'HARD ' if safety_hard_stop else ''}"
                        f"force_xyz={np.array2string(signed_wrench[:3], precision=3)} "
                        f"torque_xyz={np.array2string(signed_wrench[3:], precision=3)} "
                        f"pos_corr_m={np.array2string(safety_correction[:3], precision=5)} "
                        f"rot_corr_rad={np.array2string(safety_correction[3:], precision=5)}"
                    )

            if (
                reset_pose_requested
                or has_motion
                or safety_active
                or abs(float(current_joints[-1]) - gripper) > 0.03
            ):
                if not args.no_command:
                    if reset_pose_requested:
                        robot.command_joint_state(target_joints)
                    elif has_motion or safety_active:
                        robot.command_tcp_pose(target_tcp)
                    if not safety_hard_stop:
                        robot.command_gripper(gripper, force=args.gripper_force)

            if not recording and not preview.enabled:
                continue

            try:
                base_image, _ = base_camera.read((args.image_size, args.image_size))
                wrist_image, _ = wrist_camera.read((args.image_size, args.image_size))
            except RuntimeError as exc:
                print(f"Camera frame skipped: {exc}")
                continue
            base_image = resize_rgb(base_image, args.image_size)
            wrist_image = resize_rgb(wrist_image, args.image_size)

            if args.collect_tactile and store_tactile and tactile_read_error is not None:
                print(f"Tactile frame skipped: {tactile_read_error}")
                continue
            if args.collect_tactile and store_tactile and args.require_valid_tactile and (
                not bool(left_tactile_frame["valid"])
                or not bool(right_tactile_frame["valid"])
            ):
                now = time.time()
                if now - last_tactile_invalid_print > 1.0:
                    invalid_sides = []
                    if not bool(left_tactile_frame["valid"]):
                        invalid_sides.append("left")
                    if not bool(right_tactile_frame["valid"]):
                        invalid_sides.append("right")
                    print(
                        "Tactile frame skipped: "
                        f"{'+'.join(invalid_sides)} tactile source returned valid=False. "
                        "Check ur5_rg2/launch.py --tactile-source xense, Xense ZMQ ports, "
                        "USB permissions, and both Xense serial numbers before recording."
                    )
                    last_tactile_invalid_print = now
                continue

            obs_after = robot.get_observations()
            state = np.asarray(obs_after["joint_positions"], dtype=np.float32)
            action = state.copy()
            action[-1] = float(state[-1]) if safety_hard_stop else gripper

            # Recording now starts only when P is pressed.

            # if not recording and (
            #     has_motion
            #     or safety_active
            #     or abs(float(current_joints[-1]) - gripper) > 0.03
            # ):
            #     recording = True
            #     frames_in_episode = 0
            #     reset_episode_buffer(dataset)
            #     print(f"Start recording episode {episode_id}")

            if recording:
                frame = {
                    "observation.image": base_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": state[:6].astype(np.float32),
                    "action": action.astype(np.float32),
                    "obj_init": obj_init,
                }
                if store_tactile:
                    frame.update(
                        {
                            "observation.tactile_left.force": left_tactile_frame["force"],
                            "observation.tactile_left.force6d": left_tactile_frame[
                                "force6d"
                            ],
                            "observation.tactile_left.depth": left_tactile_frame[
                                "depth"
                            ],
                            "observation.tactile_left.rectify": left_tactile_frame[
                                "rectify"
                            ],
                            "observation.tactile_right.force": right_tactile_frame["force"],
                            "observation.tactile_right.force6d": right_tactile_frame[
                                "force6d"
                            ],
                            "observation.tactile_right.depth": right_tactile_frame[
                                "depth"
                            ],
                            "observation.tactile_right.rectify": right_tactile_frame[
                                "rectify"
                            ],
                            "observation.tactile.timestamp": np.array(
                                [
                                    left_tactile_frame["timestamp"],
                                    right_tactile_frame["timestamp"],
                                ],
                                dtype=np.float64,
                            ),
                            "observation.tactile.host_timestamp": np.array(
                                [
                                    left_tactile_frame["host_timestamp"],
                                    right_tactile_frame["host_timestamp"],
                                ],
                                dtype=np.float64,
                            ),
                        }
                    )
                if store_force:
                    frame[FORCE_TORQUE_KEY] = force_torque

                dataset.add_frame(frame, task=args.task)
                frames_in_episode += 1

            preview.show(base_image, wrist_image)
    finally:
        preview.close()
        teleop.close()
        if force_reader is not None:
            force_reader.close()
        robot.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()


if __name__ == "__main__":
    main()
