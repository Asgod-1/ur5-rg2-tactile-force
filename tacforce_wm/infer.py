"""Run a TacForceWM policy with UR5/RG2 hardware and force/tactile safety."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.experiment import add_experiment_arguments, policy_checkpoint_candidates  # noqa: E402


DEFAULT_POLICY_CHECKPOINTS = (
    TRAINING_ROOT / "outputs/tactile_insert_policy/checkpoints/best.pt",
    TRAINING_ROOT / "outputs/tactile_insert_policy/checkpoints/latest.pt",
)
LOCAL_DINO_CHECKPOINT = TRAINING_ROOT / "DINOv2/model.safetensors"
FORCE_TARE_SECONDS = 2.0
DEFAULT_FORCE_SAFETY_THRESHOLD_N = 5.0
DEFAULT_FORCE_SAFETY_HARD_STOP_N = 18.0
DEFAULT_TORQUE_SAFETY_THRESHOLD_NM = 0.5
DEFAULT_TORQUE_SAFETY_HARD_STOP_NM = 1.5
DEFAULT_FORCE_TORQUE_TOOL_OFFSET_M = 0.042 - 0.0034


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TacForceWM policy on the UR5 + RG2 ZMQ nodes."
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--robot-timeout-ms", type=int, default=8000)
    parser.add_argument("--wrist-camera-port", type=int, default=5000)
    parser.add_argument("--base-camera-port", type=int, default=5001)
    parser.add_argument("--camera-timeout-ms", type=int, default=20000)
    parser.add_argument(
        "--collect-tactile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read both Xense sensors (required by TacForceWM).",
    )
    parser.add_argument("--left-tactile-port", type=int, default=5100)
    parser.add_argument("--right-tactile-port", type=int, default=5101)
    parser.add_argument("--tactile-timeout-ms", type=int, default=15000)
    parser.add_argument(
        "--no-tactile",
        action="store_false",
        dest="collect_tactile",
        help="Alias for --no-collect-tactile.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Ctrl-C.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without commanding the robot.")
    parser.add_argument("--no-command", action="store_true", help="Alias for --dry-run.")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--action-lowpass-alpha", type=float, default=1.0)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.02,
        help="Maximum absolute joint change per inference step in radians.",
    )
    parser.add_argument(
        "--max-startup-joint-delta",
        type=float,
        default=0.02,
        help="Conservative joint delta used for the first startup steps.",
    )
    parser.add_argument("--startup-steps", type=int, default=50)
    parser.add_argument("--gripper-force", type=float, default=4.0)
    parser.add_argument(
        "--command-gripper-separately",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also send the 7th action through command_gripper after command_joint_state.",
    )
    parser.add_argument(
        "--collect-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read force/torque input (required by TacForceWM).",
    )
    parser.add_argument(
        "--force-source",
        choices=("modbus-serial", "modbus-tcp", "ur-rtde"),
        default="modbus-serial",
    )
    parser.add_argument("--force-serial-port", default=None)
    parser.add_argument("--force-serial-slave-address", type=int, default=9)
    parser.add_argument("--force-serial-register-address", type=int, default=180)
    parser.add_argument("--force-serial-function-code", type=int, default=3)
    parser.add_argument("--force-serial-baudrate", type=int, default=19200)
    parser.add_argument("--force-serial-timeout-s", type=float, default=0.1)
    parser.add_argument("--force-serial-retries", type=int, default=5)
    parser.add_argument("--force-serial-wakeup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-modbus-host", default="192.168.1.1")
    parser.add_argument("--force-modbus-port", type=int, default=502)
    parser.add_argument("--force-modbus-unit-id", type=int, default=9)
    parser.add_argument("--force-modbus-address", type=int, default=0)
    parser.add_argument("--force-modbus-function", choices=("holding", "input"), default="input")
    parser.add_argument("--force-modbus-format", choices=("int16", "int32", "float32"), default="int32")
    parser.add_argument("--force-modbus-word-order", choices=("big", "little"), default="big")
    parser.add_argument("--force-modbus-scale", type=float, default=1.0)
    parser.add_argument("--force-modbus-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--force-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Cartesian admittance force correction during inference.",
    )
    parser.add_argument(
        "--force-safety-threshold-n",
        default=f"{DEFAULT_FORCE_SAFETY_THRESHOLD_N},{DEFAULT_FORCE_SAFETY_THRESHOLD_N},{DEFAULT_FORCE_SAFETY_THRESHOLD_N}",
        help="Per-axis x,y,z force threshold in N before correction starts.",
    )
    parser.add_argument(
        "--force-safety-hard-stop-n",
        default=f"{DEFAULT_FORCE_SAFETY_HARD_STOP_N},{DEFAULT_FORCE_SAFETY_HARD_STOP_N},{DEFAULT_FORCE_SAFETY_HARD_STOP_N}",
        help="Per-axis x,y,z hard force limit in N.",
    )
    parser.add_argument(
        "--torque-safety-threshold-nm",
        default=f"{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM},{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM},{DEFAULT_TORQUE_SAFETY_THRESHOLD_NM}",
        help="Per-axis rx,ry,rz torque threshold in Nm before correction starts.",
    )
    parser.add_argument(
        "--torque-safety-hard-stop-nm",
        default=f"{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM},{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM},{DEFAULT_TORQUE_SAFETY_HARD_STOP_NM}",
        help="Per-axis rx,ry,rz hard torque limit in Nm.",
    )
    parser.add_argument("--force-torque-tool-offset-m", type=float, default=DEFAULT_FORCE_TORQUE_TOOL_OFFSET_M)
    parser.add_argument("--force-safety-axis-signs", default=None)
    parser.add_argument("--force-safety-axis-order", default=None)
    parser.add_argument(
        "--force-safety-axis-preset",
        choices=("auto", "identity", "fts300-tool"),
        default="auto",
    )
    parser.add_argument(
        "--force-safety-frame",
        choices=("auto", "tool", "base"),
        default="auto",
    )
    parser.add_argument("--force-safety-inv-mass", type=float, default=0.40)
    parser.add_argument("--force-safety-damping", type=float, default=18.0)
    parser.add_argument("--force-safety-stiffness", type=float, default=80.0)
    parser.add_argument("--torque-safety-inv-inertia", type=float, default=0.08)
    parser.add_argument("--torque-safety-damping", type=float, default=10.0)
    parser.add_argument("--torque-safety-stiffness", type=float, default=35.0)
    parser.add_argument("--force-safety-max-vel", type=float, default=0.025)
    parser.add_argument("--force-safety-max-correction-m", type=float, default=0.006)
    parser.add_argument("--torque-safety-max-angular-vel", type=float, default=0.10)
    parser.add_argument("--torque-safety-max-correction-rad", type=float, default=0.035)
    parser.add_argument("--force-safety-deadband-n", type=float, default=0.30)
    parser.add_argument("--torque-safety-deadband-nm", type=float, default=0.03)
    parser.add_argument(
        "--force-position-mixing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a Cartesian position correction toward the configured contact force.",
    )
    parser.add_argument(
        "--force-position-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Force axis used for force-position mixing.",
    )
    parser.add_argument(
        "--force-position-axis-sign",
        type=float,
        default=1.0,
        help="Sign applied to measured force on the selected axis.",
    )
    parser.add_argument(
        "--force-position-direction-sign",
        type=float,
        default=1.0,
        help="TCP direction for increasing contact force. -1 moves toward negative base axis.",
    )
    parser.add_argument(
        "--force-position-contact-threshold-n",
        type=float,
        default=1.5,
        help="Measured force magnitude threshold before force-position mixing becomes active.",
    )
    parser.add_argument(
        "--force-position-require-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require measured contact before applying force-position correction.",
    )
    parser.add_argument(
        "--force-position-manual-target-n",
        type=float,
        default=0.0,
        help="Desired contact force magnitude in N; zero disables force-position mixing.",
    )
    parser.add_argument(
        "--force-position-min-target-n",
        type=float,
        default=0.0,
        help="Minimum target contact force magnitude.",
    )
    parser.add_argument(
        "--force-position-max-target-n",
        type=float,
        default=8.0,
        help="Maximum target contact force magnitude.",
    )
    parser.add_argument(
        "--force-position-k-m-per-n",
        type=float,
        default=1.0e-3,
        help="Position correction gain in meters per Newton of force error.",
    )
    parser.add_argument(
        "--force-position-max-down-step-m",
        type=float,
        default=1.2e-3,
        help="Maximum per-step motion toward higher contact force.",
    )
    parser.add_argument(
        "--force-position-max-up-step-m",
        type=float,
        default=3.0e-4,
        help="Maximum per-step release motion when measured force is above target.",
    )
    parser.add_argument(
        "--force-position-lowpass-alpha",
        type=float,
        default=0.25,
        help="Low-pass alpha for the target force magnitude.",
    )
    parser.add_argument(
        "--force-position-use-target-force-magnitude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Limit and smooth the configured target force before force-position mixing. "
            "Disable to use the raw manual target."
        ),
    )
    parser.add_argument(
        "--hybrid-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use hybrid force/position velocity control instead of directly sending the policy joint target.",
    )
    parser.add_argument(
        "--hybrid-force-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Cartesian force-controlled translation axis.",
    )
    parser.add_argument(
        "--hybrid-torque-axis",
        choices=("rx", "ry", "rz", "none"),
        default="rz",
        help="Cartesian torque-controlled rotation axis.",
    )
    parser.add_argument(
        "--hybrid-require-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only apply hybrid force/torque correction after measured contact.",
    )
    parser.add_argument("--hybrid-contact-threshold-n", type=float, default=1.0)
    parser.add_argument(
        "--hybrid-force-direction-sign",
        type=float,
        default=-1.0,
        help="TCP/base direction that increases contact force on the selected force axis.",
    )
    parser.add_argument(
        "--hybrid-manual-target-force-n",
        type=float,
        default=5.0,
        help="Desired contact force magnitude for the selected force axis.",
    )
    parser.add_argument(
        "--hybrid-manual-target-torque-nm",
        type=float,
        default=0.0,
        help="Signed desired torque for the selected torque axis.",
    )
    parser.add_argument("--hybrid-min-target-force-n", type=float, default=0.0)
    parser.add_argument("--hybrid-max-target-force-n", type=float, default=12.0)
    parser.add_argument("--hybrid-max-target-torque-nm", type=float, default=1.5)
    parser.add_argument(
        "--hybrid-kp-force-vel",
        type=float,
        default=0.004,
        help="Force error to Cartesian linear velocity gain, m/s/N.",
    )
    parser.add_argument("--hybrid-ki-force-vel", type=float, default=0.0)
    parser.add_argument(
        "--hybrid-kp-torque-vel",
        type=float,
        default=0.20,
        help="Torque error to Cartesian angular velocity gain, rad/s/Nm.",
    )
    parser.add_argument("--hybrid-ki-torque-vel", type=float, default=0.0)
    parser.add_argument("--hybrid-max-linear-vel", type=float, default=0.015)
    parser.add_argument("--hybrid-max-angular-vel", type=float, default=0.12)
    parser.add_argument(
        "--hybrid-position-vel-gain",
        type=float,
        default=1.0,
        help="Gain from policy joint target error to joint velocity.",
    )
    parser.add_argument("--hybrid-max-joint-vel", type=float, default=0.35)
    parser.add_argument("--hybrid-integral-limit", type=float, default=4.0)
    parser.add_argument("--hybrid-lowpass-alpha", type=float, default=0.35)
    parser.add_argument(
        "--hybrid-use-target-force-magnitude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Limit the configured target force before hybrid control. "
            "Disable to use the raw manual target."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log measured force/torque, TCP pose and hybrid control to TensorBoard.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        default="runs/infer",
        help="TensorBoard log root. A timestamped run directory is created under this path.",
    )
    parser.add_argument(
        "--tensorboard-log-every",
        type=int,
        default=1,
        help="Write force TensorBoard scalars every N inference steps.",
    )
    parser.add_argument(
        "--tensorboard-flush-every",
        type=int,
        default=20,
        help="Flush TensorBoard events every N logged steps.",
    )
    add_experiment_arguments(parser, config_overrides=False)
    parser.add_argument("--output-root", default="outputs", help="Training output root for --scene.")
    parser.add_argument(
        "--policy-execute-steps",
        type=int,
        default=8,
        help="Execute this many consecutive actions from each predicted chunk before replanning.",
    )
    parser.add_argument(
        "--tactile-safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the gripper when Xense tactile force exceeds a limit.",
    )
    parser.add_argument("--tactile-soft-max-taxel-n", type=float, default=6.0)
    parser.add_argument("--tactile-hard-max-taxel-n", type=float, default=8.0)
    parser.add_argument("--tactile-soft-total-n", type=float, default=65.0)
    parser.add_argument("--tactile-hard-total-n", type=float, default=90.0)
    parser.add_argument("--tactile-soft-open-step", type=float, default=0.10)
    parser.add_argument("--tactile-hard-open-command", type=float, default=0.0)
    parser.add_argument("--tactile-trigger-frames", type=int, default=2)
    parser.add_argument("--tactile-latch-steps", type=int, default=20)
    return parser.parse_args(argv)


def _resolve_policy_checkpoint(value: str | None, *, scene: str | None = None,
                               suffix: str = "", output_root: str = "outputs") -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TacForceWM policy checkpoint not found: {path}")
        return path
    if not scene and (suffix or output_root != "outputs"):
        raise ValueError("--suffix and --output-root require --scene (or pass --policy-path).")
    candidates = (
        policy_checkpoint_candidates(scene, suffix, output_root=output_root, training_root=TRAINING_ROOT)
        if scene else DEFAULT_POLICY_CHECKPOINTS
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No TacForceWM policy checkpoint was found for the selected run. "
        "Select --scene/--suffix or pass --policy-path. "
        f"Tried:\n  {tried}"
    )


def _xense_force_grid(value: np.ndarray, *, side: str) -> np.ndarray:
    """Convert one native Xense force frame to the training-time [35,20,3] grid."""
    force = np.asarray(value, dtype=np.float32)
    if force.shape == (35, 20, 3):
        return np.ascontiguousarray(force)
    if force.shape == (700, 3) or force.size == 35 * 20 * 3:
        return np.ascontiguousarray(force.reshape(35, 20, 3))
    raise RuntimeError(
        f"Expected {side} Xense force with shape [35,20,3] or [700,3], got {force.shape}. "
        "The loaded TacForceWM checkpoint was trained on the native Xense force grid."
    )


def format_tactile_frame(frame: dict) -> dict:
    """Keep the native Xense grid for the model and every taxel for safety."""
    force_key = "force" if "force" in frame else "distributed_force"
    if force_key not in frame:
        raise RuntimeError(
            f"Xense tactile frame has no force field; available keys={sorted(frame)}"
        )
    force = _xense_force_grid(frame[force_key], side="live")
    return {
        "force": force,
        "distributed_force": force.reshape(-1, 3),
    }


def tactile_inputs_from_frames(left_frame: dict, right_frame: dict) -> dict[str, np.ndarray]:
    return {
        f"observation.tactile_{side}.{key}": np.asarray(frame[key], dtype=np.float32)
        for side, frame in (("left", left_frame), ("right", right_frame))
        for key in ("force", "distributed_force")
    }


def resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] == (image_size, image_size):
        return image.copy()
    return np.asarray(Image.fromarray(image).resize((image_size, image_size)), dtype=np.uint8)


def format_force_torque(force_torque: np.ndarray) -> np.ndarray:
    array = np.asarray(force_torque, dtype=np.float32).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return array[:6].astype(np.float32)


def _pack_tactile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _xense_force_grid(left, side="left")
    right = _xense_force_grid(right, side="right")
    tactile = np.concatenate([left, right], axis=-1)
    # This exactly matches tacforce_wm.datasets.tactile_datasets._pack_tactile
    # for the training dataset's native [35,20,3] Xense grids.
    return np.concatenate([tactile, tactile[-1:]], axis=0).astype(np.float32)


def _wrist_image_tensor(image: np.ndarray, image_size: int, device):
    import torch
    import torch.nn.functional as functional

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected wrist RGB image [H,W,3], got {array.shape}")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = functional.interpolate(
            tensor[None].float(),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )[0].round().clamp(0, 255).to(torch.uint8)
    elif tensor.dtype != torch.uint8:
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    # Policy convention: [B,T,V,C,H,W]. It was trained with T=1, V=1.
    return tensor[None, None, None].to(device, non_blocking=True)


class TacForceWMInfer:
    """Load a checkpoint, maintain observation history, and execute safe action chunks."""

    def __init__(self, args) -> None:
        import torch

        from trainers.policy_trainer import build_policy
        from utils.normalizer import MultiFieldNormalizer

        if not args.collect_tactile:
            raise ValueError("TacForceWM inference requires both Xense sensors; remove --no-tactile.")
        if not args.collect_force:
            raise ValueError("TacForceWM inference requires force/torque input; use --collect-force.")
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_path = _resolve_policy_checkpoint(
            args.policy_path,
            scene=getattr(args, "scene", None),
            suffix=getattr(args, "suffix", ""),
            output_root=getattr(args, "output_root", "outputs"),
        )
        checkpoint = torch.load(self.policy_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Invalid policy checkpoint object in {self.policy_path}")
        for key in ("config", "policy_state_dict", "normalizer_state_dict", "dynamics_bundle"):
            if key not in checkpoint:
                raise KeyError(f"Policy checkpoint is missing {key!r}: {self.policy_path}")

        self.cfg = checkpoint["config"]
        policy_cfg = self.cfg["model"]["policy"]
        data_cfg = self.cfg["data"]
        if LOCAL_DINO_CHECKPOINT.is_file():
            policy_cfg["dino_checkpoint_path"] = str(LOCAL_DINO_CHECKPOINT)
        elif policy_cfg.get("image_pretrained", True):
            raise FileNotFoundError(
                f"Offline DINOv2 checkpoint not found: {LOCAL_DINO_CHECKPOINT}"
            )

        self.policy = build_policy(
            self.cfg,
            self.device,
            train_dataset=None,
            dynamics_bundle=checkpoint["dynamics_bundle"],
        )
        incompatible = self.policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        bad_missing = [key for key in incompatible.missing_keys if not key.startswith("dynamics.")]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Policy state mismatch: "
                f"missing={bad_missing}, unexpected={incompatible.unexpected_keys}"
            )
        normalizer = MultiFieldNormalizer()
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
        self.policy.set_normalizer(normalizer)
        self.policy.normalizer.to(self.device)
        self.policy.eval()

        self.window_size = int(data_cfg.get("window_size", 16))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.num_inference_steps = 16
        self.execute_steps = int(args.policy_execute_steps)
        if not 1 <= self.execute_steps <= self.policy.n_action_steps:
            raise ValueError(
                f"--policy-execute-steps must be in [1, {self.policy.n_action_steps}], "
                f"got {self.execute_steps}"
            )
        self._action_queue: deque[np.ndarray] = deque()
        self.force_history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.state_history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.tactile_history: deque[np.ndarray] = deque(maxlen=self.window_size)

        self.tactile_safety_enabled = bool(args.tactile_safety)
        self.tactile_soft_max_taxel = float(args.tactile_soft_max_taxel_n)
        self.tactile_hard_max_taxel = float(args.tactile_hard_max_taxel_n)
        self.tactile_soft_total = float(args.tactile_soft_total_n)
        self.tactile_hard_total = float(args.tactile_hard_total_n)
        self.tactile_soft_open_step = max(0.0, float(args.tactile_soft_open_step))
        self.tactile_hard_open_command = float(np.clip(args.tactile_hard_open_command, 0.0, 1.0))
        self.tactile_trigger_frames = max(1, int(args.tactile_trigger_frames))
        self.tactile_latch_steps = max(1, int(args.tactile_latch_steps))
        self._tactile_soft_count = 0
        self._tactile_latch_remaining = 0
        self._tactile_safety_level = "off"
        self._forced_gripper_command: float | None = None
        self._last_safe_gripper_command = 0.0

        print(f"Loaded TacForceWM policy: {self.policy_path}")
        print(
            f"device={self.device}, history={self.window_size}, image=wrist-only, "
            f"action_horizon={self.policy.action_horizon}, execute_steps={self.execute_steps}"
        )
        if self.tactile_safety_enabled:
            print(
                "Xense gripper safety enabled: "
                f"soft(max_taxel={self.tactile_soft_max_taxel:.3f}, total={self.tactile_soft_total:.3f}), "
                f"hard(max_taxel={self.tactile_hard_max_taxel:.3f}, total={self.tactile_hard_total:.3f}), "
                f"trigger_frames={self.tactile_trigger_frames}, latch_steps={self.tactile_latch_steps}"
            )

    @staticmethod
    def _append_and_pad(history: deque[np.ndarray], value: np.ndarray, size: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).copy()
        history.append(value)
        frames = list(history)
        if len(frames) < size:
            frames = [frames[0]] * (size - len(frames)) + frames
        return np.stack(frames, axis=0).astype(np.float32)

    def _build_obs(
        self,
        wrist_image: np.ndarray,
        joint_positions: np.ndarray,
        force_torque: np.ndarray,
        tactile_inputs: dict[str, np.ndarray] | None,
    ) -> dict:
        import torch

        tactile_inputs = tactile_inputs or {}
        left = tactile_inputs.get("observation.tactile_left.force")
        right = tactile_inputs.get("observation.tactile_right.force")
        if left is None or right is None:
            raise KeyError("Both left and right Xense force fields are required.")
        tactile_now = _pack_tactile(left, right)
        state_now = np.asarray(joint_positions, dtype=np.float32).reshape(-1)[:6]
        if state_now.size != 6:
            raise ValueError(f"Expected six UR5 joints, got {state_now.size}")
        force_now = format_force_torque(force_torque)

        tactile = self._append_and_pad(self.tactile_history, tactile_now, self.window_size)
        state = self._append_and_pad(self.state_history, state_now, self.window_size)
        force = self._append_and_pad(self.force_history, force_now, self.window_size)
        force_4x = np.repeat(force, 4, axis=0)
        state_4x = np.repeat(state, 4, axis=0)

        return {
            "image": _wrist_image_tensor(wrist_image, self.image_size, self.device),
            "tactile": torch.from_numpy(tactile[None]).to(self.device, non_blocking=True),
            "force": torch.from_numpy(force[None]).to(self.device, non_blocking=True),
            "state": torch.from_numpy(state[None]).to(self.device, non_blocking=True),
            "force_4x": torch.from_numpy(force_4x[None]).to(self.device, non_blocking=True),
            "state_4x": torch.from_numpy(state_4x[None]).to(self.device, non_blocking=True),
        }

    def predict_action(
        self,
        base_image: np.ndarray,
        wrist_image: np.ndarray,
        joint_positions: np.ndarray,
        force_torque: np.ndarray,
        tactile_inputs: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        del base_image  # Training used observation.wrist_image only.
        import torch

        obs = self._build_obs(wrist_image, joint_positions, force_torque, tactile_inputs)
        if not self._action_queue:
            with torch.inference_mode():
                result = self.policy.predict_action(
                    obs,
                    num_inference_steps=self.num_inference_steps,
                    solver="euler",
                )
            actions = result["action"][0, :self.execute_steps].detach().float().cpu().numpy()
            if actions.shape != (self.execute_steps, 7) or not np.all(np.isfinite(actions)):
                raise RuntimeError(f"Policy produced an invalid action chunk: {actions}")
            self._action_queue.extend(actions)
        action = self._action_queue.popleft().copy()
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Policy produced an invalid action: {action}")
        action = self._apply_tactile_gripper_safety(action, tactile_inputs or {})
        return action.astype(np.float32)

    @staticmethod
    def _tactile_force_metrics(value: np.ndarray) -> tuple[float, float]:
        force = np.asarray(value, dtype=np.float32).reshape(-1, 3)
        norms = np.linalg.norm(force, axis=-1)
        return float(norms.max(initial=0.0)), float(norms.sum())

    def _apply_tactile_gripper_safety(
        self,
        action: np.ndarray,
        tactile_inputs: dict[str, np.ndarray],
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if not self.tactile_safety_enabled:
            self._last_safe_gripper_command = float(action[6])
            return action

        left = tactile_inputs.get("observation.tactile_left.distributed_force")
        right = tactile_inputs.get("observation.tactile_right.distributed_force")
        if left is None or right is None:
            raise KeyError("Tactile safety requires left/right distributed_force fields.")
        left_max, left_total = self._tactile_force_metrics(left)
        right_max, right_total = self._tactile_force_metrics(right)
        max_taxel = max(left_max, right_max)
        max_total = max(left_total, right_total)

        hard = max_taxel >= self.tactile_hard_max_taxel or max_total >= self.tactile_hard_total
        soft_now = max_taxel >= self.tactile_soft_max_taxel or max_total >= self.tactile_soft_total
        self._tactile_soft_count = self._tactile_soft_count + 1 if soft_now else 0
        soft = self._tactile_soft_count >= self.tactile_trigger_frames
        previous_level = self._tactile_safety_level

        if hard:
            self._tactile_safety_level = "hard"
            self._forced_gripper_command = self.tactile_hard_open_command
            self._tactile_latch_remaining = self.tactile_latch_steps
        elif soft:
            self._tactile_safety_level = "soft"
            baseline = min(float(action[6]), self._last_safe_gripper_command)
            self._forced_gripper_command = max(0.0, baseline - self.tactile_soft_open_step)
            self._tactile_latch_remaining = self.tactile_latch_steps
        elif self._tactile_latch_remaining > 0:
            self._tactile_latch_remaining -= 1
        else:
            self._tactile_safety_level = "off"
            self._forced_gripper_command = None

        if self._forced_gripper_command is not None:
            self._action_queue.clear()
            action[6] = self._forced_gripper_command
        else:
            self._last_safe_gripper_command = float(action[6])

        if self._tactile_safety_level != previous_level or hard:
            print(
                "Xense gripper safety "
                f"{self._tactile_safety_level.upper()}: "
                f"left(max={left_max:.3f},total={left_total:.3f}) "
                f"right(max={right_max:.3f},total={right_total:.3f}) "
                f"gripper_cmd={action[6]:.3f} latch={self._tactile_latch_remaining}"
            )
        return action

    def enforce_gripper_after_filter(self, action: np.ndarray) -> np.ndarray:
        """Ensure action low-pass filtering cannot weaken an active tactile stop."""
        action = np.asarray(action, dtype=np.float32).copy()
        if self.tactile_safety_enabled and self._forced_gripper_command is not None:
            action[6] = self._forced_gripper_command
        return action


class ForceReader:
    def __init__(self, args: argparse.Namespace, robot) -> None:
        from hardware.force.robotiq_fts300 import (
            RobotiqFTS300ModbusSerial,
            RobotiqFTS300ModbusTCP,
        )

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


def calibrate_force_bias(force_reader: ForceReader, fps: float) -> np.ndarray:
    num_samples = max(10, int(FORCE_TARE_SECONDS * fps))
    samples = []
    print("Calibrating force bias. Keep the end effector still and unloaded...")
    for _ in range(num_samples):
        samples.append(force_reader.read())
        time.sleep(1.0 / max(1.0, fps))
    bias = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    print("Force bias [Fx,Fy,Fz,Tx,Ty,Tz] =", np.array2string(bias, precision=4))
    return bias


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
    cos_theta = (np.trace(rotation) - 1.0) / 2.0
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-12:
        return np.zeros(3)
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


class CartesianForceSafetyController:
    def __init__(self, args: argparse.Namespace, dt: float) -> None:
        force_threshold = parse_xyz_vector(args.force_safety_threshold_n, "--force-safety-threshold-n", positive=True)
        torque_threshold = parse_xyz_vector(args.torque_safety_threshold_nm, "--torque-safety-threshold-nm", positive=True)
        force_hard_stop = parse_xyz_vector(args.force_safety_hard_stop_n, "--force-safety-hard-stop-n", positive=True)
        torque_hard_stop = parse_xyz_vector(args.torque_safety_hard_stop_nm, "--torque-safety-hard-stop-nm", positive=True)
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

    def corrected_tcp_from_current(
        self,
        current_tcp: np.ndarray,
        force_torque: np.ndarray,
    ) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
        correction, hard_stop, signed_wrench = self.update(force_torque, current_tcp)
        corrected = np.asarray(current_tcp, dtype=np.float64).copy()
        corrected[:3] += correction[:3]
        correction_rotation = rotvec_to_matrix(correction[3:6])
        current_rotation = rotvec_to_matrix(corrected[3:6])
        corrected[3:6] = matrix_to_rotvec(correction_rotation @ current_rotation)
        return corrected, hard_stop, correction, signed_wrench


class PreviewWindow:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._cv2 = None
        if enabled:
            import cv2

            self._cv2 = cv2

    def show(self, base_image: np.ndarray, wrist_image: np.ndarray) -> None:
        if not self.enabled or self._cv2 is None:
            return
        preview = np.concatenate([base_image, wrist_image], axis=1)
        self._cv2.imshow("base | wrist", preview[:, :, ::-1])
        self._cv2.waitKey(1)

    def close(self) -> None:
        if self.enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()


def clip_and_filter_action(
    raw_action: np.ndarray,
    current_joints: np.ndarray,
    previous_action: np.ndarray | None,
    args: argparse.Namespace,
    step: int,
) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(7).copy()
    current = np.asarray(current_joints, dtype=np.float32).reshape(-1)
    if current.size >= 6:
        max_delta = args.max_startup_joint_delta if step < args.startup_steps else args.max_joint_delta
        delta = np.clip(action[:6] - current[:6], -max_delta, max_delta)
        action[:6] = current[:6] + delta
    action[6] = float(np.clip(action[6], 0.0, 1.0))

    alpha = float(np.clip(args.action_lowpass_alpha, 0.0, 1.0))
    if previous_action is not None and alpha < 1.0:
        action = alpha * action + (1.0 - alpha) * previous_action
    return action.astype(np.float32)


def clip_action_delta(
    action: np.ndarray,
    current_joints: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    clipped = np.asarray(action, dtype=np.float32).copy()
    current = np.asarray(current_joints, dtype=np.float32).reshape(-1)
    if current.size >= 6:
        delta = np.clip(clipped[:6] - current[:6], -max_delta, max_delta)
        clipped[:6] = current[:6] + delta
    clipped[6] = float(np.clip(clipped[6], 0.0, 1.0))
    return clipped


class ForcePositionMixer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.axis_index = {"x": 0, "y": 1, "z": 2}[args.force_position_axis]
        self.axis_sign = 1.0 if float(args.force_position_axis_sign) >= 0.0 else -1.0
        self.direction_sign = 1.0 if float(args.force_position_direction_sign) >= 0.0 else -1.0
        self.contact_threshold = abs(float(args.force_position_contact_threshold_n))
        self.require_contact = bool(args.force_position_require_contact)
        self.manual_target = max(0.0, float(args.force_position_manual_target_n))
        self.min_target = max(0.0, float(args.force_position_min_target_n))
        self.max_target = max(self.min_target, float(args.force_position_max_target_n))
        self.k_m_per_n = max(0.0, float(args.force_position_k_m_per_n))
        self.max_down_step = abs(float(args.force_position_max_down_step_m))
        self.max_up_step = abs(float(args.force_position_max_up_step_m))
        self.lowpass_alpha = float(np.clip(args.force_position_lowpass_alpha, 0.0, 1.0))
        self.use_target_force_magnitude = bool(args.force_position_use_target_force_magnitude)
        self.filtered_target = None

    def _raw_target_force_magnitude(self) -> float | None:
        return self.manual_target if self.manual_target > 0.0 else None

    def _target_force_magnitude(self) -> float | None:
        if self.manual_target <= 0.0:
            return None
        target = float(np.clip(self.manual_target, self.min_target, self.max_target))
        if self.filtered_target is None or self.lowpass_alpha >= 1.0:
            self.filtered_target = target
        else:
            self.filtered_target = (
                self.lowpass_alpha * target + (1.0 - self.lowpass_alpha) * self.filtered_target
            )
        return float(self.filtered_target)

    def apply(
        self,
        action: np.ndarray,
        current_joints: np.ndarray,
        current_tcp: np.ndarray,
        measured_force: np.ndarray,
        robot,
        args: argparse.Namespace,
    ) -> tuple[np.ndarray, bool, dict[str, float]]:
        if self.use_target_force_magnitude:
            target_force = self._target_force_magnitude()
        else:
            target_force = self._raw_target_force_magnitude()
        measured = format_force_torque(measured_force)
        measured_axis = self.axis_sign * float(measured[self.axis_index])
        measured_mag = abs(measured_axis)
        in_contact = measured_mag >= self.contact_threshold
        info = {
            "target_n": 0.0 if target_force is None else float(target_force),
            "measured_n": float(measured_mag),
            "step_m": 0.0,
        }
        if target_force is None or (self.require_contact and not in_contact):
            return action, False, info

        force_error = target_force - measured_mag
        step_toward_contact = float(np.clip(
            self.k_m_per_n * force_error,
            -self.max_up_step,
            self.max_down_step,
        ))
        if abs(step_toward_contact) < 1e-7:
            return action, False, info

        corrected_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
        corrected_tcp[self.axis_index] += self.direction_sign * step_toward_contact
        corrected_action = np.asarray(action, dtype=np.float32).copy()
        try:
            mixed_joints = np.asarray(
                robot.inverse_kinematics(corrected_tcp, qnear=np.asarray(action[:6], dtype=np.float64)),
                dtype=np.float32,
            )
        except RuntimeError as exc:
            print(f"Force-position IK failed, keeping policy action: {exc}")
            return action, False, info

        corrected_action[:6] = mixed_joints[:6]
        corrected_action = clip_action_delta(corrected_action, current_joints, args.max_joint_delta)
        info["step_m"] = self.direction_sign * step_toward_contact
        return corrected_action.astype(np.float32), True, info


class HybridForceVelocityController:
    """Combine the policy position target with Cartesian force/torque corrections.

    The policy supplies a joint-space position target. This controller converts
    that target into a joint velocity baseline, then adds TCP force/torque
    velocity corrections on selected Cartesian axes and maps them back through
    IK over one control period.
    """

    _FORCE_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
    _TORQUE_AXIS_INDEX = {"rx": 3, "ry": 4, "rz": 5}

    def __init__(self, args: argparse.Namespace, dt: float) -> None:
        self.dt = float(dt)
        self.force_axis = args.hybrid_force_axis
        self.force_axis_index = self._FORCE_AXIS_INDEX[self.force_axis]
        self.torque_axis = args.hybrid_torque_axis
        self.torque_axis_index = (
            None if self.torque_axis == "none" else self._TORQUE_AXIS_INDEX[self.torque_axis]
        )
        self.require_contact = bool(args.hybrid_require_contact)
        self.contact_threshold = abs(float(args.hybrid_contact_threshold_n))
        self.force_direction_sign = 1.0 if float(args.hybrid_force_direction_sign) >= 0.0 else -1.0
        self.manual_force_target = max(0.0, float(args.hybrid_manual_target_force_n))
        self.manual_torque_target = float(args.hybrid_manual_target_torque_nm)
        self.min_force_target = max(0.0, float(args.hybrid_min_target_force_n))
        self.max_force_target = max(self.min_force_target, float(args.hybrid_max_target_force_n))
        self.max_torque_target = abs(float(args.hybrid_max_target_torque_nm))
        self.kp_force = float(args.hybrid_kp_force_vel)
        self.ki_force = float(args.hybrid_ki_force_vel)
        self.kp_torque = float(args.hybrid_kp_torque_vel)
        self.ki_torque = float(args.hybrid_ki_torque_vel)
        self.max_linear_vel = abs(float(args.hybrid_max_linear_vel))
        self.max_angular_vel = abs(float(args.hybrid_max_angular_vel))
        self.position_vel_gain = max(0.0, float(args.hybrid_position_vel_gain))
        self.max_joint_vel = abs(float(args.hybrid_max_joint_vel))
        self.integral_limit = abs(float(args.hybrid_integral_limit))
        self.lowpass_alpha = float(np.clip(args.hybrid_lowpass_alpha, 0.0, 1.0))
        self.use_target_force_magnitude = bool(args.hybrid_use_target_force_magnitude)
        self.force_integral = 0.0
        self.torque_integral = 0.0
        self.filtered_twist = np.zeros(6, dtype=np.float64)
        self.axis_order, self.axis_signs, self.axis_preset = resolve_force_safety_axis_mapping(args)
        self.force_frame = resolve_force_safety_frame(args)
        self.tool_offset_m = float(args.force_torque_tool_offset_m)

    def _wrench_in_base(self, force_torque: np.ndarray, current_tcp: np.ndarray) -> np.ndarray:
        raw = format_force_torque(force_torque).astype(np.float64)
        force_raw = raw[:3]
        torque_raw = raw[3:6]
        force = force_raw[self.axis_order] * self.axis_signs
        if self.axis_preset == "fts300-tool":
            torque = np.asarray(
                [
                    torque_raw[0] - force_raw[1] * self.tool_offset_m,
                    torque_raw[1] + self.tool_offset_m * force_raw[0],
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

    def _raw_target_force_magnitude(self) -> float:
        return self.manual_force_target

    def _target_force_magnitude(self) -> float:
        return float(np.clip(self.manual_force_target, self.min_force_target, self.max_force_target))

    def _target_torque(self) -> float:
        if self.torque_axis_index is None:
            return 0.0
        return float(np.clip(self.manual_torque_target, -self.max_torque_target, self.max_torque_target))

    def _integrate_tcp(self, current_tcp: np.ndarray, twist: np.ndarray) -> np.ndarray:
        target_tcp = np.asarray(current_tcp, dtype=np.float64).copy()
        target_tcp[:3] += twist[:3] * self.dt
        delta_rot = rotvec_to_matrix(twist[3:6] * self.dt)
        current_rot = rotvec_to_matrix(target_tcp[3:6])
        target_tcp[3:6] = matrix_to_rotvec(delta_rot @ current_rot)
        return target_tcp

    def apply(
        self,
        action: np.ndarray,
        current_joints: np.ndarray,
        current_tcp: np.ndarray,
        measured_force: np.ndarray,
        robot,
    ) -> tuple[np.ndarray, bool, dict[str, float | np.ndarray]]:
        current = np.asarray(current_joints, dtype=np.float64).reshape(-1)
        policy_action = np.asarray(action, dtype=np.float64).reshape(-1)
        qdot_policy = self.position_vel_gain * (policy_action[:6] - current[:6]) / max(self.dt, 1e-6)
        qdot_policy = np.clip(qdot_policy, -self.max_joint_vel, self.max_joint_vel)

        measured_wrench = self._wrench_in_base(measured_force, current_tcp)
        measured_force_axis = self.force_direction_sign * float(measured_wrench[self.force_axis_index])
        measured_force_mag = abs(measured_force_axis)
        in_contact = measured_force_mag >= self.contact_threshold

        if self.use_target_force_magnitude:
            target_force_mag = self._target_force_magnitude()
        else:
            target_force_mag = self._raw_target_force_magnitude()
        force_error = target_force_mag - measured_force_mag
        force_active = (not self.require_contact or in_contact) and target_force_mag > 0.0
        if force_active:
            self.force_integral = float(
                np.clip(
                    self.force_integral + force_error * self.dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )
        else:
            self.force_integral = 0.0
            force_error = 0.0

        target_torque = self._target_torque()
        measured_torque = 0.0
        torque_error = 0.0
        torque_active = False
        if self.torque_axis_index is not None:
            measured_torque = float(measured_wrench[self.torque_axis_index])
            torque_error = target_torque - measured_torque
            torque_active = (not self.require_contact or in_contact) and abs(target_torque) > 1e-5
            if torque_active:
                self.torque_integral = float(
                    np.clip(
                        self.torque_integral + torque_error * self.dt,
                        -self.integral_limit,
                        self.integral_limit,
                    )
                )
            else:
                self.torque_integral = 0.0
                torque_error = 0.0

        twist = np.zeros(6, dtype=np.float64)
        twist[self.force_axis_index] = self.force_direction_sign * (
            self.kp_force * force_error + self.ki_force * self.force_integral
        )
        twist[:3] = np.clip(twist[:3], -self.max_linear_vel, self.max_linear_vel)
        if self.torque_axis_index is not None:
            twist[self.torque_axis_index] = self.kp_torque * torque_error + self.ki_torque * self.torque_integral
            twist[3:6] = np.clip(twist[3:6], -self.max_angular_vel, self.max_angular_vel)

        if self.lowpass_alpha < 1.0:
            twist = self.lowpass_alpha * twist + (1.0 - self.lowpass_alpha) * self.filtered_twist
        self.filtered_twist = twist

        qdot_force = np.zeros(6, dtype=np.float64)
        active = bool((force_active or torque_active) and np.any(np.abs(twist) > 1e-8))
        if active:
            target_tcp = self._integrate_tcp(current_tcp, twist)
            try:
                force_joints = np.asarray(
                    robot.inverse_kinematics(target_tcp, qnear=current[:6]),
                    dtype=np.float64,
                ).reshape(-1)
                qdot_force = (force_joints[:6] - current[:6]) / max(self.dt, 1e-6)
                qdot_force = np.clip(qdot_force, -self.max_joint_vel, self.max_joint_vel)
            except RuntimeError as exc:
                print(f"Hybrid force IK failed, using policy velocity only: {exc}")
                active = False

        qdot_cmd = np.clip(qdot_policy + qdot_force, -self.max_joint_vel, self.max_joint_vel)
        cmd = np.asarray(policy_action, dtype=np.float32).copy()
        cmd[:6] = (current[:6] + qdot_cmd * self.dt).astype(np.float32)
        if current.size >= 7 and cmd.size >= 7:
            cmd[6] = float(np.clip(policy_action[6], 0.0, 1.0))

        info = {
            "active": active,
            "contact": in_contact,
            "target_force_n": target_force_mag,
            "measured_force_n": measured_force_mag,
            "force_error_n": force_error,
            "target_torque_nm": target_torque,
            "measured_torque_nm": measured_torque,
            "torque_error_nm": torque_error,
            "twist": twist.copy(),
            "qdot_force": qdot_force.copy(),
        }
        return cmd.astype(np.float32), active, info


def apply_force_safety_to_action(
    action: np.ndarray,
    current_joints: np.ndarray,
    current_tcp: np.ndarray,
    force_torque: np.ndarray,
    force_safety: CartesianForceSafetyController,
    robot,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    corrected_tcp, hard_stop, correction, signed_wrench = force_safety.corrected_tcp_from_current(
        current_tcp,
        force_torque,
    )
    safety_active = bool(np.any(np.abs(correction) > 1e-6))
    if not safety_active and not hard_stop:
        return action, False, correction, signed_wrench

    corrected_action = np.asarray(action, dtype=np.float32).copy()
    qnear = np.asarray(current_joints[:6] if hard_stop else action[:6], dtype=np.float64)
    try:
        safe_joints = np.asarray(
            robot.inverse_kinematics(corrected_tcp, qnear=qnear),
            dtype=np.float32,
        )
    except RuntimeError as exc:
        print(f"Force safety IK failed, holding current arm joints: {exc}")
        safe_joints = np.asarray(current_joints[:6], dtype=np.float32)

    corrected_action[:6] = safe_joints[:6]
    if hard_stop:
        corrected_action[6] = float(current_joints[-1])
    max_delta = args.max_startup_joint_delta if hard_stop else args.max_joint_delta
    corrected_action = clip_action_delta(corrected_action, current_joints, max_delta)
    return corrected_action.astype(np.float32), hard_stop, correction, signed_wrench


class ForceTensorBoardLogger:
    COMPONENTS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    POSE_COMPONENTS = ("x", "y", "z", "rx", "ry", "rz")

    def __init__(self, args: argparse.Namespace) -> None:
        self.enabled = bool(args.tensorboard)
        self.writer = None
        self.log_every = max(1, int(args.tensorboard_log_every))
        self.flush_every = max(1, int(args.tensorboard_flush_every))
        self._logged_steps = 0
        self.log_dir = None
        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"TensorBoard disabled: {exc}")
            self.enabled = False
            return

        run_name = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(args.tensorboard_log_dir).expanduser() / run_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f"TensorBoard force logging enabled: {self.log_dir}")

    @staticmethod
    def _norms(wrench: np.ndarray) -> tuple[float, float]:
        wrench = format_force_torque(wrench).astype(np.float64)
        return float(np.linalg.norm(wrench[:3])), float(np.linalg.norm(wrench[3:6]))

    @staticmethod
    def _format_tcp_pose(tcp_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
        if pose.size < 6:
            pose = np.pad(pose, (0, 6 - pose.size))
        return pose[:6]

    def should_log(self, step: int) -> bool:
        return bool(self.enabled and self.writer is not None and step % self.log_every == 0)

    def log(
        self,
        step: int,
        measured_force: np.ndarray,
        hybrid_info: dict[str, float | np.ndarray] | None = None,
        tcp_pose: np.ndarray | None = None,
    ) -> None:
        if not self.should_log(step):
            return

        measured = format_force_torque(measured_force)
        for idx, name in enumerate(self.COMPONENTS):
            self.writer.add_scalar(f"force/measured/{name}", float(measured[idx]), step)

        measured_force_norm, measured_torque_norm = self._norms(measured)
        self.writer.add_scalar("force/measured/force_norm", measured_force_norm, step)
        self.writer.add_scalar("force/measured/torque_norm", measured_torque_norm, step)

        if tcp_pose is not None:
            pose = self._format_tcp_pose(tcp_pose)
            for idx, name in enumerate(self.POSE_COMPONENTS[:3]):
                self.writer.add_scalar(f"tcp_pose/position/{name}", float(pose[idx]), step)
            for idx, name in enumerate(self.POSE_COMPONENTS[3:], start=3):
                self.writer.add_scalar(f"tcp_pose/orientation/{name}", float(pose[idx]), step)
            self.writer.add_scalars(
                "tcp_pose/position_xyz",
                {name: float(pose[idx]) for idx, name in enumerate(self.POSE_COMPONENTS[:3])},
                step,
            )
            self.writer.add_scalars(
                "tcp_pose/orientation_rx_ry_rz",
                {name: float(pose[idx]) for idx, name in enumerate(self.POSE_COMPONENTS[3:], start=3)},
                step,
            )

        if hybrid_info is not None:
            for key in (
                "target_force_n",
                "measured_force_n",
                "force_error_n",
                "target_torque_nm",
                "measured_torque_nm",
                "torque_error_nm",
            ):
                if key in hybrid_info:
                    self.writer.add_scalar(f"hybrid/{key}", float(hybrid_info[key]), step)

        self._logged_steps += 1
        if self._logged_steps % self.flush_every == 0:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def main() -> None:
    args = parse_args()

    from hardware.zmq_core.camera_node import ZMQClientCamera
    from hardware.zmq_core.robot_node import ZMQClientRobot
    from hardware.zmq_core.tactile_node import ZMQClientTactile
    args.dry_run = args.dry_run or args.no_command
    infer = TacForceWMInfer(args)
    tb_logger = ForceTensorBoardLogger(args)

    robot = ZMQClientRobot(port=args.robot_port, host=args.host, timeout_ms=args.robot_timeout_ms)
    base_camera = ZMQClientCamera(
        port=args.base_camera_port,
        host=args.host,
        timeout_ms=args.camera_timeout_ms,
    )
    wrist_camera = ZMQClientCamera(
        port=args.wrist_camera_port,
        host=args.host,
        timeout_ms=args.camera_timeout_ms,
    )
    force_reader = ForceReader(args, robot) if args.collect_force else None
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
    preview = PreviewWindow(args.preview)

    force_bias = np.zeros(6, dtype=np.float32)
    force_safety = None
    force_position_mixer = None
    hybrid_controller = None
    previous_action = None
    period = 1.0 / max(1.0, args.fps)
    next_tick = time.time()

    try:
        robot_dofs = robot.num_dofs()
        if robot_dofs != 7:
            raise RuntimeError(f"Expected UR5 + RG2 robot with 7 DoF, got {robot_dofs}.")
        base_camera.read((args.image_size, args.image_size))
        wrist_camera.read((args.image_size, args.image_size))
        if force_reader is not None:
            force_reader.read()
            force_bias = calibrate_force_bias(force_reader, args.fps)
        if args.force_safety and force_reader is not None:
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
                f"axis_signs={force_safety.axis_signs.tolist()}"
            )
        if args.hybrid_control and force_reader is not None:
            hybrid_controller = HybridForceVelocityController(args, dt=period)
            print(
                "Hybrid force/position velocity control enabled: "
                f"force_axis={args.hybrid_force_axis}, "
                f"torque_axis={args.hybrid_torque_axis}, "
                f"require_contact={args.hybrid_require_contact}, "
                f"contact_threshold={args.hybrid_contact_threshold_n:.3f} N, "
                f"manual_force={args.hybrid_manual_target_force_n:.3f} N, "
                f"manual_torque={args.hybrid_manual_target_torque_nm:.3f} Nm, "
                f"use_target_force_magnitude={args.hybrid_use_target_force_magnitude}, "
                f"max_twist=({args.hybrid_max_linear_vel:.4f} m/s, "
                f"{args.hybrid_max_angular_vel:.4f} rad/s)"
            )
        elif args.force_position_mixing and force_reader is not None:
            force_position_mixer = ForcePositionMixer(args)
            print(
                "Force-position mixing enabled: "
                f"axis={args.force_position_axis}, "
                f"require_contact={args.force_position_require_contact}, "
                f"contact_threshold={args.force_position_contact_threshold_n:.3f} N, "
                f"use_target_force_magnitude={args.force_position_use_target_force_magnitude}, "
                f"target_clip=[{args.force_position_min_target_n:.3f}, "
                f"{args.force_position_max_target_n:.3f}] N"
            )
        if left_tactile is not None and right_tactile is not None:
            format_tactile_frame(left_tactile.read())
            format_tactile_frame(right_tactile.read())

        print("Starting TacForceWM UR5 inference. Press Ctrl-C to stop.")
        step = 0
        while args.max_steps <= 0 or step < args.max_steps:
            now = time.time()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue
            loop_start = time.time()
            next_tick = loop_start + period

            observations = robot.get_observations()
            joints = np.asarray(observations["joint_positions"], dtype=np.float32)
            need_tcp_pose = (
                force_safety is not None
                or force_position_mixer is not None
                or hybrid_controller is not None
                or tb_logger.should_log(step)
            )
            tcp_pose = robot.get_tcp_pose() if need_tcp_pose else np.zeros(6, dtype=np.float64)
            base_image, _ = base_camera.read((args.image_size, args.image_size))
            wrist_image, _ = wrist_camera.read((args.image_size, args.image_size))
            base_image = resize_rgb(base_image, args.image_size)
            wrist_image = resize_rgb(wrist_image, args.image_size)

            force_torque = np.zeros(6, dtype=np.float32)
            if force_reader is not None:
                force_torque = (force_reader.read() - force_bias).astype(np.float32)

            tactile_inputs = {}
            if left_tactile is not None and right_tactile is not None:
                try:
                    left_frame = format_tactile_frame(left_tactile.read())
                    right_frame = format_tactile_frame(right_tactile.read())
                    tactile_inputs = tactile_inputs_from_frames(left_frame, right_frame)
                except RuntimeError as exc:
                    print(f"Tactile frame skipped: {exc}")
                    continue

            raw_action = infer.predict_action(
                base_image,
                wrist_image,
                joints[:6],
                force_torque,
                tactile_inputs,
            )
            action = clip_and_filter_action(raw_action, joints, previous_action, args, step)
            action = infer.enforce_gripper_after_filter(action)
            safety_active = False
            safety_hard_stop = False
            safety_correction = np.zeros(6, dtype=np.float64)
            signed_wrench = np.zeros(6, dtype=np.float64)
            mixing_active = False
            mixing_info = {"target_n": 0.0, "measured_n": 0.0, "step_m": 0.0}
            hybrid_active = False
            hybrid_info = {
                "target_force_n": 0.0,
                "measured_force_n": 0.0,
                "target_torque_nm": 0.0,
                "measured_torque_nm": 0.0,
                "twist": np.zeros(6, dtype=np.float64),
            }
            if hybrid_controller is not None:
                action, hybrid_active, hybrid_info = hybrid_controller.apply(
                    action,
                    joints,
                    tcp_pose,
                    force_torque,
                    robot,
                )
            elif force_position_mixer is not None:
                action, mixing_active, mixing_info = force_position_mixer.apply(
                    action,
                    joints,
                    tcp_pose,
                    force_torque,
                    robot,
                    args,
                )
            if force_safety is not None:
                action, safety_hard_stop, safety_correction, signed_wrench = apply_force_safety_to_action(
                    action,
                    joints,
                    tcp_pose,
                    force_torque,
                    force_safety,
                    robot,
                    args,
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
            tb_logger.log(step, force_torque, hybrid_info, tcp_pose=tcp_pose)
            previous_action = action.copy()

            if not args.dry_run:
                robot.command_joint_state(action)
                if args.command_gripper_separately:
                    robot.command_gripper(float(action[6]), force=args.gripper_force)

            if step % max(1, args.print_every) == 0:
                elapsed = time.time() - loop_start
                print(
                    f"step={step:05d} elapsed={elapsed:.3f}s "
                    # f"q_cmd={np.array2string(action[:6], precision=4)} "
                    f"gripper={action[6]:.3f} "
                    f"force_torque={np.array2string(force_torque, precision=3)} "
                    f"hybrid={'on' if hybrid_active else 'off'} "
                    f"hF={hybrid_info['measured_force_n']:.3f}->{hybrid_info['target_force_n']:.3f}N "
                    f"hT={hybrid_info['measured_torque_nm']:.3f}->{hybrid_info['target_torque_nm']:.3f}Nm "
                    f"twist={np.array2string(np.asarray(hybrid_info['twist']), precision=4)} "
                    f"fp_mix={'on' if mixing_active else 'off'} "
                    f"fp_target={mixing_info['target_n']:.3f}N "
                    f"fp_meas={mixing_info['measured_n']:.3f}N "
                    f"fp_step={mixing_info['step_m']:.6f}m "
                    # f"force_safety={'HARD' if safety_hard_stop else ('active' if safety_active else 'off')}"
                )
            preview.show(base_image, wrist_image)
            step += 1
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        tb_logger.close()
        preview.close()
        if force_reader is not None:
            force_reader.close()
        if left_tactile is not None:
            left_tactile.close()
        if right_tactile is not None:
            right_tactile.close()
        robot.close()


if __name__ == "__main__":
    main()
