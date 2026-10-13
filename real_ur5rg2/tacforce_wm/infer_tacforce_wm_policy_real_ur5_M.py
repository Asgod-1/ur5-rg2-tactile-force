"""Run the trained TacForceWM flow-matching policy on the real UR5 + RG2.

The robot/camera/Xense/force-sensor loop and all safety controllers are reused
from ``infer_tactile_smolvla_real_ur5_hybrid_control.py``.  This file replaces
only its model adapter with the TacForceWM policy trained by ``tacforce_wm``.
"""
from __future__ import annotations

import os
import sys
import argparse
from collections import deque
from pathlib import Path

import numpy as np


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
REAL_UR5_DIR = SCRIPT_DIR.parent


def _resolve_training_root() -> Path:
    override = os.environ.get("TACFORCE_WM_ROOT") or os.environ.get("TACFORCE_ROOT")
    root = Path(override).expanduser().resolve() if override else SCRIPT_DIR
    if not (root / "train.py").is_file() or not (root / "models").is_dir():
        raise FileNotFoundError(f"TacForceWM source directory not found: {root}")
    return root


TRAINING_ROOT = _resolve_training_root()

for path in (REAL_UR5_DIR.parent, REAL_UR5_DIR, SCRIPT_DIR, TRAINING_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import infer_tactile_smolvla_real_ur5_hybrid_control as real_control  # noqa: E402


DEFAULT_POLICY_CHECKPOINTS = (
    TRAINING_ROOT / "outputs/wrist_policy_run/checkpoints/best.pt",
    TRAINING_ROOT / "outputs/wrist_policy_run/checkpoints/latest.pt",
    TRAINING_ROOT / "outputs/best.pt",
    TRAINING_ROOT / "outputs/latest.pt",
)
LOCAL_DINO_CHECKPOINT = TRAINING_ROOT / "DINOv2/model.safetensors"
ACTIVE_INFER = None


def _extract_tactile_safety_args() -> argparse.Namespace:
    """Parse our extra flags, leaving all reference-loop flags untouched."""
    parser = argparse.ArgumentParser(add_help=False)
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
    extra, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return extra


def _resolve_policy_checkpoint(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TacForceWM policy checkpoint not found: {path}")
        return path
    for path in DEFAULT_POLICY_CHECKPOINTS:
        if path.is_file():
            return path.resolve()
    tried = "\n  ".join(str(path) for path in DEFAULT_POLICY_CHECKPOINTS)
    raise FileNotFoundError(
        "No TacForceWM policy checkpoint was found. Pass it through --policy-path. "
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


def _format_xense_tactile_frame(
    frame: dict,
    max_points: int,
    *,
    original_format,
) -> dict:
    """Preserve the native Xense force grid through the shared robot loop."""
    raw = dict(frame)
    force_key = "force" if "force" in raw else "distributed_force"
    if force_key not in raw:
        raise RuntimeError(
            f"Xense tactile frame has no force field; available keys={sorted(raw)}"
        )
    force = _xense_force_grid(raw[force_key], side="live")

    # The shared loop expects these legacy names for its safety checks.
    raw["distributed_force"] = force.reshape(-1, 3)
    if "wrench" not in raw and "force6d" in raw:
        raw["wrench"] = raw["force6d"]
    formatted = original_format(raw, max_points)
    formatted["force"] = force
    formatted["distributed_force"] = force.reshape(-1, 3)
    return formatted


def _xense_tactile_inputs_from_frames(
    left_frame: dict,
    right_frame: dict,
    *,
    original_inputs,
) -> dict[str, np.ndarray]:
    inputs = original_inputs(left_frame, right_frame)
    zero = np.zeros((35, 20, 3), dtype=np.float32)
    inputs["observation.tactile_left.force"] = np.asarray(
        left_frame.get("force", zero), dtype=np.float32
    )
    inputs["observation.tactile_right.force"] = np.asarray(
        right_frame.get("force", zero), dtype=np.float32
    )
    return inputs


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


class TacForceWMRealUR5Infer:
    """Adapter matching the interface expected by the proven real-robot loop."""

    def __init__(self, args) -> None:
        global ACTIVE_INFER
        import torch

        from trainers.policy_trainer import build_policy
        from utils.normalizer import MultiFieldNormalizer

        if not args.collect_tactile:
            raise ValueError("TacForceWM inference requires both Xense sensors; remove --no-tactile.")
        if not args.collect_force:
            raise ValueError("TacForceWM inference requires force/torque input; use --collect-force.")
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_path = _resolve_policy_checkpoint(args.policy_path)
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

        # These names make the shared loop initialize both tactile clients.
        self.tactile_keys = [
            "observation.tactile_left.force",
            "observation.tactile_right.force",
        ]
        self.policy_type = "tacforce_wm_flow_policy"
        self.last_predicted_force = None
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
        ACTIVE_INFER = self

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
        force_now = real_control.format_force_torque(force_torque)

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


def main() -> None:
    tactile_args = _extract_tactile_safety_args()
    original_parse_args = real_control.parse_args
    original_clip_action = real_control.clip_and_filter_action
    original_format_tactile_frame = real_control.format_tactile_frame
    original_tactile_inputs_from_frames = real_control.tactile_inputs_from_frames

    def parse_args_with_tactile_safety():
        args = original_parse_args()
        for key, value in vars(tactile_args).items():
            setattr(args, key, value)
        return args

    def clip_action_with_tactile_safety(*args, **kwargs):
        action = original_clip_action(*args, **kwargs)
        if ACTIVE_INFER is not None:
            action = ACTIVE_INFER.enforce_gripper_after_filter(action)
        return action

    def format_xense_frame(frame, max_points):
        return _format_xense_tactile_frame(
            frame,
            max_points,
            original_format=original_format_tactile_frame,
        )

    def xense_inputs(left_frame, right_frame):
        return _xense_tactile_inputs_from_frames(
            left_frame,
            right_frame,
            original_inputs=original_tactile_inputs_from_frames,
        )

    # Preserve the reference implementation's complete hardware and safety loop.
    real_control.parse_args = parse_args_with_tactile_safety
    real_control.clip_and_filter_action = clip_action_with_tactile_safety
    real_control.format_tactile_frame = format_xense_frame
    real_control.tactile_inputs_from_frames = xense_inputs
    real_control.RealUR5VLAInfer = TacForceWMRealUR5Infer
    real_control.main()


if __name__ == "__main__":
    main()
