"""Shared scene and run selection for training, latent caching, and inference."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCENES_CONFIG = TRAINING_ROOT / "config" / "scenes.yaml"


def add_experiment_arguments(parser: argparse.ArgumentParser, *, policy: bool = False,
                             config_overrides: bool = True) -> None:
    parser.add_argument("--scene", help="Scene name registered in config/scenes.yaml.")
    parser.add_argument("--suffix", default="", help="Run suffix, e.g. batch128 or lr1e4.")
    if policy:
        parser.add_argument(
            "--dynamics-suffix", default=None,
            help="Dynamics/cache suffix to reuse; defaults to --suffix. Use '' for no suffix.",
        )
    if config_overrides:
        parser.add_argument(
            "--set", dest="config_overrides", action="append", default=[], metavar="KEY=VALUE",
            help="Override a YAML setting; repeat for multiple settings (e.g. train.batch_size=64).",
        )


def _component(value: str, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return value
    if value in {".", ".."} or not re.fullmatch(r"[\w.-]+", value):
        raise ValueError(f"{label} must be a name containing only letters, digits, '_', '-', or '.'.")
    return value


def experiment_run_name(scene: str, task: str, suffix: str = "") -> str:
    scene = _component(scene, "--scene")
    suffix = _component(suffix, "--suffix", allow_empty=True)
    return f"{scene}_{task}" + (f"_{suffix}" if suffix else "")


def _absolute_path(value: str | Path, root: Path = TRAINING_ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def policy_checkpoint_candidates(scene: str, suffix: str = "", *, output_root: str = "outputs",
                                 training_root: Path = TRAINING_ROOT) -> tuple[Path, Path]:
    run_dir = _absolute_path(output_root, training_root) / experiment_run_name(scene, "policy", suffix)
    return tuple(run_dir / "checkpoints" / name for name in ("best.pt", "latest.pt"))


def _apply_config_override(cfg: dict, assignment: str) -> None:
    path, separator, text = assignment.partition("=")
    if not separator or not path.strip():
        raise ValueError(f"Expected --set KEY=VALUE, got {assignment!r}.")
    path = path.strip()
    keys = path.split(".")
    target = cfg
    for key in keys[:-1]:
        if not isinstance(target.get(key), dict):
            raise ValueError(f"Unknown config setting: {path}")
        target = target[key]
    if keys[-1] not in target:
        raise ValueError(f"Unknown config setting: {path}")
    value = yaml.safe_load(text)
    # YAML 1.1 treats scientific notation without a decimal point as a string.
    if isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", text):
        value = float(text)
    target[keys[-1]] = value


def build_experiment_config(task: str, source: str, args: argparse.Namespace,
                            overrides: dict | None = None) -> dict:
    from .train_utils import build_canonical_config, load_yaml

    cfg = load_yaml(source)
    scene = getattr(args, "scene", None)
    suffix = getattr(args, "suffix", "")
    dynamics_suffix = getattr(args, "dynamics_suffix", None)
    if not scene and (suffix or dynamics_suffix is not None):
        raise ValueError("--suffix and --dynamics-suffix require --scene.")
    if scene:
        run_name = experiment_run_name(scene, task, suffix)
        scenes = load_yaml(str(SCENES_CONFIG))
        if scene not in scenes:
            raise ValueError(f"Unknown scene {scene!r}. Available scenes: {', '.join(scenes)}")
        entry = scenes[scene]
        if not isinstance(entry, dict) or not entry.get("data_root"):
            raise ValueError(f"Scene {scene!r} must define data_root in {SCENES_CONFIG}.")
        cfg["data"]["root_dir"] = str(_absolute_path(entry["data_root"]))
        cfg["output"]["run_name"] = run_name
        if task == "policy":
            dynamics_suffix = suffix if dynamics_suffix is None else dynamics_suffix
            dynamics_name = experiment_run_name(scene, "dynamics", dynamics_suffix)
            backend = cfg["model"]["backend"]
            backend["config_path"] = f"${{output.root_dir}}/{dynamics_name}/resolved_config.yaml"
            backend["ckpt_path"] = f"${{output.root_dir}}/{dynamics_name}/checkpoints/best_model.pt"
            cache_root = entry.get("cache_root", f"../data/{scene}_policy_cache")
            cache_suffix = f"_{dynamics_suffix}" if dynamics_suffix else ""
            cfg["data"]["latent_cache_root_dir"] = str(_absolute_path(f"{cache_root}{cache_suffix}"))

    # Explicit command-line settings take precedence over scene defaults, including null values.
    for assignment in getattr(args, "config_overrides", []):
        _apply_config_override(cfg, assignment)
    if scene:
        cfg["output"]["root_dir"] = str(_absolute_path(cfg["output"]["root_dir"]))
        cfg["data"]["root_dir"] = str(_absolute_path(cfg["data"]["root_dir"]))
        if cfg["data"].get("latent_cache_root_dir"):
            cfg["data"]["latent_cache_root_dir"] = str(_absolute_path(cfg["data"]["latent_cache_root_dir"]))
    return build_canonical_config(task, cfg, overrides=overrides)
