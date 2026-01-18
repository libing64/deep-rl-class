"""
Shared utilities for the Deep RL class notebooks.

This module is intentionally dependency-light and focuses on:
- Recording replay videos (MP4) for debugging and reporting

It supports:
- Gymnasium only
- Stable-Baselines3-style models via `model.predict()` / `model.save()`
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import json
import datetime

import numpy as np
import gymnasium as gym  # type: ignore
from gymnasium.wrappers import RecordVideo  # type: ignore


def _reset_env(env) -> Any:
    """Handle Gymnasium reset() -> (obs, info) and Gym reset() -> obs."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _step_env(env, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """
    Handle Gymnasium step() -> (obs, reward, terminated, truncated, info)
    and Gym step() -> (obs, reward, done, info)
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), dict(info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), dict(info)


def _json_default(o):
    """Convert numpy / torch objects to JSON-serializable Python objects."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    try:
        import torch  # type: ignore

        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def record_video(
    *,
    model: Any,
    env_id: str,
    out_path: Union[str, Path],
    video_length: int = 1000,
    fps: int = 30,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Path:
    """
    Record one episode (or up to `video_length` steps) for a `model.predict()` model.

    Uses Gymnasium RecordVideo wrapper so we don't keep frames in RAM.
    Returns the final video path (best effort; wrapper may create a slightly different name).
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # For Gymnasium, render_mode must be set at env creation for rgb_array output.
    env = gym.make(env_id, render_mode="rgb_array")

    try:
        # some wrappers read this metadata
        try:
            env.metadata["render_fps"] = fps
        except Exception:
            pass

        env = RecordVideo(
            env,
            video_folder=str(out_path.parent),
            name_prefix=out_path.stem,
            episode_trigger=lambda ep: ep == 0,
        )

        # reset
        if seed is not None:
            try:
                obs = env.reset(seed=seed)
            except TypeError:
                obs = env.reset()
        else:
            obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        for _ in range(int(video_length)):
            action, _ = model.predict(obs, deterministic=deterministic)
            if isinstance(action, np.ndarray) and action.shape == (1,):
                action = action.item()
            obs, _, done, _ = _step_env(env, action)
            if done:
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Gymnasium creates `<prefix>-episode-0.mp4` (or similar). Try to normalize to out_path.
    candidates = sorted(out_path.parent.glob(out_path.stem + "*.mp4"))
    if candidates:
        try:
            if candidates[0] != out_path:
                candidates[0].replace(out_path)
        except Exception:
            # If rename fails, return the created one.
            return candidates[0]
    return out_path


def save_and_evaluate(
    *,
    model: Any,
    env_id: str,
    eval_env: Any,
    out_root: Union[str, Path] = "exports",
    video_fps: int = 30,
    n_eval_episodes: int = 10,
    video_length: int = 1000,
    deterministic: bool = True,
) -> Tuple[Path, float, float]:
    """Save model + eval results + optional replay video to disk."""
    from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / str(env_id) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model (expects SB3-style .save())
    model.save(str(out_dir / "model"))

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=int(n_eval_episodes),
        deterministic=deterministic,
    )

    results = {
        "env_id": str(env_id),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "n_eval_episodes": int(n_eval_episodes),
        "eval_datetime": datetime.datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    # Best-effort video
    try:
        record_video(
            model=model,
            env_id=str(env_id),
            out_path=out_dir / "replay.mp4",
            video_length=video_length,
            fps=video_fps,
            deterministic=deterministic,
        )
    except Exception as e:
        # Keep saving even if video fails
        with open(out_dir / "video_error.txt", "w") as f:
            f.write(repr(e) + "\n")

    return out_dir, float(mean_reward), float(std_reward)

