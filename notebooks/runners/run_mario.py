#!/usr/bin/env python3
"""PPO on Super Mario Bros. 1-1 using gym-super-mario-bros + Stable-Baselines3.

Uses conda env deep-rl-class. The NES env speaks the old Gym step API, so this
script wraps it as a Gymnasium environment before training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun

ENV_ID = "SuperMarioBros-1-1"
TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 8


def _make_nes(target=(1, 1)):
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
    from nes_py.wrappers import JoypadSpace

    # Bypass gym.make: Gym 0.26 TimeLimit expects the 5-tuple step API.
    # target=None plays the full game and advances after each flag.
    return JoypadSpace(SuperMarioBrosEnv(target=target), SIMPLE_MOVEMENT)


class MarioEnv(gym.Env):
    """Discrete simple movement, Gymnasium step API."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, target=(1, 1)):
        super().__init__()
        self._env = _make_nes(target)
        self.action_space = spaces.Discrete(self._env.action_space.n)
        self.observation_space = spaces.Box(0, 255, (84, 84, 4), dtype=np.uint8)
        self._frames = np.zeros((84, 84, 4), dtype=np.uint8)

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def _push(self, frame: np.ndarray) -> np.ndarray:
        self._frames = np.roll(self._frames, shift=-1, axis=-1)
        self._frames[:, :, -1] = frame
        return self._frames

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        self._frames.fill(0)
        frame = self._preprocess(obs)
        for _ in range(4):
            stacked = self._push(frame)
        return stacked, {}

    def step(self, action):
        total = 0.0
        terminated = False
        info = {}
        frame = None
        for _ in range(4):
            obs, reward, done, info = self._env.step(int(action))
            total += float(reward)
            frame = self._preprocess(obs)
            terminated = bool(done)
            if terminated:
                break
        return self._push(frame), total, terminated, False, info

    def render(self):
        return self._env.render(mode="rgb_array")

    def close(self):
        self._env.close()


def record_video(model, path: Path, max_steps: int = 2000) -> dict:
    import imageio.v2 as imageio

    env = _make_nes()
    frames = []
    obs = env.reset()
    stacked = np.zeros((84, 84, 4), dtype=np.uint8)
    max_x = 0
    flag = False
    ep_reward = 0.0

    def push(raw):
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        stacked[:] = np.roll(stacked, shift=-1, axis=-1)
        stacked[:, :, -1] = small

    for _ in range(4):
        push(obs)
    for _ in range(max_steps):
        action, _ = model.predict(stacked, deterministic=True)
        done = False
        for _skip in range(4):
            obs, reward, done, info = env.step(int(action))
            ep_reward += float(reward)
            max_x = max(max_x, int(info.get("x_pos", 0)))
            flag = flag or bool(info.get("flag_get", False))
            # nes-py reuses one screen buffer; store a copy or every frame is the last one.
            frames.append(np.array(obs, copy=True))
            if done:
                break
        push(obs)
        if done:
            break
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames[::2], fps=30)
    return {"video_x_pos": max_x, "video_flag": flag, "video_reward": ep_reward, "video": str(path)}


def make_mario():
    from stable_baselines3.common.monitor import Monitor

    return Monitor(MarioEnv())


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    with UnitRun("mario", "mario") as run:
        try:
            train_env = VecMonitor(SubprocVecEnv([make_mario for _ in range(N_ENVS)]))
            eval_env = VecMonitor(SubprocVecEnv([make_mario]))
            model = PPO(
                "CnnPolicy",
                train_env,
                learning_rate=2.5e-4,
                n_steps=128,
                batch_size=256,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.1,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                device="cuda",
                tensorboard_log=str(run.out / "tb"),
            )
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=str(run.out),
                log_path=str(run.out),
                eval_freq=max(10_000 // N_ENVS, 1),
                n_eval_episodes=3,
                deterministic=True,
            )
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
            model_path = run.out / "ppo-mario-1-1"
            model.save(str(model_path))

            best = run.out / "best_model.zip"
            if best.exists():
                model = PPO.load(str(best), device="cuda")
            video_stats = record_video(model, run.out / "mario-1-1.mp4")
            train_env.close()
            eval_env.close()
            run.success(
                {
                    "env": ENV_ID,
                    "timesteps": TOTAL_TIMESTEPS,
                    **video_stats,
                },
                notes="PPO CnnPolicy, SIMPLE_MOVEMENT, frame-skip 4, 84x84x4",
            )
            print("Mario PASS", video_stats)
        except Exception as exc:
            run.fail(exc)
            raise


if __name__ == "__main__":
    main()
