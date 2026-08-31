#!/usr/bin/env python3
"""Unit 1: PPO on LunarLander-v3 (1M timesteps)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    with UnitRun("1", "unit1") as run:
        try:
            env = make_vec_env("LunarLander-v3", n_envs=16)
            model = PPO(
                policy="MlpPolicy",
                env=env,
                n_steps=1024,
                batch_size=64,
                n_epochs=4,
                gamma=0.999,
                gae_lambda=0.98,
                ent_coef=0.01,
                verbose=1,
                device="cuda",
            )
            model.learn(total_timesteps=1_000_000)
            model_path = run.out / "ppo-LunarLander-v3"
            model.save(str(model_path))

            eval_env = gym.make("LunarLander-v3")
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True
            )
            eval_env.close()
            env.close()
            run.success(
                {"mean_reward": float(mean_reward), "std_reward": float(std_reward)},
                notes="LunarLander-v3 (gymnasium); 1M timesteps",
            )
            print(f"Unit1 PASS mean={mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
