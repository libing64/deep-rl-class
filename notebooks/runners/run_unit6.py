#!/usr/bin/env python3
"""Unit 6: A2C on PandaReachDense-v3 and PandaPickAndPlace-v3."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun

import gymnasium as gym
import panda_gym  # noqa: F401
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train_one(env_id: str, out: Path, timesteps: int = 1_000_000):
    env = make_vec_env(env_id, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = A2C(policy="MultiInputPolicy", env=env, verbose=1, device="cuda")
    model.learn(timesteps)
    model_path = out / f"a2c-{env_id}"
    model.save(str(model_path))
    env.save(str(out / f"vecnormalize-{env_id}.pkl"))

    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecNormalize.load(str(out / f"vecnormalize-{env_id}.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    eval_env.close()
    env.close()
    return float(mean), float(std)


def main():
    with UnitRun("6", "unit6") as run:
        try:
            mean_r, std_r = train_one("PandaReachDense-v3", run.out)
            mean_p, std_p = train_one("PandaPickAndPlace-v3", run.out)
            run.success(
                {
                    "PandaReachDense_mean": mean_r,
                    "PandaReachDense_std": std_r,
                    "PandaPickAndPlace_mean": mean_p,
                    "PandaPickAndPlace_std": std_p,
                },
                notes="target Reach >= -3.5",
            )
            print(f"Unit6 PASS reach={mean_r:.2f} pick={mean_p:.2f}")
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
