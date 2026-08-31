#!/usr/bin/env python3
"""Unit 2: Tabular Q-Learning on FrozenLake-v1 and Taxi-v3."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun

import numpy as np
import gymnasium as gym
from tqdm import tqdm


def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))


def greedy_policy(Qtable, state):
    return int(np.argmax(Qtable[state][:]))


def epsilon_greedy_policy(Qtable, state, epsilon):
    if np.random.random() > epsilon:
        return greedy_policy(Qtable, state)
    return int(np.random.randint(0, Qtable.shape[1]))


def train(
    n_training_episodes,
    min_epsilon,
    max_epsilon,
    decay_rate,
    env,
    max_steps,
    Qtable,
    learning_rate,
    gamma,
):
    for episode in tqdm(range(n_training_episodes), desc=env.spec.id):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, _ = env.reset()
        for _ in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )
            if terminated or truncated:
                break
            state = new_state
    return Qtable


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed=None):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state, _ = env.reset(seed=seed[episode])
        else:
            state, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total)
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def main():
    with UnitRun("2", "unit2") as run:
        try:
            # FrozenLake
            env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
            Q_fl = initialize_q_table(env.observation_space.n, env.action_space.n)
            Q_fl = train(10000, 0.05, 1.0, 0.0005, env, 99, Q_fl, 0.7, 0.95)
            mean_fl, std_fl = evaluate_agent(env, 99, 100, Q_fl)
            np.save(run.out / "qtable_frozenlake.npy", Q_fl)
            env.close()

            # Taxi
            env = gym.make("Taxi-v3")
            Q_taxi = initialize_q_table(env.observation_space.n, env.action_space.n)
            Q_taxi = train(25000, 0.05, 1.0, 0.005, env, 99, Q_taxi, 0.7, 0.95)
            eval_seed = [
                16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50,
                172, 100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54,
                55, 161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82,
                45, 95, 89, 59, 95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11,
                50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38, 112, 102, 168, 123,
                97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148,
            ]
            mean_taxi, std_taxi = evaluate_agent(env, 99, 100, Q_taxi, seed=eval_seed)
            np.save(run.out / "qtable_taxi.npy", Q_taxi)
            env.close()

            run.success(
                {
                    "frozenlake_mean": mean_fl,
                    "frozenlake_std": std_fl,
                    "taxi_mean": mean_taxi,
                    "taxi_std": std_taxi,
                }
            )
            print(f"Unit2 PASS FL={mean_fl:.2f} Taxi={mean_taxi:.2f}")
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
