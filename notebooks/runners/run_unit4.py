#!/usr/bin/env python3
"""Unit 4: REINFORCE on CartPole-v1 and Pixelcopter-PLE-v0."""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_gym(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_gym(env, action):
    out = env.step(action)
    if len(out) == 5:
        state, reward, terminated, truncated, info = out
        return state, reward, terminated or truncated, info
    state, reward, done, info = out
    return state, reward, done, info


class PolicyCartPole(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class PolicyPixel(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = reset_gym(env)
        for _ in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = step_gym(env, action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns_t):
            policy_loss.append(-log_prob * disc_return)
        loss = torch.cat(policy_loss).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}",
                flush=True,
            )
    return scores


def evaluate(env, policy, n_eval, max_t):
    totals = []
    for _ in range(n_eval):
        state = reset_gym(env)
        ep = 0.0
        for _ in range(max_t):
            action, _ = policy.act(state)
            state, reward, done, _ = step_gym(env, action)
            ep += reward
            if done:
                break
        totals.append(ep)
    return float(np.mean(totals)), float(np.std(totals))


def main():
    with UnitRun("4", "unit4") as run:
        try:
            # CartPole
            env = gym.make("CartPole-v1")
            s_size = env.observation_space.shape[0]
            a_size = env.action_space.n
            policy = PolicyCartPole(s_size, a_size, 16).to(device)
            opt = optim.Adam(policy.parameters(), lr=1e-2)
            scores_cp = reinforce(env, policy, opt, 1000, 1000, 1.0, 100)
            mean_cp, std_cp = evaluate(env, policy, 10, 1000)
            torch.save(policy.state_dict(), run.out / "cartpole_policy.pt")
            np.save(run.out / "cartpole_scores.npy", np.array(scores_cp))
            env.close()

            # Pixelcopter via gym-games / PLE
            import gym as old_gym  # noqa: F401
            import gym_pygame  # noqa: F401 — registers Pixelcopter-PLE-v0

            env = old_gym.make("Pixelcopter-PLE-v0")
            s_size = env.observation_space.shape[0]
            a_size = env.action_space.n
            policy_p = PolicyPixel(s_size, a_size, 64).to(device)
            opt_p = optim.Adam(policy_p.parameters(), lr=1e-4)
            scores_pc = reinforce(env, policy_p, opt_p, 50000, 10000, 0.99, 1000)
            mean_pc, std_pc = evaluate(env, policy_p, 10, 10000)
            torch.save(policy_p.state_dict(), run.out / "pixelcopter_policy.pt")
            np.save(run.out / "pixelcopter_scores.npy", np.array(scores_pc))
            env.close()

            run.success(
                {
                    "cartpole_mean": mean_cp,
                    "cartpole_std": std_cp,
                    "cartpole_last100": float(np.mean(scores_cp[-100:])),
                    "pixelcopter_mean": mean_pc,
                    "pixelcopter_std": std_pc,
                    "pixelcopter_last100": float(np.mean(scores_pc[-100:])),
                }
            )
            print(f"Unit4 PASS CP={mean_cp:.2f} PC={mean_pc:.2f}")
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
