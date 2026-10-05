#!/usr/bin/env python3
"""Reach the Super Mario Bros. 1-1 flag.

The search keeps the action sequence that has gone farthest, replays it back
to that point, and only then tries new actions. A small imitation update keeps
the network aligned with that sequence so the new tail is not pure noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import now_iso, write_json
from run_mario import MarioEnv, _make_nes

OUT = Path(__file__).resolve().parents[1] / "results" / "mario"
MAX_STEPS = 8_000
FLAG_X = 3161


def _finish(actions, xs, reward, flag):
    actions = [int(a) for a in actions]
    xs = [int(x) for x in xs]
    if xs:
        peak = int(np.argmax(xs))
        actions = actions[: peak + 1]
        xs = xs[: peak + 1]
    max_x = int(xs[-1]) if xs else 0
    return {
        "actions": np.asarray(actions, dtype=np.int64),
        "xs": np.asarray(xs, dtype=np.int32),
        "max_x": max_x,
        "flag": bool(flag or max_x >= FLAG_X),
        "reward": float(reward),
    }


def run_episode(model, prefix, explore_steps: int, epsilon: float, sticky: int) -> dict:
    env = MarioEnv()
    obs, _ = env.reset()
    actions, xs = [], []
    reward = 0.0
    flag = False
    sticky_left = 0
    sticky_action = 0
    prefix = [int(a) for a in prefix]
    try:
        for t in range(len(prefix) + explore_steps):
            if t < len(prefix):
                action = prefix[t]
            elif sticky_left > 0:
                action = sticky_action
                sticky_left -= 1
            elif np.random.rand() < epsilon:
                sticky_action = int(env.action_space.sample())
                sticky_left = max(0, sticky - 1)
                action = sticky_action
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            obs, step_reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            x = int(info.get("x_pos", 0))
            xs.append(x)
            reward += float(step_reward)
            flag = flag or bool(info.get("flag_get", False))
            if terminated or truncated or flag:
                break
    finally:
        env.close()
    return _finish(actions, xs, reward, flag)


def imitate(model, actions_src, obs_src) -> float:
    if len(actions_src) == 0:
        return 0.0
    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    actions = np.asarray(actions_src, dtype=np.int64)
    losses = []
    for _ in range(2):
        order = np.random.permutation(len(actions))
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            obs_tensor, _ = policy.obs_to_tensor(obs_src[batch])
            act_tensor = torch.as_tensor(actions[batch], device=policy.device)
            loss = -policy.get_distribution(obs_tensor).log_prob(act_tensor).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.item()))
    policy.eval()
    return float(np.mean(losses)) if losses else 0.0


def collect_obs(actions) -> np.ndarray:
    """Replay actions and return the observation that preceded each one."""
    env = MarioEnv()
    obs, _ = env.reset()
    stored = []
    try:
        for action in actions:
            stored.append(np.array(obs, copy=True))
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated or info.get("flag_get", False):
                break
    finally:
        env.close()
    return np.stack(stored) if stored else np.zeros((0, 84, 84, 4), dtype=np.uint8)


def save_best(model, episode, best_x: int) -> None:
    np.savez(
        OUT / "best_actions.npz",
        actions=episode["actions"],
        xs=episode["xs"],
        max_x=np.int32(episode["max_x"]),
    )
    model.save(str(OUT / "best_model"))
    model.save(str(OUT / f"best_model_x{best_x}"))


def record_flag_video(actions, path: Path) -> dict:
    import imageio.v2 as imageio

    env = _make_nes()
    obs = env.reset()
    frames = []
    max_x = 0
    flag = False
    reward = 0.0
    try:
        for action in actions:
            done = False
            for _ in range(4):
                obs, step_reward, done, info = env.step(int(action))
                frames.append(np.array(obs, copy=True))
                reward += float(step_reward)
                max_x = max(max_x, int(info.get("x_pos", 0)))
                flag = flag or bool(info.get("flag_get", False))
                if done:
                    break
            if done:
                break
    finally:
        env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames[::2], fps=30)
    return {"video_x_pos": max_x, "video_flag": flag, "video_reward": reward, "video_frames": len(frames)}


def latest_checkpoint() -> Path:
    ranked = []
    for path in OUT.glob("best_model_x*.zip"):
        try:
            ranked.append((int(path.stem.split("x")[-1]), path))
        except ValueError:
            continue
    if ranked:
        return max(ranked)[1]
    return OUT / "best_model.zip"


def main():
    from stable_baselines3 import PPO

    OUT.mkdir(parents=True, exist_ok=True)
    source = latest_checkpoint()
    model = PPO.load(str(source), device="cuda")
    started = now_iso()
    print(f"resume {source}", flush=True)

    archive_path = OUT / "best_actions.npz"
    if archive_path.exists():
        saved = np.load(archive_path)
        best = _finish(saved["actions"], saved["xs"], 0.0, int(saved["max_x"]) >= FLAG_X)
        print(f"loaded actions x={best['max_x']} len={len(best['actions'])}", flush=True)
    else:
        best = {"actions": np.zeros(0, dtype=np.int64), "xs": np.zeros(0, dtype=np.int32), "max_x": 0, "flag": False}
        for i in range(1, 41):
            episode = run_episode(model, [], MAX_STEPS, epsilon=0.03, sticky=4)
            print(f"recover {i} x={episode['max_x']} flag={episode['flag']}", flush=True)
            if episode["max_x"] > best["max_x"]:
                best = episode
                save_best(model, best, best["max_x"])
            if best["max_x"] >= 2400 or best["flag"]:
                break

    history = []
    stall = 0
    iteration = 0
    while not best["flag"]:
        iteration += 1
        stall += 1
        epsilon = 0.85 if stall > 25 else 0.45
        sticky = 12 if stall > 25 else 6
        found = best["max_x"]
        for _ in range(4):
            drop_choices = [0, 4, 8, 16, 32, 64, 128] if stall <= 25 else [0, 8, 16, 32, 64, 128, 256]
            drop = int(np.random.choice(drop_choices))
            keep = max(0, len(best["actions"]) - drop)
            episode = run_episode(model, best["actions"][:keep], explore_steps=160, epsilon=epsilon, sticky=sticky)
            if episode["max_x"] > found:
                found = episode["max_x"]
            if episode["max_x"] > best["max_x"] or episode["flag"]:
                best = episode
                stall = 0
                obs = collect_obs(best["actions"])
                loss = imitate(model, best["actions"][: len(obs)], obs)
                save_best(model, best, best["max_x"])
                print(
                    f"improve iter={iteration} x={best['max_x']} flag={best['flag']} "
                    f"len={len(best['actions'])} loss={loss:.3f}",
                    flush=True,
                )
                if best["flag"]:
                    break
        row = {"iter": iteration, "best_x": int(best["max_x"]), "round_max_x": int(found), "stall": stall, "flag": bool(best["flag"])}
        history.append(row)
        (OUT / "flag_progress.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"iter={iteration} best_x={best['max_x']} round_max={found} stall={stall} flag={best['flag']}",
            flush=True,
        )

    model.save(str(OUT / "ppo-mario-flag"))
    video = record_flag_video(best["actions"], OUT / "mario-1-1-flag.mp4")
    payload = {
        "status": "PASS" if video["video_flag"] or best["flag"] else "FAIL",
        "started": started,
        "ended": now_iso(),
        "metrics": {"flag": bool(video["video_flag"] or best["flag"]), "best_x": int(best["max_x"]), "iters": iteration, **video},
        "notes": "replay the farthest action sequence, then explore from that point",
    }
    write_json(OUT / "result.json", payload)
    print("FLAG", payload["metrics"], flush=True)


if __name__ == "__main__":
    main()
