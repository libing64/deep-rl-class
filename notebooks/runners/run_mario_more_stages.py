#!/usr/bin/env python3
"""Clear Super Mario Bros. stages after 1-1.

Replays the action sequence that already finishes 1-1, then searches only past
the farthest point reached. Progress counts the stage first and x position
second, so a new stage outranks any position in the previous one.
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

ROOT = Path(__file__).resolve().parents[1] / "results" / "mario"
OUT = ROOT / "worlds"
# 8 worlds × 4 stages. Done once play has moved past 8-4.
TOTAL_STAGES = 32


def stage_index(world: int, stage: int) -> int:
    return (int(world) - 1) * 4 + (int(stage) - 1)


def score_of(world: int, stage: int, x: int) -> int:
    return stage_index(world, stage) * 100_000 + int(x)


def _is_clear(world: int, stage: int, flag: bool) -> bool:
    if int(world) > 8 or stage_index(world, stage) >= TOTAL_STAGES:
        return True
    return int(world) == 8 and int(stage) == 4 and bool(flag)


def campaign_done(episode) -> bool:
    return bool(episode.get("game_clear")) or stage_index(episode["world"], episode["stage"]) >= TOTAL_STAGES


def pack(actions, worlds, stages, xs, reward, ys=None, flags=None) -> dict:
    actions = [int(a) for a in actions]
    worlds = [int(w) for w in worlds]
    stages = [int(s) for s in stages]
    xs = [int(x) for x in xs]
    ys = [int(y) for y in ys] if ys is not None else [100] * len(xs)
    flags = [bool(f) for f in flags] if flags is not None else [False] * len(xs)
    game_clear = False
    if xs:
        peak = 0
        best_score = -1
        for i in range(len(xs)):
            if _is_clear(worlds[i], stages[i], flags[i]):
                peak = i
                game_clear = True
                break
            sc = score_of(worlds[i], stages[i], xs[i])
            advanced = stage_index(worlds[i], stages[i]) > stage_index(worlds[peak], stages[peak])
            stable = (
                i >= 2
                and 40 <= ys[i] <= 200
                and abs(ys[i] - ys[i - 1]) <= 4
                and abs(ys[i - 1] - ys[i - 2]) <= 4
            )
            # Falling into a pit still increases x. Only keep a foothold or a new stage.
            if (stable or advanced) and sc > best_score:
                best_score = sc
                peak = i
        actions = actions[: peak + 1]
        worlds = worlds[: peak + 1]
        stages = stages[: peak + 1]
        xs = xs[: peak + 1]
        ys = ys[: peak + 1]
        flags = flags[: peak + 1]
        game_clear = game_clear or any(_is_clear(w, s, f) for w, s, f in zip(worlds, stages, flags))
    world = worlds[-1] if worlds else 1
    stage = stages[-1] if stages else 1
    x = xs[-1] if xs else 0
    return {
        "actions": np.asarray(actions, dtype=np.int64),
        "worlds": np.asarray(worlds, dtype=np.int32),
        "stages": np.asarray(stages, dtype=np.int32),
        "xs": np.asarray(xs, dtype=np.int32),
        "ys": np.asarray(ys, dtype=np.int32),
        "world": world,
        "stage": stage,
        "max_x": x,
        "score": score_of(world, stage, x) if xs else 0,
        "game_clear": game_clear,
        "reward": float(reward),
    }


def explore_action(env) -> int:
    roll = np.random.rand()
    if roll < 0.55:
        return 4  # run + jump
    if roll < 0.75:
        return 2  # jump right
    if roll < 0.90:
        return 3  # run right
    return int(env.action_space.sample())


def run_episode(model, prefix, explore_steps: int, epsilon: float, sticky: int) -> dict:
    env = MarioEnv(target=None)
    obs, _ = env.reset()
    actions, worlds, stages, xs, ys, flags = [], [], [], [], [], []
    reward = 0.0
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
                sticky_action = explore_action(env)
                sticky_left = max(0, sticky - 1)
                action = sticky_action
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            obs, step_reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            worlds.append(int(info.get("world", 1)))
            stages.append(int(info.get("stage", 1)))
            xs.append(int(info.get("x_pos", 0)))
            ys.append(int(info.get("y_pos", 100)))
            flags.append(bool(info.get("flag_get", False)))
            reward += float(step_reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    return pack(actions, worlds, stages, xs, reward, ys, flags)


def _record(bucket, action, info) -> None:
    bucket["actions"].append(int(action))
    bucket["worlds"].append(int(info.get("world", 1)))
    bucket["stages"].append(int(info.get("stage", 1)))
    bucket["xs"].append(int(info.get("x_pos", 0)))
    bucket["ys"].append(int(info.get("y_pos", 100)))
    bucket["flags"].append(bool(info.get("flag_get", False)))


def jump_search(prefix, min_score: int = 0, wide: bool = False) -> dict | None:
    """Try timed jumps from the platforms just before the frontier.

    One replay walks up to the frontier. Each branch point is restored in the
    emulator, so the search does not replay world 1-1 for every jump.
    """
    prefix = [int(a) for a in prefix]
    drops = list(range(0, 21, 2))
    if wide:
        drops += list(range(28, 61, 8))
    max_drop = max(drops)
    env = MarioEnv(target=None)
    best_ep = None
    try:
        env.reset()
        base = {"actions": [], "worlds": [], "stages": [], "xs": [], "ys": [], "flags": []}
        info = {"world": 1, "stage": 1, "x_pos": 0, "y_pos": 0}
        cut = max(0, len(prefix) - max_drop)
        for action in prefix[:cut]:
            _, _, terminated, truncated, info = env.step(action)
            _record(base, action, info)
            if terminated or truncated:
                return None
        nes = env._env.env
        extra = 0
        remaining = prefix[cut:]
        for drop in sorted(drops, reverse=True):
            want = max_drop - drop
            while extra < want and extra < len(remaining):
                action = remaining[extra]
                extra += 1
                _, _, terminated, truncated, info = env.step(action)
                _record(base, action, info)
                if terminated or truncated:
                    break
            nes._backup()
            frames = env._frames.copy()
            print(
                f"jump-from {info.get('world', 1)}-{info.get('stage', 1)} "
                f"x={info.get('x_pos', 0)} y={info.get('y_pos', 0)} drop={drop}",
                flush=True,
            )
            tails = []
            for wait in (0, 2, 4, 8):
                for hold in (8, 14, 20, 28):
                    tails.append([1] * wait + [4] * hold + [1] * 30)
                    tails.append([3] * min(wait, 6) + [4] * hold + [1] * 24)
            for tail in tails:
                nes._restore()
                nes.done = False
                env._frames[:] = frames
                actions = list(base["actions"])
                worlds = list(base["worlds"])
                stages = list(base["stages"])
                xs = list(base["xs"])
                ys = list(base["ys"])
                flags = list(base["flags"])
                for action in tail:
                    _, _, terminated, truncated, info = env.step(int(action))
                    actions.append(int(action))
                    worlds.append(int(info.get("world", 1)))
                    stages.append(int(info.get("stage", 1)))
                    xs.append(int(info.get("x_pos", 0)))
                    ys.append(int(info.get("y_pos", 100)))
                    flags.append(bool(info.get("flag_get", False)))
                    if terminated or truncated or _is_clear(info.get("world", 1), info.get("stage", 1), info.get("flag_get", False)):
                        break
                episode = pack(actions, worlds, stages, xs, 0.0, ys, flags)
                if best_ep is None or episode["score"] > best_ep["score"]:
                    best_ep = episode
                    if best_ep.get("game_clear") or best_ep["score"] >= min_score + 60:
                        nes._restore()
                        nes.done = False
                        env._frames[:] = frames
                        print(f"jump-best {describe(best_ep)}", flush=True)
                        return best_ep
            nes._restore()
            nes.done = False
            env._frames[:] = frames
        if best_ep is not None:
            print(f"jump-best {describe(best_ep)}", flush=True)
        return best_ep
    finally:
        env.close()


def imitate(model, actions_src, obs_src) -> float:
    if len(actions_src) < 8:
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
    env = MarioEnv(target=None)
    obs, _ = env.reset()
    stored = []
    try:
        for action in actions:
            stored.append(np.array(obs, copy=True))
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
    finally:
        env.close()
    if not stored:
        return np.zeros((0, 84, 84, 4), dtype=np.uint8)
    return np.stack(stored)


def save_best(model, episode) -> None:
    np.savez(
        OUT / "best_actions.npz",
        actions=episode["actions"],
        worlds=episode["worlds"],
        stages=episode["stages"],
        xs=episode["xs"],
        ys=episode["ys"],
        score=np.int32(episode["score"]),
    )
    tag = f"w{episode['world']}_s{episode['stage']}_x{episode['max_x']}"
    model.save(str(OUT / "best_model"))
    model.save(str(OUT / f"best_model_{tag}"))


def record_video(actions, path: Path) -> dict:
    import imageio.v2 as imageio

    env = _make_nes(target=None)
    obs = env.reset()
    frames = []
    world, stage, max_x = 1, 1, 0
    reward = 0.0
    seen = []
    raw_i = 0
    try:
        for action in actions:
            done = False
            for _ in range(4):
                obs, step_reward, done, info = env.step(int(action))
                raw_i += 1
                if raw_i % 8 == 0 and len(frames) < 12000:
                    frames.append(np.array(obs, copy=True))
                reward += float(step_reward)
                world = int(info.get("world", world))
                stage = int(info.get("stage", stage))
                max_x = int(info.get("x_pos", 0))
                mark = f"{world}-{stage}"
                if not seen or seen[-1] != mark:
                    seen.append(mark)
                if done:
                    break
            if done:
                break
    finally:
        env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=30)
    return {
        "video_world": world,
        "video_stage": stage,
        "video_x_pos": max_x,
        "video_reward": reward,
        "video_frames": len(frames),
        "video_stages": seen,
        "video": str(path),
    }


def load_episode(path: Path):
    saved = np.load(path)
    if "worlds" not in saved:
        return None
    ys = saved["ys"] if "ys" in saved else None
    return pack(saved["actions"], saved["worlds"], saved["stages"], saved["xs"], 0.0, ys)


def describe(episode) -> str:
    return f"{episode['world']}-{episode['stage']} x={episode['max_x']} score={episode['score']} len={len(episode['actions'])}"


def main():
    from stable_baselines3 import PPO

    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "ppo-mario-flag.zip"
    model = PPO.load(str(source), device="cuda")
    started = now_iso()
    print(f"resume {source}", flush=True)

    archive = OUT / "best_actions.npz"
    if archive.exists() and "actions" in np.load(archive):
        previous = np.load(archive)["actions"]
        best = run_episode(model, previous, explore_steps=0, epsilon=0.0, sticky=1)
        save_best(model, best)
        print(f"rebuilt {describe(best)} from {len(previous)} actions", flush=True)
        jumped = jump_search(best["actions"], min_score=int(best["score"]))
        if jumped is not None and jumped["score"] > best["score"] + 24:
            best = jumped
            save_best(model, best)
            print(f"startup jump {describe(best)}", flush=True)
    else:
        seed_actions = np.load(ROOT / "best_actions.npz")["actions"]
        best = run_episode(model, seed_actions, explore_steps=0, epsilon=0.0, sticky=1)
        save_best(model, best)
        print(f"seed {describe(best)}", flush=True)

    history = []
    stall = 0
    iteration = 0
    cleared = stage_index(best["world"], best["stage"])
    while not campaign_done(best):
        iteration += 1
        stall += 1
        found = best["score"]
        found_ep = best
        if stall >= 4:
            jumped = jump_search(best["actions"], min_score=int(best["score"]), wide=True)
            if jumped is not None and jumped["score"] > found:
                found = jumped["score"]
                found_ep = jumped
            same_stage = (
                jumped is not None
                and jumped["world"] == best["world"]
                and jumped["stage"] == best["stage"]
            )
            gained = 0 if jumped is None else jumped["max_x"] - best["max_x"]
            if jumped is not None and jumped["score"] > best["score"] and (not same_stage or gained >= 24):
                best = jumped
                stall = 0
                obs = collect_obs(best["actions"])
                tail = min(512, len(obs))
                loss = imitate(model, best["actions"][-tail:], obs[-tail:])
                save_best(model, best)
                print(f"improve iter={iteration} {describe(best)} loss={loss:.3f}", flush=True)
                now_cleared = stage_index(best["world"], best["stage"])
                if now_cleared > cleared:
                    video = record_video(best["actions"], OUT / f"mario-reach-{best['world']}-{best['stage']}.mp4")
                    print(f"stage {video}", flush=True)
                    cleared = now_cleared
                if campaign_done(best):
                    break
        for epsilon, sticky, explore_steps in (
            (0.45, 10, 220),
            (0.8, 14, 280),
            (1.0, 16, 360),
            (0.6, 12, 420),
        ):
            if stall > 6:
                epsilon = 1.0
            drop = int(np.random.choice([0, 8, 16, 24, 40] if stall <= 6 else [16, 32, 48, 64, 96, 128]))
            keep = max(0, len(best["actions"]) - drop)
            episode = run_episode(model, best["actions"][:keep], explore_steps, epsilon, sticky)
            if episode["score"] > found:
                found = episode["score"]
                found_ep = episode
            same_stage = episode["world"] == best["world"] and episode["stage"] == best["stage"]
            gained = episode["max_x"] - best["max_x"]
            # Ignore one-pixel shuffles on the same platform. They reset the stall counter
            # and never leave room for a real jump.
            if episode["score"] > best["score"] and (not same_stage or gained >= 24):
                best = episode
                stall = 0
                obs = collect_obs(best["actions"])
                tail = min(512, len(obs))
                loss = imitate(model, best["actions"][-tail:], obs[-tail:])
                save_best(model, best)
                print(f"improve iter={iteration} {describe(best)} loss={loss:.3f}", flush=True)
                now_cleared = stage_index(best["world"], best["stage"])
                if now_cleared > cleared:
                    video = record_video(best["actions"], OUT / f"mario-reach-{best['world']}-{best['stage']}.mp4")
                    print(f"stage {video}", flush=True)
                    cleared = now_cleared
                if campaign_done(best):
                    break
        row = {
            "iter": iteration,
            "world": int(best["world"]),
            "stage": int(best["stage"]),
            "x": int(best["max_x"]),
            "score": int(best["score"]),
            "round_score": int(found),
            "stall": stall,
        }
        history.append(row)
        (OUT / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"iter={iteration} best={best['world']}-{best['stage']} x={best['max_x']} "
            f"round={found_ep['world']}-{found_ep['stage']} x={found_ep['max_x']} stall={stall}",
            flush=True,
        )

    video = record_video(best["actions"], OUT / "mario-all-clear.mp4")
    model.save(str(OUT / "ppo-mario-all"))
    payload = {
        "status": "PASS" if campaign_done(best) else "FAIL",
        "started": started,
        "ended": now_iso(),
        "metrics": {
            "world": int(best["world"]),
            "stage": int(best["stage"]),
            "x": int(best["max_x"]),
            "score": int(best["score"]),
            "iters": iteration,
            **video,
        },
        "notes": "replay the 1-1 clear, then search forward through later stages",
    }
    write_json(OUT / "result.json", payload)
    print("CLEARED", payload["metrics"], flush=True)


if __name__ == "__main__":
    main()
