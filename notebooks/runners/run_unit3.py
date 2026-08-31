#!/usr/bin/env python3
"""Unit 3: DQN SpaceInvaders via rl-zoo3 (1M timesteps)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun, ROOT

import ale_py  # noqa: F401 — register Atari envs


def main():
    out = ROOT / "results" / "unit3"
    cfg = Path(__file__).resolve().parent / "dqn.yml"
    with UnitRun("3", "unit3") as run:
        try:
            train_cmd = [
                sys.executable,
                "-m",
                "rl_zoo3.train",
                "--algo",
                "dqn",
                "--env",
                "SpaceInvadersNoFrameskip-v4",
                "-f",
                str(out),
                "-c",
                str(cfg),
                "--device",
                "cuda",
            ]
            print("Running:", " ".join(train_cmd), flush=True)
            subprocess.run(train_cmd, check=True, cwd=str(ROOT))

            enjoy_cmd = [
                sys.executable,
                "-m",
                "rl_zoo3.enjoy",
                "--algo",
                "dqn",
                "--env",
                "SpaceInvadersNoFrameskip-v4",
                "--no-render",
                "--n-timesteps",
                "5000",
                "--folder",
                str(out),
                "--device",
                "cuda",
            ]
            print("Running:", " ".join(enjoy_cmd), flush=True)
            proc = subprocess.run(
                enjoy_cmd, check=True, cwd=str(ROOT), capture_output=True, text=True
            )
            (run.out / "enjoy_stdout.txt").write_text(proc.stdout + "\n" + proc.stderr)
            mean = None
            m = re.search(r"Mean reward:\s*([-\d.]+)", proc.stdout + proc.stderr)
            if m:
                mean = float(m.group(1))
            run.success(
                {"enjoy_mean_reward": mean, "n_timesteps": 1_000_000},
                notes="rl_zoo3 DQN 1e6; see enjoy_stdout.txt",
            )
            print("Unit3 PASS", mean)
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
