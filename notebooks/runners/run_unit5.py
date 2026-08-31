#!/usr/bin/env python3
"""Unit 5: ML-Agents SnowballTarget (local, no Hub push)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun, ROOT

UNIT5 = ROOT / "results" / "unit5"
ENVS = UNIT5 / "training-envs-executables" / "linux"
CFG_DIR = UNIT5 / "config" / "ppo"
RESULTS_DIR = UNIT5 / "results"

SNOWBALL_CFG = """behaviors:
  SnowballTarget:
    trainer_type: ppo
    summary_freq: 10000
    keep_checkpoints: 10
    checkpoint_interval: 50000
    max_steps: 200000
    time_horizon: 64
    threaded: false
    hyperparameters:
      learning_rate: 0.0003
      learning_rate_schedule: linear
      batch_size: 128
      buffer_size: 2048
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: false
      hidden_units: 256
      num_layers: 2
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
"""


def find_exe() -> Path:
    preferred = ENVS / "SnowballTarget" / "SnowballTarget.x86_64"
    if preferred.is_file():
        preferred.chmod(preferred.stat().st_mode | 0o755)
        return preferred
    candidates = [
        p
        for p in ENVS.rglob("SnowballTarget*")
        if p.is_file() and "Data" not in str(p) and p.suffix in {".x86_64", ""}
    ]
    if not candidates:
        raise FileNotFoundError(f"SnowballTarget executable not under {ENVS}")
    exe = candidates[0]
    exe.chmod(exe.stat().st_mode | 0o755)
    return exe


def main():
    with UnitRun("5", "unit5") as run:
        try:
            exe = find_exe()
            CFG_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            cfg_path = CFG_DIR / "SnowballTarget.yaml"
            cfg_path.write_text(SNOWBALL_CFG)

            mlagents = "/home/libing/.conda/envs/deep-rl-class/bin/mlagents-learn"
            if not Path(mlagents).is_file():
                mlagents = str(Path(sys.executable).parent / "mlagents-learn")
            cmd = [
                mlagents,
                str(cfg_path),
                f"--env={exe}",
                "--run-id=SnowballTarget1",
                "--no-graphics",
                f"--results-dir={RESULTS_DIR}",
                "--force",
            ]
            env = os.environ.copy()
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, env=env)

            run.success(
                {"run_id": "SnowballTarget1", "max_steps": 200000, "exe": str(exe)},
                notes=f"results={RESULTS_DIR}",
            )
            print("Unit5 PASS")
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
