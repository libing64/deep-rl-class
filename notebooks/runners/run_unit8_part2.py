#!/usr/bin/env python3
"""Unit 8 part2: Sample Factory VizDoom health gathering (4M env steps)."""
from __future__ import annotations

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import UnitRun


def main():
    with UnitRun("8.2", "unit8_part2") as run:
        try:
            import torch
            _orig_load = torch.load
            def _load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_load(*args, **kwargs)
            torch.load = _load

            from sample_factory.algo.utils.context import global_model_factory
            from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
            from sample_factory.enjoy import enjoy
            from sample_factory.envs.env_utils import register_env
            from sample_factory.train import run_rl
            from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
            from sf_examples.vizdoom.doom.doom_params import (
                add_doom_env_args,
                doom_override_defaults,
            )
            from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec

            def register_vizdoom_envs():
                for env_spec in DOOM_ENVS:
                    make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
                    register_env(env_spec.name, make_env_func)

            def register_vizdoom_models():
                global_model_factory().register_encoder_factory(make_vizdoom_encoder)

            def register_vizdoom_components():
                register_vizdoom_envs()
                register_vizdoom_models()

            def parse_vizdoom_cfg(argv=None, evaluation=False):
                parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
                add_doom_env_args(parser)
                doom_override_defaults(parser)
                return parse_full_cfg(parser, argv)

            train_dir = str(run.out / "train_dir")
            Path(train_dir).mkdir(parents=True, exist_ok=True)

            register_vizdoom_components()
            env = "doom_health_gathering_supreme"
            cfg = parse_vizdoom_cfg(
                argv=[
                    f"--env={env}",
                    "--num_workers=8",
                    "--num_envs_per_worker=4",
                    "--train_for_env_steps=4000000",
                    f"--train_dir={train_dir}",
                    "--experiment=doom_health_gathering_supreme",
                ]
            )
            status = run_rl(cfg)

            cfg_eval = parse_vizdoom_cfg(
                argv=[
                    f"--env={env}",
                    "--num_workers=1",
                    "--save_video",
                    "--no_render",
                    "--max_num_episodes=10",
                    f"--train_dir={train_dir}",
                    "--experiment=doom_health_gathering_supreme",
                ],
                evaluation=True,
            )
            enjoy_status = enjoy(cfg_eval)

            run.success(
                {
                    "train_status": status,
                    "enjoy_status": enjoy_status,
                    "env_steps": 4_000_000,
                },
                notes=f"train_dir={train_dir}",
            )
            print("Unit8.2 PASS", status, enjoy_status)
        except Exception as e:
            run.fail(e)
            raise


if __name__ == "__main__":
    main()
