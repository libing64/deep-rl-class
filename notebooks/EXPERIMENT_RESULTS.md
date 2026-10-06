# Deep RL Class — Local Experiment Results

Environment: `conda` env `deep-rl-class`  
Host GPU: NVIDIA GeForce RTX 5060 Ti  
Policy: no Hugging Face Hub push; local artifacts only.

| Unit | Status | Duration | Key metrics | Artifacts |
|------|--------|----------|-------------|-----------|
| 1 PPO LunarLander | PASS | 371.3s | mean_reward=256.5946796237051, std_reward=15.161667885356957 | /home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit1 |
| 2 Q-Learning | PASS | 3.7s | frozenlake_mean=1.0, frozenlake_std=0.0, taxi_mean=7.48, taxi_std=2.744011661782 | results/unit2/ |
| 3 DQN SpaceInvaders | PASS | 1412.6s | enjoy_mean_reward=656.25, n_timesteps=1000000, eval_mean_reward_last=616.0 | results/unit3/ |
| 4 REINFORCE | PASS | 5311.3s | cartpole_mean=156.9, cartpole_std=8.93812060782355, cartpole_last100=155.15, pix | results/unit4/ |
| 5 ML-Agents | PASS | 151.2s | final_mean_reward=25.864, steps=200000 | results/unit5/ |
| 6 A2C Panda | PASS | 1713.0s | PandaReachDense_mean=-0.15928514348343015, PandaReachDense_std=0.058832826602861 | results/unit6/ |
| 8.1 CleanRL PPO | PASS | 11.1s | mean_reward=-231.0672619366657, std_reward=107.20504105552129, total_timesteps=5 | results/unit8_part1/ |
| 8.2 VizDoom SF | PASS | 190.0s | final_avg=25.258, best=25.632, replay=True | /home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit8_part2 |
| mario 1-1 | PASS | 220s search | x_pos=3161, reward=2904, flag=True | results/mario/mario-1-1-flag.mp4 |
| mario worlds | RUNNING | through 2-2 | entered 2-3 x=1031, goal 8-4 | results/mario/worlds/mario-reach-2-3.mp4 |

---

## Unit details

### Unit 1

- **Status**: PASS
- **Started**: 2026-08-31T23:14:21+08:00
- **Ended**: 2026-08-31T23:20:32+08:00
- **Duration (s)**: 371.2920515537262
- **Metrics**: {'mean_reward': 256.5946796237051, 'std_reward': 15.161667885356957}
- **Artifacts**: /home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit1
- **Notes**: LunarLander-v3 (gymnasium); 1M timesteps

### Unit 2

- **Status**: PASS
- **Started**: 2026-08-31T08:54:52+08:00
- **Ended**: 2026-08-31T08:54:55+08:00
- **Duration (s)**: 3.7433981895446777
- **Metrics**: {'frozenlake_mean': 1.0, 'frozenlake_std': 0.0, 'taxi_mean': 7.48, 'taxi_std': 2.744011661782799}
- **Artifacts**: results/unit2/
- **Notes**: 

### Unit 3

- **Status**: PASS
- **Started**: 2026-08-31T09:05:38+08:00
- **Ended**: 2026-08-31T09:29:10+08:00
- **Duration (s)**: 1412.552087545395
- **Metrics**: {'enjoy_mean_reward': 656.25, 'n_timesteps': 1000000, 'eval_mean_reward_last': 616.0, 'enjoy_scores': [390.0, 610.0, 800.0, 825.0]}
- **Artifacts**: results/unit3/
- **Notes**: rl_zoo3 DQN 1e6; see enjoy_stdout.txt

### Unit 4

- **Status**: PASS
- **Started**: 2026-08-31T09:05:38+08:00
- **Ended**: 2026-08-31T10:34:10+08:00
- **Duration (s)**: 5311.295429944992
- **Metrics**: {'cartpole_mean': 156.9, 'cartpole_std': 8.93812060782355, 'cartpole_last100': 155.15, 'pixelcopter_mean': 120.5, 'pixelcopter_std': 107.63387013389419, 'pixelcopter_last100': 83.99}
- **Artifacts**: results/unit4/
- **Notes**: 

### Unit 5

- **Status**: PASS
- **Started**: 2026-08-31T10:30:57+08:00
- **Ended**: 2026-08-31T10:33:28+08:00
- **Duration (s)**: 151.18228483200073
- **Metrics**: {'run_id': 'SnowballTarget1', 'max_steps': 200000, 'exe': '/home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit5/training-envs-executables/linux/SnowballTarget/SnowballTarget.x86_64', 'final_step': 200000, 'final_mean_reward': 25.864}
- **Artifacts**: results/unit5/
- **Notes**: results=/home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit5/results

### Unit 6

- **Status**: PASS
- **Started**: 2026-08-31T10:47:43+08:00
- **Ended**: 2026-08-31T11:16:16+08:00
- **Duration (s)**: 1713.0339262485504
- **Metrics**: {'PandaReachDense_mean': -0.15928514348343015, 'PandaReachDense_std': 0.058832826602861286, 'PandaPickAndPlace_mean': -50.0, 'PandaPickAndPlace_std': 0.0}
- **Artifacts**: results/unit6/
- **Notes**: target Reach >= -3.5

### Unit 8.1

- **Status**: PASS
- **Started**: 2026-08-31T11:16:28+08:00
- **Ended**: 2026-08-31T11:16:39+08:00
- **Duration (s)**: 11.077168464660645
- **Metrics**: {'mean_reward': -231.0672619366657, 'std_reward': 107.20504105552129, 'total_timesteps': 50000}
- **Artifacts**: results/unit8_part1/
- **Notes**: CleanRL PPO gymnasium LunarLander-v3; no Hub

### Unit 8.2

- **Status**: PASS
- **Started**: 2026-08-31T11:18:47+08:00
- **Ended**: 2026-08-31T11:21:57+08:00
- **Duration (s)**: 190.0
- **Metrics**: {'env_steps': 4000000, 'final_avg_episode_reward': 25.258, 'best_avg_episode_reward': 25.632, 'replay_mp4': '/home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit8_part2/train_dir/doom_health_gathering_supreme/replay.mp4', 'replay_exists': True}
- **Artifacts**: /home/libing/source/ml/rl/deep-rl-class/notebooks/results/unit8_part2
- **Notes**: Sample Factory APPO 4M env steps (~46k fps); enjoy+replay after torch.load weights_only patch

### Unit mario

- **Status**: PASS
- **Started**: 2026-10-05T08:54:29+08:00
- **Ended**: 2026-10-05T08:58:09+08:00
- **Duration (s)**: 220
- **Metrics**: {'env': 'SuperMarioBros-1-1', 'best_x': 3161, 'video_x_pos': 3161, 'video_flag': True, 'video_reward': 2904.0, 'video_frames': 4345, 'video': '/home/libing/source/ml/rl/deep-rl-class/notebooks/results/mario/mario-1-1-flag.mp4'}
- **Artifacts**: /home/libing/source/ml/rl/deep-rl-class/notebooks/results/mario
- **Notes**: Reached the flag from the earlier PPO checkpoint by replaying the farthest action sequence and exploring from that point. The first 2M-step PPO run stopped at x=899.

### Unit mario worlds

- **Status**: RUNNING
- **Metrics**: cleared through 2-2, entered 2-3 at x=1031
- **Video**: /home/libing/source/ml/rl/deep-rl-class/notebooks/results/mario/worlds/mario-reach-2-3.mp4
- **Notes**: Full game, normal 3 lives. Water stages 2-2 and 7-2 use swim search. Continues until 8-4.
