python -m rl_zoo3.train --algo ppo --env CartPole-v1 --eval-freq 10000 --save-freq 50000

python -m rl_zoo3.enjoy --algo ppo --env CartPole-v1 --folder ./logs/



<!-- python -m rl_zoo3.enjoy --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000 -->


<!-- python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -->


python -m rl_zoo3.train --algo ppo --env SpaceInvadersNoFrameskip-v4 --eval-freq 10000 --save-freq 50000

python -m rl_zoo3.enjoy --algo ppo --env SpaceInvadersNoFrameskip-v4 --folder ./logs/
