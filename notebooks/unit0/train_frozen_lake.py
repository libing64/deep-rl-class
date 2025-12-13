#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Frozen Lake with Q-Learning
冰湖游戏Q-Learning训练脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_env import FrozenLakeWrapper
from q_learning_frozen_lake import QLearningAgent, train_q_learning
import time


def plot_training_results(rewards_history, steps_history, epsilon_history, map_size):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Frozen Lake {map_size}x{map_size} Q-Learning Training Results', fontsize=16, fontweight='bold')

    episodes = range(1, len(rewards_history) + 1)

    # 奖励曲线
    axes[0, 0].plot(episodes, rewards_history, 'b-', alpha=0.7, linewidth=1)
    if len(rewards_history) > 50:
        moving_avg = np.convolve(rewards_history, np.ones(50)/50, mode='same')
        axes[0, 0].plot(episodes, moving_avg,
                        'r-', linewidth=2, label='Moving Average (50)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (0 or 1)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.1, 1.1)

    # 成功率曲线
    window_size = min(100, len(rewards_history)//10)
    if window_size > 0:
        success_rates = []
        for i in range(len(rewards_history)):
            start = max(0, i - window_size + 1)
            success_rate = np.mean(rewards_history[start:i+1])
            success_rates.append(success_rate)

        axes[0, 1].plot(episodes, success_rates, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_title(f'Success Rate (Window: {window_size})')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

    # 探索率曲线
    axes[1, 0].plot(episodes, epsilon_history, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon)')
    axes[1, 0].grid(True, alpha=0.3)

    # 步数曲线
    axes[1, 1].plot(episodes, steps_history, 'm-', alpha=0.7, linewidth=1)
    if len(steps_history) > 50:
        moving_avg = np.convolve(steps_history, np.ones(50)/50, mode='same')
        axes[1, 1].plot(episodes, moving_avg,
                        'purple', linewidth=2, label='Moving Average (50)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps per Episode')
    axes[1, 1].set_title('Episode Length')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    filename = f"frozen_lake_{map_size}x{map_size}_training_results.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Training plot saved as: {filename}")

    plt.show()


def plot_policy_heatmap(agent, env, map_size):
    """绘制策略热力图"""
    policy = agent.get_optimal_policy()
    state_values = agent.get_state_values()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Frozen Lake {map_size}x{map_size} Learned Policy & Values', fontsize=14, fontweight='bold')

    # 策略热力图
    policy_grid = policy.reshape(map_size, map_size)
    action_symbols = ['←', '↓', '→', '↑']  # LEFT, DOWN, RIGHT, UP

    # 创建策略网格
    policy_display = np.empty((map_size, map_size), dtype='<U5')
    for i in range(map_size):
        for j in range(map_size):
            state = i * map_size + j
            cell_type = env.desc[i, j]
            if cell_type == 'S':
                policy_display[i, j] = '🏠'  # 起点
            elif cell_type == 'G':
                policy_display[i, j] = '🏆'  # 目标
            elif cell_type == 'H':
                policy_display[i, j] = '🕳️'  # 洞
            else:
                policy_display[i, j] = action_symbols[policy[state]]

    axes[0].imshow(np.zeros((map_size, map_size)), cmap='Blues', alpha=0.3)
    for i in range(map_size):
        for j in range(map_size):
            axes[0].text(j, i, policy_display[i, j], ha='center', va='center',
                        fontsize=12 if policy_display[i, j] in ['🏠', '🏆', '🕳️'] else 16,
                        fontweight='bold')
    axes[0].set_title('Learned Policy')
    axes[0].set_xticks(range(map_size))
    axes[0].set_yticks(range(map_size))
    axes[0].set_xticklabels(range(map_size))
    axes[0].set_yticklabels(range(map_size))
    axes[0].grid(True, alpha=0.3)

    # 状态值热力图
    values_grid = state_values.reshape(map_size, map_size)
    im = axes[1].imshow(values_grid, cmap='RdYlGn', alpha=0.8)
    axes[1].set_title('State Values (Q-max)')
    axes[1].set_xticks(range(map_size))
    axes[1].set_yticks(range(map_size))
    axes[1].set_xticklabels(range(map_size))
    axes[1].set_yticklabels(range(map_size))
    axes[1].grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
    cbar.set_label('State Value')

    plt.tight_layout()

    # 保存图表
    filename = f"frozen_lake_{map_size}x{map_size}_policy_values.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Policy plot saved as: {filename}")

    plt.show()


def evaluate_agent(agent, env, num_episodes=100):
    """评估训练好的智能体"""
    print(f"\n🔍 Evaluating trained agent for {num_episodes} episodes...")

    success_count = 0
    total_steps = 0
    rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # 使用训练好的策略（无探索）
            action = agent.get_action(state, training=False)

            state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            done = terminated or truncated

            # 防止无限循环
            if steps >= 100:
                done = True

        rewards.append(episode_reward)
        total_steps += steps

        if episode_reward == 1.0:
            success_count += 1

    success_rate = success_count / num_episodes * 100
    avg_steps = total_steps / num_episodes

    print("\n📊 Evaluation Results:")
    print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"  Average Reward: {np.mean(rewards):.3f}")

    return success_rate, avg_steps, rewards


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Q-Learning agent for Frozen Lake')
    parser.add_argument('--map', type=str, default='4x4', choices=['4x4', '8x8'],
                       help='Map size (4x4 or 8x8)')
    parser.add_argument('--slippery', action='store_true', default=True,
                       help='Use slippery ice (default: True)')
    parser.add_argument('--no-slippery', action='store_true',
                       help='Use non-slippery ice')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save the trained model')
    parser.add_argument('--evaluate', type=int, default=100,
                       help='Number of evaluation episodes (0 to skip)')

    args = parser.parse_args()

    # 处理参数
    if args.no_slippery:
        args.slippery = False

    if args.save_path is None:
        slippery_str = "slippery" if args.slippery else "non_slippery"
        args.save_path = f"frozen_lake_{args.map}_{slippery_str}_agent.pkl"

    # 创建环境
    env = FrozenLakeWrapper(args.map, args.slippery)
    print(f"Environment: {env}")
    print(f"Map size: {args.map}")
    print(f"Slippery: {args.slippery}")

    # 训练智能体
    start_time = time.time()
    agent, rewards_history, steps_history, epsilon_history = train_q_learning(
        env, args.episodes, args.save_path
    )

    training_time = time.time() - start_time
    print(".1f")
    # 评估智能体
    if args.evaluate > 0:
        success_rate, avg_steps, eval_rewards = evaluate_agent(agent, env, args.evaluate)

        # 保存评估结果
        eval_results = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'eval_rewards': eval_rewards,
            'training_episodes': args.episodes,
            'map_size': args.map,
            'slippery': args.slippery,
            'training_time': training_time
        }

        eval_filename = args.save_path.replace('.pkl', '_eval_results.pkl')
        import pickle
        with open(eval_filename, 'wb') as f:
            pickle.dump(eval_results, f)
        print(f"Evaluation results saved to: {eval_filename}")

    # 绘制训练结果
    plot_training_results(rewards_history, steps_history, epsilon_history,
                         int(args.map.split('x')[0]))

    # 绘制策略和值函数
    plot_policy_heatmap(agent, env, int(args.map.split('x')[0]))

    env.close()

    print("\n✅ Training and evaluation completed!")
    print(f"📁 Model saved as: {args.save_path}")
    print("📊 Check the generated PNG files for detailed analysis!")


if __name__ == "__main__":
    main()
