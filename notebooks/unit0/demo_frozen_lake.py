#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Script for Trained Frozen Lake Agent
训练好的冰湖智能体演示脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from frozen_lake_env import FrozenLakeWrapper
from q_learning_frozen_lake import QLearningAgent
import time


def compare_agents(map_size="4x4", slippery=True, model_path=None, num_episodes=20):
    """
    比较训练前后智能体的表现

    参数:
        map_size: 地图大小
        slippery: 是否光滑
        model_path: 训练好的模型路径
        num_episodes: 对比回合数
    """
    print("🎯 Frozen Lake Agent Performance Comparison")
    print("=" * 60)

    # 创建环境
    env = FrozenLakeWrapper(map_size, slippery)

    # 加载训练好的智能体
    trained_agent = QLearningAgent(env.n_states, env.n_actions)
    if model_path:
        model_loaded = trained_agent.load_model(model_path)
    else:
        # 尝试自动查找模型
        slippery_str = "slippery" if slippery else "non_slippery"
        default_path = f"frozen_lake_{map_size}_{slippery_str}_agent.pkl"
        model_loaded = trained_agent.load_model(default_path)

    if not model_loaded:
        print("❌ No trained model found.")
        print("Please run training first or specify model path with --model")
        return

    # 设置训练好的智能体为测试模式（无探索）
    trained_agent.epsilon = 0.0

    # 创建随机智能体作为对比
    random_agent = QLearningAgent(env.n_states, env.n_actions)
    random_agent.epsilon = 1.0  # 总是随机动作

    results = {
        'trained': {'rewards': [], 'steps': [], 'paths': []},
        'random': {'rewards': [], 'steps': [], 'paths': []}
    }

    print(f"Running {num_episodes} episodes for each agent...")
    print("-" * 60)

    # 测试训练好的智能体
    print("🤖 Testing Trained Agent:")
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        path = [state]
        done = False

        while not done:
            action = trained_agent.get_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            path.append(state)
            done = terminated or truncated

            # 防止无限循环
            if steps >= 100:
                done = True

        results['trained']['rewards'].append(episode_reward)
        results['trained']['steps'].append(steps)
        results['trained']['paths'].append(path)

        status = "SUCCESS ✅" if episode_reward == 1.0 else "FAILED ❌"
        print(f"  Episode {episode + 1:2d}: {status} | Steps: {steps:2d} | Path: {path}")

    print("\n🎲 Testing Random Agent:")
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        path = [state]
        done = False

        while not done:
            action = random_agent.get_action(state, training=True)  # 总是随机
            state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            path.append(state)
            done = terminated or truncated

            # 防止无限循环
            if steps >= 100:
                done = True

        results['random']['rewards'].append(episode_reward)
        results['random']['steps'].append(steps)
        results['random']['paths'].append(path)

        status = "SUCCESS ✅" if episode_reward == 1.0 else "FAILED ❌"
        print(f"  Episode {episode + 1:2d}: {status} | Steps: {steps:2d} | Path: {path}")

    env.close()

    # 分析结果
    analyze_results(results, map_size, slippery)

    # 可视化结果
    plot_comparison(results, map_size, slippery)


def analyze_results(results, map_size, slippery):
    """分析比较结果"""
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)

    for agent_name, data in results.items():
        rewards = data['rewards']
        steps = data['steps']

        success_count = sum(1 for r in rewards if r == 1.0)
        success_rate = success_count / len(rewards) * 100

        print(f"\n{agent_name.upper()} AGENT ({map_size}, {'slippery' if slippery else 'non-slippery'}):")
        print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{len(rewards)})")
        print(f"  Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"  Min Steps: {np.min(steps)}")
        print(f"  Max Steps: {np.max(steps)}")

        # 路径长度分析
        path_lengths = [len(path) - 1 for path in data['paths']]  # -1 因为包含起点
        print(f"  Average Path Length: {np.mean(path_lengths):.1f}")

    # 计算改进程度
    trained_success = sum(1 for r in results['trained']['rewards'] if r == 1.0) / len(results['trained']['rewards'])
    random_success = sum(1 for r in results['random']['rewards'] if r == 1.0) / len(results['random']['rewards'])

    if random_success > 0:
        success_improvement = ((trained_success - random_success) / random_success) * 100
        print("\n🎯 IMPROVEMENT:")
        print(f"  Success Rate Improvement: {success_improvement:+.1f}%")
    else:
        print("\n🎯 RESULT:")
        print(f"  Random agent success rate: 0.0%")
        print(f"  Trained agent success rate: {trained_success*100:.1f}%")

    if trained_success > 0.5:
        print("  ✅ Trained agent performs VERY WELL!")
    elif trained_success > 0.2:
        print("  🟡 Trained agent performs reasonably well.")
    else:
        print("  ❌ Trained agent needs more training.")


def plot_comparison(results, map_size, slippery):
    """绘制对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    slippery_str = "slippery" if slippery else "non-slippery"
    fig.suptitle(f'Frozen Lake {map_size} {slippery_str.capitalize()} - Agent Comparison',
                 fontsize=16, fontweight='bold')

    # 成功率对比
    trained_success = sum(1 for r in results['trained']['rewards'] if r == 1.0) / len(results['trained']['rewards'])
    random_success = sum(1 for r in results['random']['rewards'] if r == 1.0) / len(results['random']['rewards'])

    bars = axes[0, 0].bar(['Trained Agent', 'Random Agent'],
                          [trained_success * 100, random_success * 100],
                          color=['blue', 'red'], alpha=0.7, width=0.6)
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       '.1f', ha='center', va='bottom', fontweight='bold')

    # 步数分布
    axes[0, 1].hist(results['trained']['steps'], bins=10, alpha=0.7,
                    label='Trained', color='blue')
    axes[0, 1].hist(results['random']['steps'], bins=10, alpha=0.7,
                    label='Random', color='red')
    axes[0, 1].set_xlabel('Steps per Episode')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Steps Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 奖励曲线
    episodes = range(1, len(results['trained']['rewards']) + 1)
    axes[1, 0].plot(episodes, np.cumsum(results['trained']['rewards']) / episodes,
                    'b-', linewidth=2, label='Trained (Cumulative)')
    axes[1, 0].plot(episodes, np.cumsum(results['random']['rewards']) / episodes,
                    'r-', linewidth=2, label='Random (Cumulative)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].set_title('Learning Curve (Cumulative Average)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # 路径长度对比
    trained_path_lengths = [len(path) - 1 for path in results['trained']['paths']]
    random_path_lengths = [len(path) - 1 for path in results['random']['paths']]

    axes[1, 1].boxplot([trained_path_lengths, random_path_lengths],
                       labels=['Trained Agent', 'Random Agent'])
    axes[1, 1].set_ylabel('Path Length')
    axes[1, 1].set_title('Path Length Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图表
    slippery_str = "slippery" if slippery else "non_slippery"
    filename = f"frozen_lake_{map_size}_{slippery_str}_demo_results.png"
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\nDemo plot saved as: {filename}")

    plt.show()


def interactive_demo(model_path=None, map_size="4x4", slippery=True, max_steps=50):
    """交互式演示"""
    print("🎮 Interactive Frozen Lake Demo")
    print("=" * 40)

    env = FrozenLakeWrapper(map_size, slippery)

    # 加载训练好的智能体
    agent = QLearningAgent(env.n_states, env.n_actions)
    if model_path:
        if not agent.load_model(model_path):
            print("❌ No trained model found.")
            return
    else:
        # 尝试自动查找模型
        slippery_str = "slippery" if slippery else "non_slippery"
        default_path = f"frozen_lake_{map_size}_{slippery_str}_agent.pkl"
        if not agent.load_model(default_path):
            print("❌ No trained model found.")
            print("Please run training first or specify model path with --model")
            return

    agent.epsilon = 0.0  # 无探索

    print("🤖 Trained agent is ready!")
    print("🎯 Watch the agent navigate the frozen lake...")
    print("Map legend: 🏠=Start, 🏆=Goal, 🕳️=Hole, 🧊=Ice, 🤖=Agent")
    print("-" * 60)

    # 运行演示
    state, info = env.reset()
    total_reward = 0
    steps = 0
    path = [state]

    env.render_ascii()
    print(f"Start position: State {state}")
    print("-" * 60)

    while steps < max_steps:
        # 智能体选择动作
        action = agent.get_action(state, training=False)
        action_names = ['LEFT ←', 'DOWN ↓', 'RIGHT →', 'UP ↑']

        print(f"Step {steps + 1}: Choosing action {action_names[action]}")

        # 执行动作
        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        path.append(state)

        # 显示结果
        env.render_ascii()
        print(f"  New position: State {state}, Reward: {reward}")
        print(f"  Total reward: {total_reward}")
        print("-" * 60)

        if terminated or truncated:
            if reward == 1.0:
                print("🎉 SUCCESS! Agent reached the goal!")
            elif terminated:
                print("💥 FAILED! Agent fell into a hole!")
            else:
                print("⏰ Episode truncated (max steps reached)")
            break

        time.sleep(0.5)  # 短暂暂停以观察

    print("\n🎯 DEMO COMPLETED!")
    print(f"Final reward: {total_reward}")
    print(f"Total steps: {steps}")
    print(f"Path taken: {path}")

    if total_reward == 1.0:
        print("✅ Agent successfully navigated the frozen lake!")
    else:
        print("❌ Agent failed to reach the goal.")

    env.close()


def show_policy_demo(model_path=None, map_size="4x4", slippery=True):
    """显示学习到的策略"""
    print("🧠 Learned Policy Visualization")
    print("=" * 40)

    env = FrozenLakeWrapper(map_size, slippery)

    # 加载训练好的智能体
    agent = QLearningAgent(env.n_states, env.n_actions)
    if model_path:
        if not agent.load_model(model_path):
            return
    else:
        slippery_str = "slippery" if slippery else "non_slippery"
        default_path = f"frozen_lake_{map_size}_{slippery_str}_agent.pkl"
        if not agent.load_model(default_path):
            return

    policy = agent.get_optimal_policy()
    state_values = agent.get_state_values()

    print(f"Environment: {map_size} {'slippery' if slippery else 'non-slippery'}")
    print("\nLearned Policy (Optimal Actions):")
    print("Legend: ←=Left, ↓=Down, →=Right, ↑=Up")
    print("-" * (env.map_size * 4 + 1))

    action_symbols = ['←', '↓', '→', '↑']
    for i in range(env.map_size):
        for j in range(env.map_size):
            state = i * env.map_size + j
            cell_type = env.desc[i, j]

            if cell_type == 'S':
                symbol = '🏠'  # 起点
            elif cell_type == 'G':
                symbol = '🏆'  # 目标
            elif cell_type == 'H':
                symbol = '🕳️'  # 洞
            else:
                symbol = action_symbols[policy[state]]

            print(f" {symbol} ", end="")
        print()
    print("-" * (env.map_size * 4 + 1))

    print("\nState Values (Q-max):")
    print("-" * (env.map_size * 8 + 1))
    for i in range(env.map_size):
        for j in range(env.map_size):
            state = i * env.map_size + j
            value = state_values[state]
            print("5.2f", end=" ")
        print()
    print("-" * (env.map_size * 8 + 1))

    # 统计信息
    stats = agent.get_stats()
    print("\n📊 Model Statistics:")
    print(f"  Q-table shape: {stats['q_table_shape']}")
    print(f"  Max Q-value: {stats['max_q_value']:.3f}")
    print(f"  Min Q-value: {stats['min_q_value']:.3f}")
    print(f"  Explored states: {stats['explored_states']}/{stats['q_table_shape'][0]}")
    print(".1f")
    env.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Demo trained Frozen Lake agent')
    parser.add_argument('--map', type=str, default='4x4', choices=['4x4', '8x8'],
                       help='Map size')
    parser.add_argument('--slippery', action='store_true', default=True,
                       help='Use slippery ice (default: True)')
    parser.add_argument('--no-slippery', action='store_true',
                       help='Use non-slippery ice')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of comparison episodes')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive demo')
    parser.add_argument('--policy', action='store_true',
                       help='Show learned policy visualization')

    args = parser.parse_args()

    if args.no_slippery:
        args.slippery = False

    if args.policy:
        show_policy_demo(args.model, args.map, args.slippery)
    elif args.interactive:
        interactive_demo(args.model, args.map, args.slippery)
    else:
        compare_agents(args.map, args.slippery, args.model, args.episodes)


if __name__ == "__main__":
    main()
