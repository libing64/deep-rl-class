#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LunarLander-v2 环境仿真示例
使用随机策略运行月球着陆器环境
"""

import gymnasium as gym
import numpy as np


def run_simulation(max_steps=100, num_episodes=1, verbose=True):
    """
    运行 LunarLander-v2 仿真
    
    参数:
        max_steps: 每个回合的最大步数
        num_episodes: 运行的回合数
        verbose: 是否显示详细信息
    """
    # 创建 LunarLander-v2 环境
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    
    if verbose:
        print("=" * 70)
        print("LunarLander-v2 环境仿真示例")
        print("=" * 70)
        print(f"\n环境信息:")
        print(f"  观察空间形状: {env.observation_space.shape}")
        print(f"  观察空间范围: {env.observation_space}")
        print(f"  动作空间大小: {env.action_space.n}")
        print(f"  可用动作:")
        print(f"    0 - 无操作")
        print(f"    1 - 左引擎点火")
        print(f"    2 - 主引擎点火")
        print(f"    3 - 右引擎点火")
        print(f"\n观察状态包含 8 个维度:")
        print(f"  [0] 水平坐标 (x)")
        print(f"  [1] 垂直坐标 (y)")
        print(f"  [2] 水平速度 (x)")
        print(f"  [3] 垂直速度 (y)")
        print(f"  [4] 角度")
        print(f"  [5] 角速度")
        print(f"  [6] 左腿接触地面 (布尔值)")
        print(f"  [7] 右腿接触地面 (布尔值)")
    
    all_rewards = []
    
    # 运行多个回合
    for episode in range(num_episodes):
        # 重置环境
        observation, info = env.reset()
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"回合 {episode + 1}/{num_episodes}")
            print(f"{'=' * 70}")
            print(f"初始观察状态: {observation}")
        
        # 运行一个完整的回合
        episode_reward = 0
        step_count = 0
        done = False
        
        if verbose:
            print(f"\n开始仿真...")
            print("-" * 70)
            print(f"{'步数':>5} | {'动作':>6} | {'即时奖励':>10} | {'累积奖励':>10} | {'状态摘要':>20}")
            print("-" * 70)
        
        while not done and step_count < max_steps:
            # 随机选择一个动作
            action = env.action_space.sample()
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 显示详细信息
            if verbose:
                action_name = ['无操作', '左引擎', '主引擎', '右引擎'][action]
                state_summary = f"h:{observation[0]:.2f} v:{observation[1]:.2f}"
                print(f"{step_count:5d} | {action_name:>6} | {reward:10.2f} | {episode_reward:10.2f} | {state_summary:>20}")
            
            done = terminated or truncated
        
        all_rewards.append(episode_reward)
        
        if verbose:
            print("-" * 70)
            print(f"\n回合 {episode + 1} 结束!")
            print(f"  总步数: {step_count}")
            print(f"  最终累积奖励: {episode_reward:.2f}")
            print(f"  终止原因: {'着陆/坠毁 (terminated)' if terminated else '超时 (truncated)'}")
            print(f"  最终位置: x={observation[0]:.2f}, y={observation[1]:.2f}")
            print(f"  最终速度: vx={observation[2]:.2f}, vy={observation[3]:.2f}")
            print(f"  最终角度: {observation[4]:.2f}")
    
    # 关闭环境
    env.close()
    
    # 显示统计信息
    if verbose and num_episodes > 1:
        print(f"\n{'=' * 70}")
        print(f"所有回合统计:")
        print(f"  总回合数: {num_episodes}")
        print(f"  平均奖励: {np.mean(all_rewards):.2f}")
        print(f"  最高奖励: {np.max(all_rewards):.2f}")
        print(f"  最低奖励: {np.min(all_rewards):.2f}")
        print(f"  奖励标准差: {np.std(all_rewards):.2f}")
        print(f"{'=' * 70}")
    
    return all_rewards


def main():
    """主函数"""
    print("\n" + "🚀" * 35)
    print("欢迎使用 LunarLander-v2 仿真系统")
    print("🚀" * 35 + "\n")
    
    # 运行单个回合的详细仿真
    print("运行模式 1: 单个回合详细仿真")
    run_simulation(max_steps=200, num_episodes=1, verbose=True)
    
    print("\n\n")
    
    # 运行多个回合的统计分析
    print("运行模式 2: 多回合统计分析")
    rewards = run_simulation(max_steps=200, num_episodes=5, verbose=False)
    
    print(f"\n{'=' * 70}")
    print(f"5 个回合的奖励统计:")
    print(f"{'=' * 70}")
    for i, reward in enumerate(rewards, 1):
        print(f"  回合 {i}: {reward:8.2f}")
    print(f"  {'平均值'}: {np.mean(rewards):8.2f}")
    print(f"  {'标准差'}: {np.std(rewards):8.2f}")
    print(f"{'=' * 70}")
    
    print("\n✅ 仿真完成！")
    print("\n提示: 这是使用随机策略，表现较差是正常的。")
    print("      训练后的智能体可以获得 200+ 的奖励！\n")


if __name__ == "__main__":
    main()

