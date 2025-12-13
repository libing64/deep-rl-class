#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen Lake Environment Wrapper
冰湖游戏环境封装 - 基于Gymnasium的Frozen Lake
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import spaces


class FrozenLakeWrapper:
    """
    Frozen Lake环境包装器

    游戏规则：
    - 冰湖上有冰块(F)、洞(H)、起点(S)和目标(G)
    - 智能体需要从起点到达目标
    - 掉入洞中则失败
    - 冰面光滑，可能滑行到非预期位置
    """

    def __init__(self, map_name="4x4", is_slippery=True, render_mode=None):
        """
        初始化Frozen Lake环境

        参数:
            map_name: 地图大小 ("4x4" 或 "8x8")
            is_slippery: 是否光滑（True表示会滑行）
            render_mode: 渲染模式
        """
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.render_mode = render_mode

        # 创建环境
        self.env = gym.make(
            f'FrozenLake-v1',
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode=render_mode
        )

        # 环境信息
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

        # 地图信息
        if map_name == "4x4":
            self.map_size = 4
            self.desc = np.array([
                ['S', 'F', 'F', 'F'],
                ['F', 'H', 'F', 'H'],
                ['F', 'F', 'F', 'H'],
                ['H', 'F', 'F', 'G']
            ])
        elif map_name == "8x8":
            self.map_size = 8
            self.desc = self.env.desc
        else:
            raise ValueError("Unsupported map_name. Use '4x4' or '8x8'")

        print(f"Frozen Lake {map_name} Environment initialized:")
        print(f"  States: {self.n_states}")
        print(f"  Actions: {self.n_actions} (0:LEFT, 1:DOWN, 2:RIGHT, 3:UP)")
        print(f"  Slippery: {self.is_slippery}")

    def reset(self):
        """重置环境"""
        observation, info = self.env.reset()
        return observation, info

    def step(self, action):
        """执行一步"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self.env.render()
        else:
            self.render_ascii()

    def render_ascii(self):
        """ASCII渲染"""
        if hasattr(self.env, 's'):
            pos = self.env.s
            row, col = pos // self.map_size, pos % self.map_size

            print("\nFrozen Lake Map:")
            print("-" * (self.map_size * 2 + 1))
            for i in range(self.map_size):
                print("|", end="")
                for j in range(self.map_size):
                    if i == row and j == col:
                        print("🤖|", end="")  # 智能体位置
                    else:
                        cell = self.desc[i, j]
                        if cell == 'S':
                            print("🏠|", end="")  # 起点
                        elif cell == 'G':
                            print("🏆|", end="")  # 目标
                        elif cell == 'H':
                            print("🕳️|", end="")  # 洞
                        elif cell == 'F':
                            print("🧊|", end="")  # 冰块
                print()
            print("-" * (self.map_size * 2 + 1))
        else:
            print("Environment not initialized properly")

    def get_map_info(self):
        """获取地图信息"""
        return {
            'size': self.map_size,
            'desc': self.desc,
            'start_pos': np.where(self.desc == 'S'),
            'goal_pos': np.where(self.desc == 'G'),
            'hole_pos': np.where(self.desc == 'H')
        }

    def close(self):
        """关闭环境"""
        self.env.close()

    def __str__(self):
        """字符串表示"""
        return f"FrozenLake-{self.map_name}({'slippery' if self.is_slippery else 'not slippery'})"


# 测试环境
if __name__ == "__main__":
    print("Testing Frozen Lake Environment...")

    # 测试4x4地图
    print("\n" + "="*50)
    print("4x4 MAP TEST")
    print("="*50)

    env_4x4 = FrozenLakeWrapper("4x4", is_slippery=True)
    env_4x4.render_ascii()

    # 随机游走测试
    print("\nRandom walk test:")
    observation, info = env_4x4.reset()
    print(f"Start at position: {observation}")

    for step in range(10):
        action = env_4x4.env.action_space.sample()
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        print(f"Step {step+1}: Action = {action_names[action]}")

        observation, reward, terminated, truncated, info = env_4x4.step(action)
        print(f"  New position: {observation}, Reward: {reward}")

        env_4x4.render_ascii()

        if terminated or truncated:
            result = "SUCCESS! 🎉" if reward == 1 else "FAILED! 💥"
            print(f"  Episode ended: {result}")
            break

    env_4x4.close()

    # 测试8x8地图
    print("\n" + "="*50)
    print("8x8 MAP TEST")
    print("="*50)

    env_8x8 = FrozenLakeWrapper("8x8", is_slippery=False)
    observation, info = env_8x8.reset()
    print(f"8x8 map start position: {observation}")
    env_8x8.render_ascii()

    env_8x8.close()

    print("\nEnvironment tests completed!")
