#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Learning Agent for Frozen Lake
冰湖游戏的Q-Learning智能体实现
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    """Q-Learning智能体"""

    def __init__(self, n_states, n_actions,
                 learning_rate=0.1, discount_factor=0.99, epsilon=1.0,
                 epsilon_decay=0.9995, epsilon_min=0.01):
        """
        初始化Q-Learning智能体

        参数:
            n_states: 状态数量
            n_actions: 动作数量
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # Q-Learning参数
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 初始化Q表
        self.q_table = np.zeros((n_states, n_actions))

        print(f"Q-Learning Agent initialized:")
        print(f"  States: {n_states}, Actions: {n_actions}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Discount factor: {self.discount_factor}")
        print(f"  Initial epsilon: {self.epsilon}")

    def get_action(self, state, training=True):
        """
        根据当前状态选择动作

        参数:
            state: 当前状态
            training: 是否处于训练模式

        返回:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        """
        更新Q表

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # Q-Learning更新公式
        current_q = self.q_table[state, action]

        if done:
            # 终止状态，没有未来奖励
            target = reward
        else:
            # 非终止状态，使用Bellman方程
            max_future_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_future_q

        # 更新Q值
        self.q_table[state, action] += self.learning_rate * (target - current_q)

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        """保存模型"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'n_states': self.n_states,
            'n_actions': self.n_actions
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model saved to {filename}")
        print(f"Q-table shape: {self.q_table.shape}")

    def load_model(self, filename):
        """加载模型"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.n_states = data['n_states']
            self.n_actions = data['n_actions']

            print(f"Model loaded from {filename}")
            print(f"Q-table shape: {self.q_table.shape}")
            return True
        else:
            print(f"Model file {filename} not found")
            return False

    def get_optimal_policy(self):
        """获取最优策略"""
        policy = np.zeros(self.n_states, dtype=int)
        for state in range(self.n_states):
            policy[state] = np.argmax(self.q_table[state])
        return policy

    def get_state_values(self):
        """获取状态值函数"""
        return np.max(self.q_table, axis=1)

    def get_stats(self):
        """获取统计信息"""
        return {
            'q_table_shape': self.q_table.shape,
            'total_q_values': self.q_table.size,
            'max_q_value': np.max(self.q_table),
            'min_q_value': np.min(self.q_table),
            'avg_q_value': np.mean(self.q_table),
            'epsilon': self.epsilon,
            'explored_states': np.count_nonzero(np.max(self.q_table, axis=1))
        }


def train_q_learning(env, episodes=5000, save_path="frozen_lake_agent.pkl"):
    """
    训练Q-Learning智能体

    参数:
        env: 环境对象
        episodes: 训练回合数
        save_path: 模型保存路径

    返回:
        训练好的智能体和奖励历史
    """
    print("🚀 Starting Q-Learning Training")
    print("=" * 50)

    # 创建智能体
    agent = QLearningAgent(env.n_states, env.n_actions)

    # 训练统计
    rewards_history = []
    steps_history = []
    epsilon_history = []
    success_count = 0

    print(f"Training for {episodes} episodes...")
    print("-" * 50)

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # 选择动作
            action = agent.get_action(state, training=True)

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)

            # 更新Q表
            done = terminated or truncated
            agent.update_q_table(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            # 防止无限循环
            if steps >= 100:
                done = True

        # 衰减探索率
        agent.decay_epsilon()

        # 记录统计信息
        rewards_history.append(total_reward)
        steps_history.append(steps)
        epsilon_history.append(agent.epsilon)

        # 记录成功次数
        if total_reward == 1.0:
            success_count += 1

        # 每500回合显示进度
        if (episode + 1) % 500 == 0:
            recent_success_rate = np.mean(rewards_history[-500:]) * 100
            avg_reward = np.mean(rewards_history[-500:])
            print(f"Episode {episode + 1:5d}/{episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Success Rate: {recent_success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f}")

    print("\n" + "=" * 50)
    print("🎯 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final success rate (last 500): {np.mean(rewards_history[-500:]) * 100:.1f}%")
    print(f"Total successful episodes: {success_count}/{episodes}")
    print(f"Overall success rate: {success_count/episodes * 100:.1f}%")

    # 保存模型
    agent.save_model(save_path)

    return agent, rewards_history, steps_history, epsilon_history


# 测试Q-Learning智能体
if __name__ == "__main__":
    from frozen_lake_env import FrozenLakeWrapper

    print("Testing Q-Learning Agent...")

    # 创建环境和智能体
    env = FrozenLakeWrapper("4x4", is_slippery=True)
    agent = QLearningAgent(env.n_states, env.n_actions)

    # 测试几个回合的学习
    print("\nTraining test (100 episodes):")
    agent, rewards, steps, epsilons = train_q_learning(env, episodes=100)

    # 显示最终Q表
    print("\nFinal Q-table (first 5 states):")
    print(agent.q_table[:5])

    # 显示最优策略
    policy = agent.get_optimal_policy()
    action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    print("\nOptimal policy:")
    for i in range(env.n_states):
        row, col = i // env.map_size, i % env.map_size
        print(f"State {i:2d} ({row},{col}): {action_names[policy[i]]}")

    env.close()

    print("\nQ-Learning test completed!")
