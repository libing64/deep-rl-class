#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LunarLander-v2 Environment Simulation with GUI
Runs the Lunar Lander environment with visual display
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys


class LunarLanderSimulator:
    """LunarLander-v2 simulator with graphical interface"""
    
    def __init__(self, render_mode='human'):
        """Initialize the simulator"""
        self.render_mode = render_mode
        self.env = None
        self.fig = None
        self.ax = None
        
    def create_environment(self):
        """Create the environment"""
        self.env = gym.make('LunarLander-v2', render_mode=self.render_mode)
        return self.env
    
    def print_env_info(self):
        """Print environment information"""
        print("=" * 70)
        print("LunarLander-v2 Environment Simulation")
        print("=" * 70)
        print(f"\nEnvironment Information:")
        print(f"  Observation Space: {self.env.observation_space.shape}")
        print(f"  Action Space: {self.env.action_space.n}")
        print(f"  Available Actions:")
        print(f"    0 - Do nothing")
        print(f"    1 - Fire left engine")
        print(f"    2 - Fire main engine")
        print(f"    3 - Fire right engine")
        print(f"\nObservation State (8 dimensions):")
        print(f"  [0] Horizontal coordinate (x)")
        print(f"  [1] Vertical coordinate (y)")
        print(f"  [2] Horizontal velocity (vx)")
        print(f"  [3] Vertical velocity (vy)")
        print(f"  [4] Angle")
        print(f"  [5] Angular velocity")
        print(f"  [6] Left leg contact (boolean)")
        print(f"  [7] Right leg contact (boolean)")
        print("=" * 70)
    
    def run_with_display(self, max_steps=500, num_episodes=3, delay=0.01):
        """Run simulation with graphical display"""
        
        self.create_environment()
        self.print_env_info()
        
        all_rewards = []
        
        for episode in range(num_episodes):
            print(f"\n{'=' * 70}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'=' * 70}")
            
            # Reset environment
            observation, info = self.env.reset()
            
            episode_reward = 0
            step_count = 0
            done = False
            
            print(f"\nStarting simulation...")
            print(f"Close the window to stop the current episode")
            print("-" * 70)
            
            while not done and step_count < max_steps:
                # Render the environment
                if self.render_mode == 'human':
                    self.env.render()
                
                # Random action
                action = self.env.action_space.sample()
                
                # Execute action
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # Print info every 50 steps
                if step_count % 50 == 0:
                    action_names = ['None', 'Left', 'Main', 'Right']
                    print(f"Step {step_count:3d} | Action: {action_names[action]:5s} | "
                          f"Reward: {reward:7.2f} | Total: {episode_reward:8.2f}")
                
                done = terminated or truncated
                
                # Add delay for better visualization
                time.sleep(delay)
            
            all_rewards.append(episode_reward)
            
            print("-" * 70)
            print(f"\nEpisode {episode + 1} Complete!")
            print(f"  Total Steps: {step_count}")
            print(f"  Final Reward: {episode_reward:.2f}")
            print(f"  Termination: {'Landed/Crashed' if terminated else 'Timeout'}")
            print(f"  Final Position: x={observation[0]:.2f}, y={observation[1]:.2f}")
            
            if episode < num_episodes - 1:
                print(f"\nStarting next episode in 2 seconds...")
                time.sleep(2)
        
        self.env.close()
        
        # Print statistics
        if num_episodes > 1:
            print(f"\n{'=' * 70}")
            print(f"Statistics for All Episodes:")
            print(f"  Total Episodes: {num_episodes}")
            print(f"  Average Reward: {np.mean(all_rewards):.2f}")
            print(f"  Best Reward: {np.max(all_rewards):.2f}")
            print(f"  Worst Reward: {np.min(all_rewards):.2f}")
            print(f"  Std Deviation: {np.std(all_rewards):.2f}")
            print(f"{'=' * 70}")
        
        return all_rewards
    
    def run_with_plot(self, max_steps=500):
        """Run simulation and plot results using matplotlib"""
        
        self.create_environment()
        self.print_env_info()
        
        print(f"\n{'=' * 70}")
        print(f"Running simulation with matplotlib visualization...")
        print(f"{'=' * 70}")
        
        # Reset environment
        observation, info = self.env.reset()
        
        # Storage for visualization
        positions_x = []
        positions_y = []
        velocities_x = []
        velocities_y = []
        angles = []
        rewards = []
        cumulative_rewards = []
        
        episode_reward = 0
        step_count = 0
        done = False
        
        # Collect data
        while not done and step_count < max_steps:
            # Store current state
            positions_x.append(observation[0])
            positions_y.append(observation[1])
            velocities_x.append(observation[2])
            velocities_y.append(observation[3])
            angles.append(observation[4])
            
            # Random action
            action = self.env.action_space.sample()
            
            # Execute action
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            episode_reward += reward
            rewards.append(reward)
            cumulative_rewards.append(episode_reward)
            step_count += 1
            
            done = terminated or truncated
        
        self.env.close()
        
        print(f"\nSimulation complete! Creating plots...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'LunarLander-v2 Simulation Results (Steps: {step_count}, Final Reward: {episode_reward:.2f})', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Trajectory
        axes[0, 0].plot(positions_x, positions_y, 'b-', linewidth=2)
        axes[0, 0].plot(positions_x[0], positions_y[0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(positions_x[-1], positions_y[-1], 'ro', markersize=10, label='End')
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Y Position')
        axes[0, 0].set_title('Lander Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Velocities
        steps = range(len(velocities_x))
        axes[0, 1].plot(steps, velocities_x, 'r-', label='Horizontal Velocity', linewidth=2)
        axes[0, 1].plot(steps, velocities_y, 'b-', label='Vertical Velocity', linewidth=2)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Velocity')
        axes[0, 1].set_title('Velocities Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Angle
        axes[0, 2].plot(steps, angles, 'g-', linewidth=2)
        axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Angle (radians)')
        axes[0, 2].set_title('Lander Angle Over Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Position X over time
        axes[1, 0].plot(steps, positions_x, 'c-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('X Position')
        axes[1, 0].set_title('Horizontal Position Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Instantaneous Rewards
        axes[1, 1].plot(steps, rewards, 'm-', linewidth=1, alpha=0.7)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Instantaneous Rewards')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Cumulative Rewards
        axes[1, 2].plot(steps, cumulative_rewards, 'r-', linewidth=2)
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Cumulative Reward')
        axes[1, 2].set_title('Cumulative Reward Over Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = 'lunar_lander_simulation_results.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        
        plt.show()
        
        print(f"\nFinal Statistics:")
        print(f"  Total Steps: {step_count}")
        print(f"  Final Reward: {episode_reward:.2f}")
        print(f"  Average Reward per Step: {episode_reward/step_count:.2f}")
        
        return episode_reward


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("  LunarLander-v2 Simulation with Graphical Interface")
    print("=" * 70)
    print("\nChoose simulation mode:")
    print("  1. Real-time display (pygame window)")
    print("  2. Plot analysis (matplotlib charts)")
    print("  3. Both modes")
    print("=" * 70)
    
    try:
        choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"
    
    simulator = LunarLanderSimulator()
    
    if choice == "1":
        print("\n[Mode 1: Real-time Display]")
        print("The environment will open in a new window.")
        print("Watch the lander attempt to land!")
        simulator.render_mode = 'human'
        simulator.run_with_display(max_steps=500, num_episodes=3, delay=0.01)
        
    elif choice == "2":
        print("\n[Mode 2: Plot Analysis]")
        print("Running simulation and generating analysis plots...")
        simulator.render_mode = 'rgb_array'
        simulator.run_with_plot(max_steps=500)
        
    elif choice == "3":
        print("\n[Mode 3: Both Modes]")
        print("\nFirst: Real-time display...")
        simulator.render_mode = 'human'
        simulator.run_with_display(max_steps=500, num_episodes=2, delay=0.01)
        
        print("\n\nSecond: Plot analysis...")
        simulator.render_mode = 'rgb_array'
        simulator2 = LunarLanderSimulator()
        simulator2.run_with_plot(max_steps=500)
    else:
        print("\nInvalid choice. Running default mode (Real-time display)...")
        simulator.render_mode = 'human'
        simulator.run_with_display(max_steps=500, num_episodes=2, delay=0.01)
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)
    print("\nNote: This uses a random policy, so poor performance is expected.")
    print("A trained agent can achieve 200+ reward!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
