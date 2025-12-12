#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LunarLander-v2 Plot Analysis
Runs simulation and generates detailed analysis plots
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def run_and_plot(max_steps=500):
    """Run simulation and create detailed plots"""
    
    # Create environment
    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    
    print("=" * 70)
    print("LunarLander-v2 Plot Analysis")
    print("=" * 70)
    print(f"\nEnvironment Information:")
    print(f"  Observation Space: {env.observation_space.shape}")
    print(f"  Action Space: {env.action_space.n}")
    print("=" * 70)
    
    # Reset environment
    observation, info = env.reset()
    
    # Storage for visualization
    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []
    angles = []
    rewards = []
    cumulative_rewards = []
    actions_taken = []
    
    episode_reward = 0
    step_count = 0
    done = False
    
    print(f"\nRunning simulation...")
    
    # Collect data
    while not done and step_count < max_steps:
        # Store current state
        positions_x.append(observation[0])
        positions_y.append(observation[1])
        velocities_x.append(observation[2])
        velocities_y.append(observation[3])
        angles.append(observation[4])
        
        # Random action
        action = env.action_space.sample()
        actions_taken.append(action)
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        rewards.append(reward)
        cumulative_rewards.append(episode_reward)
        step_count += 1
        
        # Print progress
        if step_count % 100 == 0:
            print(f"  Step {step_count}/{max_steps}, Reward: {episode_reward:.2f}")
        
        done = terminated or truncated
    
    env.close()
    
    print(f"\nSimulation complete!")
    print(f"  Total Steps: {step_count}")
    print(f"  Final Reward: {episode_reward:.2f}")
    print(f"  Termination: {'Landed/Crashed' if terminated else 'Timeout'}")
    print(f"\nGenerating plots...")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Main title
    fig.suptitle(f'LunarLander-v2 Detailed Analysis\n' + 
                 f'Steps: {step_count} | Final Reward: {episode_reward:.2f}',
                 fontsize=16, fontweight='bold')
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    steps = range(len(positions_x))
    
    # Plot 1: Trajectory (large plot, top-left 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(positions_x, positions_y, c=steps, cmap='viridis', 
                         s=20, alpha=0.6)
    ax1.plot(positions_x, positions_y, 'b-', linewidth=1, alpha=0.3)
    ax1.plot(positions_x[0], positions_y[0], 'go', markersize=15, 
             label='Start', zorder=5)
    ax1.plot(positions_x[-1], positions_y[-1], 'ro', markersize=15, 
             label='End', zorder=5)
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, 
                alpha=0.5, label='Ground')
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Lander Trajectory (colored by time)', fontsize=14, 
                  fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Step', fontsize=10)
    
    # Plot 2: Velocities
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(steps, velocities_x, 'r-', label='Vx (horizontal)', linewidth=2)
    ax2.plot(steps, velocities_y, 'b-', label='Vy (vertical)', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('Velocity', fontsize=10)
    ax2.set_title('Velocities', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Angle
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(steps, np.degrees(angles), 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Step', fontsize=10)
    ax3.set_ylabel('Angle (degrees)', fontsize=10)
    ax3.set_title('Lander Angle', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Position X
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(steps, positions_x, 'c-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.fill_between(steps, positions_x, 0, alpha=0.3)
    ax4.set_xlabel('Step', fontsize=10)
    ax4.set_ylabel('X Position', fontsize=10)
    ax4.set_title('Horizontal Position', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Rewards
    ax5 = fig.add_subplot(gs[2, 1])
    colors = ['red' if r < 0 else 'green' for r in rewards]
    ax5.bar(steps, rewards, color=colors, alpha=0.6, width=1)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax5.set_xlabel('Step', fontsize=10)
    ax5.set_ylabel('Reward', fontsize=10)
    ax5.set_title('Instantaneous Rewards', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Cumulative Rewards
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(steps, cumulative_rewards, 'purple', linewidth=2.5)
    ax6.fill_between(steps, cumulative_rewards, 0, alpha=0.3, color='purple')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax6.axhline(y=200, color='g', linestyle=':', linewidth=2, 
                alpha=0.5, label='Target (200)')
    ax6.set_xlabel('Step', fontsize=10)
    ax6.set_ylabel('Cumulative Reward', fontsize=10)
    ax6.set_title('Total Reward Over Time', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Save figure
    filename = 'lunar_lander_analysis.png'
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"\n✓ Plot saved as: {filename}")
    
    # Create action distribution plot
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 5))
    
    action_names = ['No-op', 'Left', 'Main', 'Right']
    action_counts = [actions_taken.count(i) for i in range(4)]
    
    # Action distribution pie chart
    colors_pie = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
    ax7.pie(action_counts, labels=action_names, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax7.set_title('Action Distribution', fontsize=14, fontweight='bold')
    
    # Action frequency bar chart
    ax8.bar(action_names, action_counts, color=colors_pie, alpha=0.7)
    ax8.set_xlabel('Action', fontsize=12)
    ax8.set_ylabel('Frequency', fontsize=12)
    ax8.set_title('Action Frequency', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(action_counts):
        ax8.text(i, v + max(action_counts)*0.02, str(v), 
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    filename2 = 'lunar_lander_actions.png'
    plt.savefig(filename2, dpi=200, bbox_inches='tight')
    print(f"✓ Actions plot saved as: {filename2}")
    
    # Print statistics
    print(f"\n{'=' * 70}")
    print("Final Statistics:")
    print(f"{'=' * 70}")
    print(f"  Total Steps: {step_count}")
    print(f"  Final Reward: {episode_reward:.2f}")
    print(f"  Average Reward per Step: {episode_reward/step_count:.2f}")
    print(f"  Final Position: x={positions_x[-1]:.2f}, y={positions_y[-1]:.2f}")
    print(f"  Final Velocity: vx={velocities_x[-1]:.2f}, vy={velocities_y[-1]:.2f}")
    print(f"  Final Angle: {np.degrees(angles[-1]):.2f}°")
    print(f"\n  Action Distribution:")
    for i, name in enumerate(action_names):
        print(f"    {name}: {action_counts[i]} ({action_counts[i]/step_count*100:.1f}%)")
    print(f"{'=' * 70}")
    
    return episode_reward


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("  LunarLander-v2 Analysis & Visualization")
    print("🚀" * 35 + "\n")
    
    reward = run_and_plot(max_steps=500)
    
    print("\n✅ Analysis complete!")
    print("\n📊 Generated files:")
    print("  - lunar_lander_analysis.png (detailed trajectory and metrics)")
    print("  - lunar_lander_actions.png (action distribution)")
    print("\n💡 Note: Random policy typically scores -200 to +50")
    print("   Trained agents can achieve 200+ rewards!\n")

