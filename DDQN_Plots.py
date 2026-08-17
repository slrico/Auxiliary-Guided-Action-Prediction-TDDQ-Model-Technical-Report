import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

episodes = range(len(episode_rewards))

# --- Reward ---
kernel_size = 10
plt.figure(figsize=(10, 4))
plt.plot(episodes, episode_rewards, label="Reward")

if len(episode_rewards) >= kernel_size:
    smoothed_rewards = np.convolve(
        episode_rewards, np.ones(kernel_size)/kernel_size, mode='valid'
    )
    smoothed_episodes = episodes[kernel_size - 1:]
    plt.plot(smoothed_episodes, smoothed_rewards, label="Smoothed", linestyle='--')
elif len(episode_rewards) > 0:
    smoothed_rewards = pd.Series(episode_rewards).rolling(
        window=kernel_size, min_periods=1
    ).mean()
    plt.plot(episodes, smoothed_rewards, label="Smoothed", linestyle='--')

plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Loss ---
if len(episode_losses) == len(episodes):
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, episode_losses, label="Mean Loss")
    plt.fill_between(
        episodes,
        np.array(episode_losses) - np.array(episode_loss_stds),
        np.array(episode_losses) + np.array(episode_loss_stds),
        alpha=0.3,
        label="±1 Std"
    )
    plt.title("Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping loss plot: mismatch or empty data.")

# --- Entropy ---
if len(episode_entropies) == len(episodes) and len(episode_entropies) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, episode_entropies, color='purple')
    plt.title("Policy Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
elif len(episode_entropies) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(episode_entropies)), episode_entropies, color='purple')
    plt.title("Policy Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping entropy plot: mismatch or empty data.")

# --- Action histogram (last episode) ---
if len(episode_actions) > 0:
    last_actions = episode_actions[-1]
    action_counts = np.bincount(last_actions, minlength=action_dim)

    plt.figure(figsize=(10, 4))
    plt.bar(range(action_dim), action_counts)
    plt.title("Action Usage (Last Episode)")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Skipping action histogram: no actions recorded.")
    
    