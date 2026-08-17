import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim_text, obs_dim_struct, action_dim):
        super().__init__()
        self.text_proj = nn.Linear(obs_dim_text, 128)
        self.struct_proj = nn.Linear(obs_dim_struct, 64)

        self.shared = nn.Sequential(
            nn.Linear(128 + 64, 256), nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs_text, obs_struct):
        text_feat = self.text_proj(obs_text)
        struct_feat = self.struct_proj(obs_struct)
        x = torch.cat([text_feat, struct_feat], dim=1)
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
### -- the transition model ----- 
def get_real_transition(sample):
    obs_text, obs_struct, decision_tensor, _ = encode_sample(sample)
    next_sample = generate_sample()
    next_obs_text, next_obs_struct, _, _ = encode_sample(next_sample)
    decision_key = str(sample.get("decision_options"))
    if decision_key not in action_vocab:
        action_vocab[decision_key] = len(action_vocab)
    action = action_vocab[decision_key]
    with torch.no_grad():
        reward_tensor = reward_model(
            obs_text.unsqueeze(0).to(device),
            obs_struct.unsqueeze(0).to(device),
            decision_tensor.unsqueeze(0).to(device)
        )
    reward = float(reward_tensor.item())

    done = False
    return obs_text, obs_struct, action, reward, next_obs_text, next_obs_struct, done


## --- The training ---------
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

# --- Hyperparameters ---
TEXT_DIM = 384
STRUCT_DIM = 384
NUM_ACTIONS = len(action_vocab)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE_FREQ = 10
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
REPLAY_CAPACITY = 10000
TOTAL_STEPS = 5000

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_text, obs_struct, action, reward, next_text, next_struct, done):
        self.buffer.append((obs_text, obs_struct, action, reward, next_text, next_struct, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_text, obs_struct, actions, rewards, next_text, next_struct, dones = zip(*batch)
        return (
            torch.stack(obs_text),
            torch.stack(obs_struct),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.stack(next_text),
            torch.stack(next_struct),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# --- Dueling Q-Network ---
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim_text, obs_dim_struct, action_dim):
        super().__init__()
        self.text_proj = nn.Linear(obs_dim_text, 128)
        self.struct_proj = nn.Linear(obs_dim_struct, 64)
        self.shared = nn.Sequential(nn.Linear(128 + 64, 256), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, obs_text, obs_struct):
        text_feat = self.text_proj(obs_text)
        struct_feat = self.struct_proj(obs_struct)
        x = torch.cat([text_feat, struct_feat], dim=1)
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


policy_net = DuelingQNetwork(TEXT_DIM, STRUCT_DIM, NUM_ACTIONS).to(device)
target_net = DuelingQNetwork(TEXT_DIM, STRUCT_DIM, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)


for step in range(TOTAL_STEPS):
    sample = synthetic_dataset[step % len(synthetic_dataset)]
    obs_text, obs_struct, action, reward, next_text, next_struct, done = get_real_transition(sample)
    if random.random() < EPSILON:
        action = random.randint(0, NUM_ACTIONS - 1)
    else:
        with torch.no_grad():
            q_vals = policy_net(obs_text.unsqueeze(0).to(device), obs_struct.unsqueeze(0).to(device))
            action = q_vals.argmax().item()

    replay_buffer.push(obs_text, obs_struct, action, reward, next_text, next_struct, done)

    if len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        obs_text_b, obs_struct_b, actions_b, rewards_b, next_text_b, next_struct_b, dones_b = batch

        obs_text_b = obs_text_b.to(device)
        obs_struct_b = obs_struct_b.to(device)
        actions_b = actions_b.to(device)
        rewards_b = rewards_b.to(device)
        next_text_b = next_text_b.to(device)
        next_struct_b = next_struct_b.to(device)
        dones_b = dones_b.to(device)
        q_values = policy_net(obs_text_b, obs_struct_b).gather(1, actions_b)

        with torch.no_grad():
            next_q_values = target_net(next_text_b, next_struct_b).max(dim=1, keepdim=True)[0]
            target_q = rewards_b + GAMMA * next_q_values * (1 - dones_b)

        loss = nn.functional.mse_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
    EPSILON = max(EPSILON_MIN, EPSILON - 1e-4)
    if step % 100 == 0 and len(replay_buffer) >= BATCH_SIZE:
        print(f"Step {step} — Loss: {loss.item():.4f} — Epsilon: {EPSILON:.3f}") 
        
        
##-- The plots ------------------
episodes = range(len(episode_rewards))
# --- Reward ---
plt.figure(figsize=(10, 4))
plt.plot(episodes, episode_rewards, label="Reward", alpha=0.5)
smoothed_rewards = pd.Series(episode_rewards).rolling(window=20, min_periods=1).mean()
plt.plot(episodes, smoothed_rewards, label="Smoothed Reward", color='blue')
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Loss ---
plt.figure(figsize=(10, 4))
plt.plot(episodes, episode_losses, label="Loss", alpha=0.5, color='orange')
smoothed_losses = pd.Series(episode_losses).rolling(window=20, min_periods=1).mean()
plt.plot(episodes, smoothed_losses, label="Smoothed Loss", color='red')
plt.title("Training Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Epsilon ---
plt.figure(figsize=(10, 4))
plt.plot(episodes, epsilon, label="Epsilon", color='green')
plt.title("Exploration vs Exploitation")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Action distribution ---
action_counts = np.bincount(action_history, minlength=NUM_ACTIONS)
plt.figure(figsize=(10, 4))
plt.bar(range(NUM_ACTIONS), action_counts, color='purple')
plt.title("Action Usage Distribution")
plt.xlabel("Action")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

