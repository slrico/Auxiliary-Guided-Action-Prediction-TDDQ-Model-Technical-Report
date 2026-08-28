from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

obs = env.reset()
print(obs)

dataset = load_dataset("infinite-dataset-hub/AutonomousDriveDecisions", streaming=True)
sample = next(iter(dataset["train"]))
print(sample)


sample = next(iter(dataset["train"]))
print("Sample keys:", sample.keys())
print("Sample content:", sample)

ACTION_KEY = "decision_options"  

action_counter = Counter()
for example in dataset["train"]:
    action = example[ACTION_KEY]
    action_counter[action] += 1
    if len(action_counter) > 1000:
        break

ACTION_TEXTS = list(action_counter.keys())
print("Extracted actions:", ACTION_TEXTS)


print("Action type:", type(action))
if isinstance(action, dict):
    print("Action keys:", action.keys())


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

def embed_action_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state.mean(dim=1)
    return embedding.squeeze()


ACTION_TEXTS = [  # from your dataset
    "Proceed", "Change Lane", "Emergency Brake", "Use Detour Sign", "Take highway exit",
    "Alternate Route", "Use Alternate Route", "Proceed with caution", "Recommend detour",
    "Find Sheltered Area", "Find Alternate Route", "Take alternative route",
    "Opt for Side Roads", "Use Pre-Planned Winter Route", "Suggest safer path",
    "Adjust Speed and Steering", "Use Nearest Accessible Bridge", "Obstacle Avoidance",
    "Reduce speed", "Maintain current speed and position", "Follow Path with Caution",
    "Avoid Waterlogged Areas", "Find charging station", "Route not recommended due to tides",
    "Use Detour Signal", "Use Sports Event Detour", "Take long straight path",
    "Maintain Current Path", "Use Detour Around", "Suggest using GPS guidance",
    "Follow Main Street", "Use Festival Circuit", "Maintain distance",
    "Recommend off-peak travel time", "Increase Headlights and Decrease Speed",
    "Use Partial Lane Alternative"
]

action_embeddings = torch.stack([embed_action_text(a) for a in ACTION_TEXTS])  # [N, D]
NUM_ACTIONS = len(ACTION_TEXTS)
ACTION_DIM = action_embeddings.shape[1]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.net(state)  # latent action vector
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action_embed):
        x = torch.cat([state, action_embed], dim=1)
        return self.net(x)
    
def select_action(actor_output):
    sims = F.cosine_similarity(actor_output.unsqueeze(0), action_embeddings, dim=1)
    best_idx = torch.argmax(sims).item()
    return ACTION_TEXTS[best_idx], action_embeddings[best_idx]

def train_step(buffer, actor, critic, target_actor, target_critic,
               actor_opt, critic_opt, action_embeddings,
               gamma=0.99, tau=0.005):

    if len(buffer) < BATCH_SIZE:
        print("[Skip] Buffer too small:", len(buffer))
        return

    s, a_idx, r, s_next, done = buffer.sample(BATCH_SIZE)
    a_embed = action_embeddings[a_idx]  # [B, D]

    print(f"[Batch] State shape: {s.shape}, Action idx: {a_idx[:5].tolist()}, Reward: {r[:5].squeeze().tolist()}")

    with torch.no_grad():
        a_next = target_actor(s_next)  # [B, D]
        sims = F.cosine_similarity(a_next.unsqueeze(1), action_embeddings.unsqueeze(0), dim=2)  # [B, N]
        best_next_idx = torch.argmax(sims, dim=1)  # [B]
        a_next_embed = action_embeddings[best_next_idx]  # [B, D]
        q_target = target_critic(s_next, a_next_embed)  # [B, 1]
        y = r + (1 - done) * gamma * q_target  # [B, 1]

    print(f"[Target Q] Mean: {q_target.mean().item():.4f}, Min: {q_target.min().item():.4f}, Max: {q_target.max().item():.4f}")

    q_val = critic(s, a_embed)  # [B, 1]
    critic_loss = F.mse_loss(q_val, y)

    print(f"[Critic] Loss: {critic_loss.item():.4f}, Q mean: {q_val.mean().item():.4f}")

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    a_pred = actor(s)  
    sims = F.cosine_similarity(a_pred.unsqueeze(1), action_embeddings.unsqueeze(0), dim=2)  
    best_idx = torch.argmax(sims, dim=1)  
    best_embed = action_embeddings[best_idx]  
    actor_loss = -critic(s, best_embed).mean()

    print(f"[Actor] Loss: {actor_loss.item():.4f}, Avg similarity: {sims.mean().item():.4f}")
    print(f"[Actor] Selected actions: {best_idx[:5].tolist()}")

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    with torch.no_grad():
        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    print("[Update] Target networks updated.\n")
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM = 384  
BATCH_SIZE = 64
GAMMA = 0.99
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
TAU = 0.005
REPLAY_CAPACITY = 10000
TOTAL_EPISODES = 100
MAX_STEPS_PER_EPISODE = 50

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action_idxs, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.tensor(action_idxs, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action_embed):
        x = torch.cat([state, action_embed], dim=1)
        return self.net(x)

# --- Initialize Networks ---
actor = Actor(STATE_DIM, ACTION_DIM).to(device)
target_actor = Actor(STATE_DIM, ACTION_DIM).to(device)
target_actor.load_state_dict(actor.state_dict())

critic = Critic(STATE_DIM, ACTION_DIM).to(device)
target_critic = Critic(STATE_DIM, ACTION_DIM).to(device)
target_critic.load_state_dict(critic.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_opt = optim.Adam(critic.parameters(), lr=LR_CRITIC)

replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

# --- Training Loop ---##
episode_rewards = []
episode_losses_critic = []
episode_losses_actor = []

for episode in range(TOTAL_EPISODES):
    state = torch.randn(STATE_DIM).to(device)  
    episode_reward = 0
    episode_actor_loss = 0
    episode_critic_loss = 0
    step_count = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        with torch.no_grad():
            a_pred = actor(state.unsqueeze(0))  
            sims = F.cosine_similarity(a_pred.unsqueeze(1), action_embeddings.unsqueeze(0), dim=2)
            action_idx = torch.argmax(sims).item()
            action_embed = action_embeddings[action_idx]
        next_state = torch.randn(STATE_DIM).to(device)  
        reward = random.uniform(-1, 1)
        done = False
        replay_buffer.push(state.detach().cpu(), action_idx, reward, next_state.detach().cpu(), done)
        state = next_state.clone()
        episode_reward += reward
        step_count += 1

    if len(replay_buffer) >= BATCH_SIZE:
        s, a_idx, r, s_next, done = replay_buffer.sample(BATCH_SIZE)
        a_embed = action_embeddings[a_idx]

        with torch.no_grad():
            a_next = target_actor(s_next)
            sims = F.cosine_similarity(a_next.unsqueeze(1), action_embeddings.unsqueeze(0), dim=2)
            best_next_idx = torch.argmax(sims, dim=1)
            a_next_embed = action_embeddings[best_next_idx]
            q_target = target_critic(s_next, a_next_embed)
            y = r + (1 - done) * GAMMA * q_target

        q_val = critic(s, a_embed)
        critic_loss = F.mse_loss(q_val, y)

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        a_pred = actor(s)
        sims = F.cosine_similarity(a_pred.unsqueeze(1), action_embeddings.unsqueeze(0), dim=2)
        best_idx = torch.argmax(sims, dim=1)
        best_embed = action_embeddings[best_idx]
        actor_loss = -critic(s, best_embed).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # --- Soft update targets ---
        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        episode_actor_loss = actor_loss.item()
        episode_critic_loss = critic_loss.item()

    episode_rewards.append(episode_reward)
    episode_losses_actor.append(episode_actor_loss)
    episode_losses_critic.append(episode_critic_loss)

    if episode % 10 == 0:
        print(f"Episode {episode} — Reward: {episode_reward:.4f} — Critic Loss: {episode_critic_loss:.4f} — Actor Loss: {episode_actor_loss:.4f}")

# --- Plotting ---###
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(episode_rewards, color="green")
plt.title("Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(episode_losses_critic, color="red", label="Critic")
plt.plot(episode_losses_actor, color="blue", label="Actor")
plt.title("Training Losses")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
window = 5
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed, color="purple")
plt.title(f"Smoothed Reward ({window}-episode avg)")
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.grid(True)

plt.tight_layout()
plt.show()

