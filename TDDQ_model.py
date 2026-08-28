import torch
import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STATE_DIM = 10
ACTION_DIM = 4
HIDDEN_DIM = 128
REPLAY_CAPACITY = 10000
MODEL_BUFFER_CAPACITY = 2 * 64
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TOTAL_STEPS = 5000

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            torch.stack(s).to(device),
            torch.tensor(a, dtype=torch.long).unsqueeze(1).to(device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device),
            torch.stack(s_next).to(device),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

real_buffer = ReplayBuffer(REPLAY_CAPACITY)
model_buffer = ReplayBuffer(MODEL_BUFFER_CAPACITY)
action_embedding = nn.Embedding(ACTION_DIM, STATE_DIM).to(device)

class DynamicsModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, state_dim + 1)
        )

    def forward(self, state, embedded_action):
        x = torch.cat([state, embedded_action], dim=1)
        out = self.fc(x)
        next_state = out[:, :-1]
        reward = out[:, -1:]
        return next_state, reward

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, state, action):
        action_onehot = torch.nn.functional.one_hot(action.squeeze(), ACTION_DIM).float()
        x = torch.cat([state, action_onehot], dim=1)
        return self.fc(x)


dynamics_model = DynamicsModel(STATE_DIM).to(device)
critic = Critic(STATE_DIM, ACTION_DIM).to(device)
target_critic = Critic(STATE_DIM, ACTION_DIM).to(device)
target_critic.load_state_dict(critic.state_dict())

optimizer_dyn = optim.Adam(
    list(dynamics_model.parameters()) + list(action_embedding.parameters()), lr=LR
)
optimizer_critic = optim.Adam(critic.parameters(), lr=LR)

##---- Traning Phase ----##
def env_step(state, action):
    action_tensor = torch.tensor([action], dtype=torch.long).to(device)
    embedded_action = action_embedding(action_tensor).squeeze(0)
    next_state = state + 0.1 * embedded_action
    reward = -state.norm().item()
    done = False
    return next_state, reward, done


state = torch.randn(STATE_DIM).to(device)
loss_dyn = None
loss_critic = None


dyn_losses = []
critic_losses = []
steps = []

for step in range(TOTAL_STEPS):
    action = random.randint(0, ACTION_DIM - 1)
    next_state, reward, done = env_step(state, action)
    real_buffer.push(state, action, reward, next_state, done)
    state = next_state.clone()

   
    if len(real_buffer) >= BATCH_SIZE:
        s, a, r, s_next, _ = real_buffer.sample(BATCH_SIZE)
        embedded_a = action_embedding(a.squeeze())
        pred_next, pred_reward = dynamics_model(s, embedded_a)

        loss_dyn = nn.functional.mse_loss(pred_next, s_next) + nn.functional.mse_loss(pred_reward, r)
        optimizer_dyn.zero_grad()
        loss_dyn.backward(retain_graph=True)
        optimizer_dyn.step()

        with torch.no_grad():
            s_model, a_model, _, _, _ = real_buffer.sample(BATCH_SIZE)
            embedded_model_a = action_embedding(a_model.squeeze())
            s_next_model, r_model = dynamics_model(s_model, embedded_model_a)
            done_model = torch.zeros_like(r_model)

        model_buffer.buffer.clear()
        for i in range(BATCH_SIZE):
            model_buffer.push(s_model[i], a_model[i].item(), r_model[i].item(), s_next_model[i], done_model[i].item())

    if len(model_buffer) >= BATCH_SIZE:
        s, a, r, s_next, done = model_buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            next_actions = torch.randint(0, ACTION_DIM, (BATCH_SIZE, 1)).to(device)
            q_next = target_critic(s_next, next_actions)
            target_q = r + GAMMA * q_next * (1 - done)

        q = critic(s, a)
        loss_critic = nn.functional.mse_loss(q, target_q)
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

    tau = 0.005
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    if step % 100 == 0:
        dyn_str = f"{loss_dyn.item():.4f}" if loss_dyn is not None else "N/A"
        critic_str = f"{loss_critic.item():.4f}" if loss_critic is not None else "N/A"
        print(f"Step {step} — Dyn Loss: {dyn_str} — Critic Loss: {critic_str}")


        steps.append(step)
        dyn_losses.append(loss_dyn.item() if loss_dyn is not None else float('nan'))
        critic_losses.append(loss_critic.item() if loss_critic is not None else float('nan'))
        
## --- The Plots ----- ##
# --- Dynamics Loss ---##
plt.figure(figsize=(10, 4))
plt.plot(steps, dyn_losses, label="Dynamics Loss", alpha=0.5, color='blue')
smoothed_dyn = pd.Series(dyn_losses).rolling(window=10, min_periods=1).mean()
plt.plot(steps, smoothed_dyn, label="Smoothed", color='navy')
plt.title("Dynamics Model Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Critic Loss ---##
plt.figure(figsize=(10, 4))
plt.plot(steps, critic_losses, label="Critic Loss", alpha=0.5, color='orange')
smoothed_critic = pd.Series(critic_losses).rolling(window=10, min_periods=1).mean()
plt.plot(steps, smoothed_critic, label="Smoothed", color='red')
plt.title("Critic Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Compare Dynamics vs Critic ---##
plt.figure(figsize=(10, 4))
plt.plot(steps, smoothed_dyn, label="Dynamics Loss (smoothed)", color='blue')
plt.plot(steps, smoothed_critic, label="Critic Loss (smoothed)", color='red')
plt.title("Loss Comparison")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
