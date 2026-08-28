from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

policy_net = QNetwork(obs_dim_text=384, obs_dim_struct=384, action_dim=5).to(device)
target_net = QNetwork(obs_dim_text=384, obs_dim_struct=384, action_dim=5).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_text, obs_struct, action, reward, next_obs_text, next_obs_struct, done):
        self.buffer.append((obs_text, obs_struct, action, reward, next_obs_text, next_obs_struct, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_text, obs_struct, actions, rewards, next_text, next_struct, dones = zip(*batch)
        return (
            torch.stack(obs_text),
            torch.stack(obs_struct),
            torch.tensor(actions).unsqueeze(1),
            torch.tensor(rewards).unsqueeze(1),
            torch.stack(next_text),
            torch.stack(next_struct),
            torch.tensor(dones).unsqueeze(1)
        )

replay_buffer = ReplayBuffer(capacity=10000)
for sample in synthetic_dataset:
    try:
        obs_text, obs_struct, decision, _ = encode_sample(sample)
        next_sample = generate_sample()
        next_obs_text, next_obs_struct, _, _ = encode_sample(next_sample)

        # Map decision to action index
        if sample["decision_options"] not in action_vocab:
            action_vocab[sample["decision_options"]] = len(action_vocab)
        action = action_vocab[sample["decision_options"]]

        with torch.no_grad():
            reward = reward_model(
                obs_text.unsqueeze(0).to(device),
                obs_struct.unsqueeze(0).to(device),
                decision.unsqueeze(0).to(device)
            ).item()

        done = False  

        replay_buffer.push(
            obs_text, obs_struct, action, reward,
            next_obs_text, next_obs_struct, done
        )

    except Exception as e:
        print(f"Skipping sample due to error: {e}")
        continue 
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_text, obs_struct, action, reward, next_text, next_struct, done):
        self.buffer.append((obs_text, obs_struct, action, reward, next_text, next_struct, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_text, obs_struct, actions, rewards, next_text, next_struct, dones = zip(*batch)

        return (
            torch.stack(obs_text),  # shape: [B, ...]
            torch.stack(obs_struct),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.stack(next_text),
            torch.stack(next_struct),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
    
####-- Applying The Transition Condition 
def get_structured_transition():
    sample = next(data_iterator)  
    obs_text, obs_struct, decision, _ = encode_sample(sample)

    next_sample = generate_sample() 
    next_obs_text, next_obs_struct, _, _ = encode_sample(next_sample)

    decision_key = str(sample["decision_options"])
    if decision_key not in action_vocab:
        action_vocab[decision_key] = len(action_vocab)
    action = action_vocab[decision_key]

    with torch.no_grad():
        reward = reward_model(
            obs_text.unsqueeze(0).to(device),
            obs_struct.unsqueeze(0).to(device),
            decision.unsqueeze(0).to(device)
        ).item()

    done = False 
    return obs_text, obs_struct, action, reward, next_obs_text, next_obs_struct, done 


plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class QNetwork(nn.Module):
    def __init__(self, text_dim, struct_dim, hidden_dim, num_actions):
        super().__init__()
        self.fc_text = nn.Linear(text_dim, hidden_dim)
        self.fc_struct = nn.Linear(struct_dim, hidden_dim)
        self.fc_combined = nn.Linear(hidden_dim * 2, num_actions)

    def forward(self, obs_text, obs_struct):
        x_text = torch.relu(self.fc_text(obs_text))
        x_struct = torch.relu(self.fc_struct(obs_struct))
        x = torch.cat([x_text, x_struct], dim=1)
        return self.fc_combined(x)


TEXT_DIM = 384
STRUCT_DIM = 384
HIDDEN_DIM = 64
NUM_ACTIONS = len(action_vocab)
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE_FREQ = 10
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1 
EPSILON = max(EPSILON_MIN, EPSILON - 1e-4)
REPLAY_CAPACITY = 10000
TOTAL_STEPS = 5000

policy_net = QNetwork(TEXT_DIM, STRUCT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)
target_net = QNetwork(TEXT_DIM, STRUCT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
synthetic_dataset = [generate_sample() for _ in range(5000)]
action_vocab = {}

def get_real_transition(sample):
    obs_text, obs_struct, decision_tensor, _ = encode_sample(sample)
    next_sample = generate_sample()
    next_obs_text, next_obs_struct, _, _ = encode_sample(next_sample)
    decision_key = str(sample["decision_options"])
    if decision_key not in action_vocab:
        action_vocab[decision_key] = len(action_vocab)
    action = action_vocab[decision_key]

    # Predict reward
    with torch.no_grad():
        reward_tensor = reward_model(
            obs_text.unsqueeze(0).to(device),
            obs_struct.unsqueeze(0).to(device),
            decision_tensor.unsqueeze(0).to(device)
        )
    reward = reward_tensor.item()

    done = False
    return obs_text, obs_struct, action, reward, next_obs_text, next_obs_struct, done
for step in range(TOTAL_STEPS):
    sample = synthetic_dataset[step % len(synthetic_dataset)]
    transition = get_real_transition(sample)
    replay_buffer.push(*transition)

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

        # --- Q-value computation ---
        q_values = policy_net(obs_text_b, obs_struct_b).gather(1, actions_b)

        with torch.no_grad():
            next_actions = policy_net(next_text_b, next_struct_b).argmax(dim=1, keepdim=True)
            next_q_values = target_net(next_text_b, next_struct_b).gather(1, next_actions)
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
        
### Same Variant --- From here---###
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class QNetwork(nn.Module):
    def __init__(self, text_dim, struct_dim, hidden_dim, num_actions):
        super().__init__()
        self.fc_text = nn.Linear(text_dim, hidden_dim)
        self.fc_struct = nn.Linear(struct_dim, hidden_dim)
        self.fc_combined = nn.Linear(hidden_dim * 2, num_actions)

    def forward(self, obs_text, obs_struct):
        x_text = torch.relu(self.fc_text(obs_text))
        x_struct = torch.relu(self.fc_struct(obs_struct))
        x = torch.cat([x_text, x_struct], dim=1)
        return self.fc_combined(x)

TEXT_DIM = 384
STRUCT_DIM = 384
HIDDEN_DIM = 64
NUM_ACTIONS = len(action_vocab)
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE_FREQ = 10
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1 
EPSILON = max(EPSILON_MIN, EPSILON - 1e-4)
REPLAY_CAPACITY = 10000
TOTAL_STEPS = 5000

policy_net = QNetwork(TEXT_DIM, STRUCT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)
target_net = QNetwork(TEXT_DIM, STRUCT_DIM, HIDDEN_DIM, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
synthetic_dataset = [generate_sample() for _ in range(5000)]
action_vocab = {}

def get_real_transition(sample):
    obs_text, obs_struct, decision_tensor, _ = encode_sample(sample)
    next_sample = generate_sample()
    next_obs_text, next_obs_struct, _, _ = encode_sample(next_sample)
    decision_key = str(sample["decision_options"])
    if decision_key not in action_vocab:
        action_vocab[decision_key] = len(action_vocab)
    action = action_vocab[decision_key]

    with torch.no_grad():
        reward_tensor = reward_model(
            obs_text.unsqueeze(0).to(device),
            obs_struct.unsqueeze(0).to(device),
            decision_tensor.unsqueeze(0).to(device)
        )
    reward = reward_tensor.item()

    done = False
    return obs_text, obs_struct, action, reward, next_obs_text, next_obs_struct, done

for step in range(TOTAL_STEPS):
    sample = synthetic_dataset[step % len(synthetic_dataset)]
    transition = get_real_transition(sample)
    replay_buffer.push(*transition)

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

        # --- Q-value computation ---
        q_values = policy_net(obs_text_b, obs_struct_b).gather(1, actions_b)

        with torch.no_grad():
            next_actions = policy_net(next_text_b, next_struct_b).argmax(dim=1, keepdim=True)
            next_q_values = target_net(next_text_b, next_struct_b).gather(1, next_actions)
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
        
        