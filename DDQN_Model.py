import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class QuestionEnv:
    def __init__(self, encoder, questions, true_actions, semantics, valid_actions):
        self.encoder = encoder
        self.questions = questions
        self.true_actions = true_actions
        self.semantics = semantics
        self.valid_actions = valid_actions
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.index = 0

    def reset(self):
        self.index = (self.index + 1) % len(self.questions)
        tokens = self.tokenizer(self.questions[self.index], return_tensors="pt")
        with torch.no_grad():
            state = self.encoder(tokens["input_ids"], tokens["attention_mask"])
        return state.squeeze(0)

    def step(self, action):
        true_action = self.true_actions[self.index]
        sem = self.semantics[self.index]
        reward = compute_reward(action, true_action, sem, self.valid_actions)
        done = True  # one-step episode
        next_state = self.reset()
        return next_state, reward, done
    
class DDQNAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.net(state)
    

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)
    
questions = [ex['question'] for ex in processed_dataset]
unique_prototypes = sorted(set(ex['hazard_prototype'] for ex in processed_dataset))
prototype_to_id = {proto: i for i, proto in enumerate(unique_prototypes)}
valid_actions = list(prototype_to_id.values())
true_actions = [prototype_to_id[ex['hazard_prototype']] for ex in processed_dataset]
#true_actions
semantics_list = [ex['question_semantics'] for ex in processed_dataset]
#semantics_list
env = QuestionEnv(
    encoder=encoder,
    questions=questions,
    true_actions=true_actions,
    semantics=semantics_list,
    valid_actions=valid_actions
)

online_net = DDQNAgent(state_dim=768, action_dim=len(valid_actions))
target_net = DDQNAgent(state_dim=768, action_dim=len(valid_actions))
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-4)
buffer = ReplayBuffer()

#### Training The Model
gamma = 0.99
batch_size = 32
update_target_every = 50
for step in range(1000):
    print(f"\n🔁 Step {step} -----------------------------")

    state = env.reset()
    print(f"📥 Reset environment. State shape: {state.shape}")

    q_values = online_net(state)
    print(f"📊 Q-values from online_net: {q_values.detach().cpu().numpy()}")

    action = torch.argmax(q_values).item()
    print(f"🎯 Selected action: {action}")

    next_state, reward, done = env.step(action)
    print(f"📤 Step result — Reward: {reward}, Done: {done}")
    print(f"📥 Next state shape: {next_state.shape}")

    buffer.push(state, action, reward, next_state, done)
    print(f"📦 Buffer size: {len(buffer)}")

    if len(buffer) >= batch_size:
        print("🧪 Sampling batch from buffer...")
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        print(f"📚 Batch shapes — States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}")

        q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        print(f"🔍 Q-values for actions: {q_values.detach().cpu().numpy()}")

        next_actions = torch.argmax(online_net(next_states), dim=1)
        next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        print(f"🔮 Next Q-values from target_net: {next_q_values.detach().cpu().numpy()}")

        target = rewards + gamma * next_q_values * (1 - dones)
        print(f"🎯 Target values: {target.detach().cpu().numpy()}")

        loss = nn.MSELoss()(q_values, target.detach())
        print(f"📉 Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✅ Updated online_net parameters.")

    if step % update_target_every == 0:
        target_net.load_state_dict(online_net.state_dict())
        print("🔄 Synced target_net with online_net.")
        
## Plots for DDQN Model - Q-Value Distribution Over Time
q_value_history = []
q_value_history.append(q_values.detach().cpu().numpy())
q_array = np.array(q_value_history)  # shape: [steps, actions]

plt.figure(figsize=(12, 6))
sns.heatmap(q_array.T, cmap="viridis", cbar=True)
plt.xlabel("Training Step")
plt.ylabel("Action Index")
plt.title("Q-Value Distribution Over Time")
plt.show()

## Re Plots
reward_trace = []
reward_trace.append(reward)
plt.figure(figsize=(10, 4))
plt.plot(reward_trace, label="Reward")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.show()
