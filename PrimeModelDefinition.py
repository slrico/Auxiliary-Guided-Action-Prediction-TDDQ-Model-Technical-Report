import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

class QuestionEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

class RLAgent(nn.Module):
    def __init__(self, encoder, hidden_dim=768, num_actions=5):
        super().__init__()
        self.encoder = encoder
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, input_ids, attention_mask):
        embedding = self.encoder(input_ids, attention_mask)
        logits = self.policy_head(embedding)
        return logits
    
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
question = "What should be done in case of a fire?"
tokens = tokenizer(question, return_tensors="pt")

input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

def compute_reward(pred_action, true_action, semantics, valid_actions):
    reward = 0
    if pred_action == true_action:
        reward += 1
    else:
        reward -= 1

    if pred_action in semantics['entities'] or pred_action in semantics['actions']:
        reward += 0.5

    if pred_action not in valid_actions:
        reward -= 1.5

    return reward

encoder = QuestionEncoder()
agent = RLAgent(encoder)
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

logits = agent(input_ids, attention_mask)
action_dist = torch.distributions.Categorical(logits=logits)
action = action_dist.sample()


true_action = 2
semantics = {
    'entities': [1, 2],
    'actions': [3]
}
valid_actions = list(range(5))


reward = compute_reward(action.item(), true_action, semantics, valid_actions)
loss = -action_dist.log_prob(action) * reward
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"🎲 Action: {action.item()} | 🏆 Reward: {reward} | 📉 Loss: {loss.item():.4f}")


rewards_log = []
losses_log = []
actions_log = []

num_steps = 50
for step in range(num_steps):
    logits = agent(input_ids, attention_mask)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()

    reward = compute_reward(action.item(), true_action, semantics, valid_actions)
    loss = -action_dist.log_prob(action) * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log values
    rewards_log.append(reward)
    losses_log.append(loss.item())
    actions_log.append(action.item())

    print(f"Step {step+1:02d} | 🎲 Action: {action.item()} | 🏆 Reward: {reward} | 📉 Loss: {loss.item():.4f}")
    
#### Plots of loss and Cross Entropy 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_log, label="Reward", color="green")
plt.title("Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses_log, label="Loss", color="red")
plt.title("Loss Over Time")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


entropy = action_dist.entropy().mean()
loss = -action_dist.log_prob(action) * reward - 0.01 * entropy

plt.hist(actions_log, bins=len(valid_actions), color="skyblue", edgecolor="black")
plt.title("Action Distribution")
plt.xlabel("Action ID")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()