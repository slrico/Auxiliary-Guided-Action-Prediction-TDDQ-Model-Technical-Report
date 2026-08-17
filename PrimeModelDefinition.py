import torch
from torch import nn
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