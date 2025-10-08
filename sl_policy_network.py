import torch.nn as nn
import torch.nn.functional as F
from dlgo.agent.base import Agent
import torch

__all__ = [
    'SLPolicyNetwork',
    'SLPolicyBot'
]

class SLPolicyNetwork(nn.Module):
    def __init__(self, features=48, filters=192):
        super(SLPolicyNetwork, self).__init__()
        self.features = features
        self.filters = filters
        self.first_layer = nn.Conv2d(self.features, self.filters, kernel_size=5, stride=1, padding=2)

        self.hidden_layers = nn.ModuleList([
            nn.Conv2d(filters, filters, kernel_size=3, padding=1)
            for _ in range(11)
        ])

        self.final_layer = nn.Conv2d(self.filters, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # Logits are the unnormalized scores output by the network for each possible move before softmax
        policy_logits = self.final_layer(x)  # (batch, 1, board_size, board_size)
        
        # Reshape and apply log_softmax for NLLLoss compatibility
        batch_size = policy_logits.size(0)
        policy_logits = policy_logits.view(batch_size, -1)
        log_probs = F.log_softmax(policy_logits, dim=1)
        
        return log_probs

class SLPolicyBot(Agent):
    def __init__(self, model):
        self.model = model

    def select_move(self, game_state):
        self.model.eval()
        with torch.no_grad():
            encoded_state = self.encoder.encode(game_state)
            policy, _ = self.model(encoded_state)
            action = torch.argmax(policy, dim=1)
            return action
