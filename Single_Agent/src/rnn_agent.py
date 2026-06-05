import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class UniversalMarketRNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, seq_len=29, input_dim=4, features_dim=64):
        # +2 for the context variables (position and PnL)
        super().__init__(observation_space, features_dim=features_dim + 2)
        self.seq_len = seq_len
        self.input_dim = input_dim 
        # 1. 1D Convolution (Extracts local patterns, filters noise)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # 2. GRU (Sequential memory, respects time-decay)
        self.gru = nn.GRU(input_size=32, hidden_size=features_dim, num_layers=1, batch_first=True)
    def forward(self, obs):
        batch_size = obs.shape[0]
        # Split observation into sequence data and context data
        seq_size = self.seq_len * self.input_dim
        seq = obs[:, :seq_size]
        ctx = obs[:, seq_size:]  # Shape: (batch, 2)  
        # Reshape for Conv1D: (batch, channels, length)
        x = seq.view(batch_size, self.seq_len, self.input_dim)
        x = x.permute(0, 2, 1)    
        # Apply Convolution
        x = self.relu(self.conv1(x))  
        # Reshape for GRU: (batch, length, channels)
        x = x.permute(0, 2, 1)
        # Pass through GRU
        _, hidden = self.gru(x)
        market_context = hidden[-1]  
        # Combine market understanding with agent's current position/PnL
        return torch.cat([market_context, ctx], dim=1)