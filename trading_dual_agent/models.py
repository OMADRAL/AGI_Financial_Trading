# ============================================================
# models.py - Architecture TCN+GRU
# ============================================================

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(2)]
        return self.relu(self.bn(out)) + self.res(x)

class TCN(nn.Module):
    def __init__(self, input_size, hidden=32, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(TCNBlock(input_size if i == 0 else hidden, hidden, dilation=2**i, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)

class TCNGRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, tcn_hidden=32, tcn_layers=3, gru_hidden=64, gru_layers=1, dropout=0.1, features_dim=128):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.tcn = TCN(obs_dim, tcn_hidden, tcn_layers, dropout)
        self.gru = nn.GRU(tcn_hidden, gru_hidden, gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = obs.unsqueeze(1)
        x = self.tcn(x)
        x, _ = self.gru(x)
        return self.head(x[:, -1, :])

def get_tcn_gru_kwargs(tcn_hidden=32, tcn_layers=3, gru_hidden=64, gru_layers=1, dropout=0.1, features_dim=128, net_arch=None):
    return {
        'features_extractor_class': TCNGRUExtractor,
        'features_extractor_kwargs': {
            'tcn_hidden': tcn_hidden,
            'tcn_layers': tcn_layers,
            'gru_hidden': gru_hidden,
            'gru_layers': gru_layers,
            'dropout': dropout,
            'features_dim': features_dim,
        },
        'net_arch': net_arch or [64, 64],
        'normalize_images': False,
    }