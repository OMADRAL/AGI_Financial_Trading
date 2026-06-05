# ============================================================
# env.py - Version OPTIMALE
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class UniversalTradingEnv(gym.Env):
    def __init__(self, features, prices, initial_capital=10000, transaction_cost=0.0005):
        super().__init__()
        self.features = np.array(features, dtype=np.float32)
        self.prices = np.array(prices, dtype=np.float32)
        self.initial_capital = float(initial_capital)
        self.transaction_cost = transaction_cost
        self.n_steps = len(self.features)
        self.n_features = self.features.shape[1]
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features + 3,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self.n_steps - 252 - 1
        self.current_step = np.random.randint(50, max_start) if max_start > 50 else 50
        self.end_step = min(self.current_step + 252, self.n_steps - 1)
        self.net_worth = self.initial_capital
        self.position = 0.0
        self.prev_net_worth = self.initial_capital
        self.returns_hist = []
        self.best_net_worth = self.initial_capital
        return self._get_obs(), {}
    
    def _get_obs(self):
        idx = min(self.current_step, self.n_steps - 1)
        obs = self.features[idx].copy()
        obs = np.clip(obs, -5, 5)
        
        vol = np.std(self.returns_hist[-10:]) * np.sqrt(252) if len(self.returns_hist) >= 10 else 0.02
        obs = np.append(obs, np.clip(vol, 0, 0.5))
        
        if self.net_worth > self.best_net_worth:
            self.best_net_worth = self.net_worth
        drawdown = (self.best_net_worth - self.net_worth) / (self.best_net_worth + 1e-9)
        obs = np.append(obs, drawdown)
        obs = np.append(obs, self.position)
        return obs.astype(np.float32)
    
    def step(self, action):
        new_pos = float(np.clip(action[0], -1.0, 1.0))
        price_now = self.prices[self.current_step]
        price_next = self.prices[min(self.current_step + 1, self.n_steps - 1)]
        
        delta = abs(new_pos - self.position)
        cost = delta * self.transaction_cost
        pct_change = (price_next - price_now) / (price_now + 1e-9)
        pnl = self.position * pct_change - cost
        
        self.net_worth *= (1 + pnl)
        self.net_worth = max(self.net_worth, 0.01)
        
        daily_ret = (self.net_worth - self.prev_net_worth) / (self.prev_net_worth + 1e-9)
        self.returns_hist.append(daily_ret)
        self.prev_net_worth = self.net_worth
        
        if len(self.returns_hist) >= 20:
            recent = np.array(self.returns_hist[-20:])
            sharpe = np.mean(recent) / (np.std(recent) + 1e-9) * np.sqrt(252)
            reward = sharpe * 0.1
        else:
            reward = daily_ret * 10
        
        self.position = new_pos
        self.current_step += 1
        done = self.current_step >= self.end_step
        
        info = {"net_worth": self.net_worth, "position": self.position, "daily_return": daily_ret}
        return self._get_obs(), reward, done, False, info