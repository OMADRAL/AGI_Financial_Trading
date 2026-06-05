import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UniversalTradingEnv(gym.Env):
    def __init__(self, sequences, prices, initial_balance=10000, transaction_fee=0.0015):
        super().__init__()
        self.sequences = sequences   
        self.prices = prices      
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        seq_len, input_dim = sequences.shape[1], sequences.shape[2]

        # Action space: -1 to 1 continuous output from PPO
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(seq_len * input_dim + 2,), dtype=np.float32
        )

    def reset(self, seed=None):
        self.current_step = 0
        self.position = 0.0
        self.net_worth = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        seq = self.sequences[self.current_step].flatten()
        nw_ratio = np.log(self.net_worth / self.initial_balance)
        ctx = np.array([self.position, nw_ratio], dtype=np.float32)
        return np.concatenate([seq, ctx]).astype(np.float32)

    def step(self, action):
        prev_nw = self.net_worth
        current_price = self.prices[self.current_step]
        
        # LONG OR CASH ONLY
        raw_action = action[0]
        if raw_action > 0.0: 
            desired_pos = 1.0  # 100% Long
        else: 
            desired_pos = 0.0  # 100% Cash

        # Apply transaction fees only on position changes
        if desired_pos != self.position:
            fee_pct = self.transaction_fee * abs(desired_pos - self.position)
            self.net_worth -= self.net_worth * fee_pct
            self.position = desired_pos

        # Advance time & calculate PnL
        self.current_step += 1
        next_price = self.prices[self.current_step]
        asset_return = (next_price - current_price) / current_price
        
        self.net_worth += self.net_worth * (self.position * asset_return)

        # -------------------------------------------------------------
        #  Pure Symmetric Reward
        # No artificial fear factor. The agent learns true market dynamics.
        # -------------------------------------------------------------
        step_return = (self.net_worth - prev_nw) / prev_nw
        reward = step_return * 100.0  

        reward = np.clip(reward, -5.0, 5.0)

        done = (self.current_step >= len(self.prices) - 1)
        info = {"net_worth": self.net_worth, "position": self.position}
        
        return self._get_obs(), float(reward), done, False, info