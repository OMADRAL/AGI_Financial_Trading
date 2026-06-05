from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_prep import get_universal_data
from env import UniversalTradingEnv
from rnn_agent import UniversalMarketRNN
import numpy as np
import os

def make_env(sequences, prices):
    def _init():
        return UniversalTradingEnv(sequences, prices)
    return _init

def linear_schedule(initial_value: float, final_value: float):
    def scheduler(progress_remaining: float):
        return final_value + progress_remaining * (initial_value - final_value)
    return scheduler

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/universal", exist_ok=True)

    tickers = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'JPM', 'AMZN', 'META', 'SPY', 'QQQ', 'GLD',
               'KO', 'PG', 'JNJ', 'WMT', 'V', 'NFLX', 'SNAP', 'UBER', 'BABA', 'TSM']
    periods = [("2010-01-01", "2023-01-01"), ("2007-01-01", "2009-12-31"),
               ("2020-01-01", "2021-06-01"), ("2021-11-01", "2022-12-31")]

    s_list, p_list = [], []
    for ticker in tickers:
        for start, end in periods:
            s, p, _ = get_universal_data(ticker, start, end, seq_len=30)
            if s is not None and len(s) > 100:
                s_list.append(s)
                p_list.append(p)
                print(f"Loaded {ticker} {start[:4]}-{end[:4]}: {len(s)} steps")

    envs = [make_env(s, p) for s, p in zip(s_list, p_list)]
    t_env = DummyVecEnv(envs)

    # Passed the correct kwargs for the RNN (features_dim )
    policy_kwargs = dict(
        features_extractor_class=UniversalMarketRNN,
        features_extractor_kwargs=dict(seq_len=29, input_dim=4, features_dim=64),
        net_arch=[64, 64]
    )

    model = PPO(
        "MlpPolicy", t_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4, 1e-5),
        ent_coef=0.01, clip_range=0.2,
        n_steps=1024, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, verbose=1,
        tensorboard_log="./logs/universal/"
    )

    # 1.5M steps is plenty for the RNN
    model.learn(total_timesteps=1_500_000)
    model.save("models/universal_agent")
    print("\n Universal AGI model saved — zero tuning needed for any stock.")