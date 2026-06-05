import yfinance as yf
import pandas as pd
import numpy as np

def get_universal_data(ticker, start_date, end_date, seq_len=30):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None, None, None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    closes  = df['Close'].values.flatten()
    volumes = df['Volume'].values.flatten()
    highs   = df['High'].values.flatten()
    lows    = df['Low'].values.flatten()

    sequences, prices = [], []

    for i in range(seq_len, len(closes) - 1):
        window_close  = closes[i - seq_len:i]
        window_volume = volumes[i - seq_len:i]
        window_high   = highs[i - seq_len:i]
        window_low    = lows[i - seq_len:i]

        # Scaled stationary features
        log_returns = np.log(window_close[1:] / (window_close[:-1] + 1e-9)) * 50.0  
        hl_ratio = ((window_high[1:] - window_low[1:]) / (window_close[1:] + 1e-9)) * 10.0
        
        vol_mean = window_volume.mean() + 1e-9
        rel_volume = (window_volume[1:] / vol_mean) - 1.0

        w_max = np.max(window_close)
        w_min = np.min(window_close)
        price_position = ((window_close[1:] - w_min) / (w_max - w_min + 1e-9)) * 2.0 - 1.0

        features = np.stack([log_returns, hl_ratio, rel_volume, price_position], axis=1)
        features = np.clip(features, -5.0, 5.0)

        sequences.append(features.astype(np.float32))
        prices.append(closes[i])

    if len(sequences) == 0:
        return None, None, None

    return np.array(sequences, dtype=np.float32), np.array(prices, dtype=np.float32), df