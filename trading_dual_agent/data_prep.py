# ============================================================
# data_prep.py - Version OPTIMALE
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf

def get_universal_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df is None or len(df) < 60:
            return None, None, None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.dropna(inplace=True)
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        
        # Features techniques (13 indicateurs)
        feat = pd.DataFrame(index=df.index)
        feat["ret_1d"] = close.pct_change(1)
        feat["ret_5d"] = close.pct_change(5)
        feat["ret_20d"] = close.pct_change(20)
        feat["vol_10"] = feat["ret_1d"].rolling(10).std()
        feat["vol_20"] = feat["ret_1d"].rolling(20).std()
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        feat["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        feat["price_ma20"] = (close - ma20) / (ma20 + 1e-9)
        feat["price_ma50"] = (close - ma50) / (ma50 + 1e-9)
        feat["ma20_ma50"] = (ma20 - ma50) / (ma50 + 1e-9)
        
        vol_ma = volume.rolling(20).mean()
        feat["vol_ratio"] = volume / (vol_ma + 1e-9)
        
        std20 = close.rolling(20).std()
        bb_up = ma20 + 2 * std20
        bb_lo = ma20 - 2 * std20
        feat["bb_pos"] = (close - bb_lo) / (bb_up - bb_lo + 1e-9)
        
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        atr = (high - low).rolling(14).mean()
        feat["macd"] = (ema12 - ema26) / (atr + 1e-9)
        
        feat["mom_10"] = close.pct_change(10)
        feat.dropna(inplace=True)
        
        prices = close.loc[feat.index].values.astype(np.float32)
        macro = _get_macro(start_date, end_date, feat.index)
        
        return feat.values.astype(np.float32), prices, macro
        
    except Exception as e:
        return None, None, None

def _get_macro(start_date, end_date, index):
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"].squeeze()
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"].squeeze()
        
        df = pd.DataFrame({"vix": vix, "spy": spy}, index=vix.index).dropna()
        vix_norm = df["vix"] / 20.0
        spy_ret = df["spy"].pct_change().fillna(0)
        corr = vix_norm.rolling(20).corr(spy_ret).fillna(0)
        
        macro = pd.DataFrame({"vix_norm": vix_norm, "spy_ret": spy_ret, "corr": corr})
        macro = macro.reindex(index, method="ffill").fillna(0)
        return macro.values.astype(np.float32)
    except:
        return np.zeros((len(index), 3), dtype=np.float32)

def get_multi_stock_data(tickers, periods):
    all_f, all_p, all_m = [], [], []
    for ticker in tickers:
        for start, end in periods:
            f, p, m = get_universal_data(ticker, start, end)
            if f is not None and len(f) > 100:
                all_f.append(f); all_p.append(p); all_m.append(m)
                print(f"✅ {ticker} {start[:4]}-{end[:4]}: {len(f)} jours")
    
    features = np.vstack(all_f).astype(np.float32)
    prices = np.concatenate(all_p).astype(np.float32)
    min_len = min(len(m) for m in all_m)
    macro = np.mean([m[:min_len] for m in all_m], axis=0).astype(np.float32)
    return features, prices, macro