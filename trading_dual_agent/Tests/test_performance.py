# full_performance_test_corrected

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import TradingEnvWithRisk
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("=" * 80)

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

RM_PATH = "./models/rm_20260530_152900.zip"
risk_model = PPO.load(RM_PATH, device="cpu")
trader = PPO.load("./models/trader_final.zip", device="cpu")



# ============================================================
# FONCTION DE BACKTEST CORRIGÉE (PRIX BRUTS)
# ============================================================

def backtest(ticker, start, end):
    features, prices, macro = get_universal_data(ticker, start, end)
    if features is None or len(features) < 50:
        return None

    
    prices_raw = prices.flatten().astype(np.float32)

    # Normalisation des features uniquement
    f_mean = features.mean(axis=0)
    f_std = features.std(axis=0) + 1e-8
    features_norm = (features - f_mean) / f_std

    # Macro prête à l'emploi
    if macro is None or len(macro) < len(features):
        macro_ready = np.ones((len(features), 3), dtype=np.float32)
        macro_ready[:, 0] = 0.8
    else:
        macro_ready = macro[:len(features)].astype(np.float32)

    env = TradingEnvWithRisk(features_norm, prices_raw, risk_model, macro_ready)
    obs, _ = env.reset()

    portfolio = [10000.0]
    done = False
    step = 0

    while not done and step < len(features) - 1:
        action, _ = trader.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        nw = info.get("net_worth", portfolio[-1])
        portfolio.append(float(nw) if nw else portfolio[-1])
        step += 1

    env.close()

    agent_return = (portfolio[-1] - 10000) / 10000 * 100
    bh_return = (prices_raw[-1] - prices_raw[0]) / prices_raw[0] * 100

    # Calcul du Sharpe
    returns = []
    for i in range(1, len(portfolio)):
        ret = (portfolio[i] - portfolio[i-1]) / portfolio[i-1]
        returns.append(ret)
    returns = np.array(returns)
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252) if len(returns) > 5 else 0

    # Drawdown
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-9)
    max_dd = drawdown.max() * 100

    # Allocation moyenne
    # (à récupérer depuis l'environnement)

    return {
        "agent": agent_return,
        "bh": bh_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win": agent_return > bh_return
    }

# ============================================================
# SCÉNARIOS DE TEST
# ============================================================

scenarios = [
    # CRISES
    ("SPY", "2022-01-01", "2022-12-31", "Bear Market 2022", "CRISE"),
    ("META", "2022-01-01", "2022-12-31", "Tech Crash 2022", "CRISE"),
    ("NFLX", "2022-01-01", "2022-06-30", "Netflix Crash", "CRISE"),
    ("COIN", "2022-04-01", "2022-12-31", "Crypto Winter", "CRISE"),

    # BULL RUNS
    ("NVDA", "2023-01-01", "2024-12-31", "IA Bull Run", "BULL"),
    ("SPY", "2023-01-01", "2024-12-31", "Bull Market 2023-24", "BULL"),
    ("MSFT", "2023-01-01", "2024-12-31", "Tech Bull", "BULL"),

    # VOLATILITÉ
    ("TSLA", "2022-01-01", "2023-12-31", "Tesla Volatile", "VOLATILE"),
    ("GME", "2021-01-01", "2021-12-31", "Meme Stock", "VOLATILE"),

    # DÉFENSIFS
    ("KO", "2022-01-01", "2023-12-31", "Defensive KO", "DEFENSIF"),
    ("JNJ", "2022-01-01", "2023-12-31", "Defensive JNJ", "DEFENSIF"),
]

# ============================================================
# EXÉCUTION
# ============================================================

print("\n" + "=" * 60)
print(" TEST DE PERFORMANCE COMPLET")
print("=" * 60)

print(f"\n{'Scénario':<20} {'Ticker':<8} {'Période':<12} {'Type':<10} {'Dual%':>8} {'B&H%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'Win':>5}")
print("-" * 95)

results = []
for ticker, start, end, name, market_type in scenarios:
    r = backtest(ticker, start, end)
    if r:
        r["name"] = name
        r["market_type"] = market_type
        results.append(r)

        period = f"{start[:4]}-{end[:4]}"
        symbol = "✅" if r["win"] else "❌"
        print(f"{name:<20} {ticker:<8} {period:<12} {market_type:<10} {r['agent']:>+7.1f}% {r['bh']:>+7.1f}% {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}% {symbol:>5}")

# ============================================================
# STATISTIQUES
# ============================================================

print("\n" + "=" * 60)
print(" STATISTIQUES GLOBALES")
print("=" * 60)

if results:
    df = pd.DataFrame(results)
    wins = sum(1 for r in results if r["win"])
    total = len(results)

    print(f"\n  RÉSULTATS GLOBAUX ({total} scénarios):")
    print(f"     Win rate: {wins}/{total} ({wins/total*100:.0f}%)")
    print(f"     Performance Dual moyenne: {df['agent'].mean():+.1f}%")
    print(f"     Performance B&H moyenne: {df['bh'].mean():+.1f}%")
    print(f"     Sharpe moyen: {df['sharpe'].mean():.2f}")
    print(f"     Drawdown moyen: {df['max_dd'].mean():.1f}%")

    # Par type de marché
    print(f"\n   PAR TYPE DE MARCHÉ:")
    for market_type in df['market_type'].unique():
        subset = df[df['market_type'] == market_type]
        subset_wins = subset['win'].sum()
        print(f"     {market_type}: {int(subset_wins)}/{len(subset)} wins ({subset_wins/len(subset)*100:.0f}%) - Perf Dual: {subset['agent'].mean():+.1f}%")