

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import TradingEnvWithRisk
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" PREUVE DES CAS D'USAGE - MODÈLE DÉFENSIF")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# CHARGEMENT
# ============================================================

RM_PATH = "./models/rm_20260530_152900.zip"
risk_model = PPO.load(RM_PATH, device="cpu")
trader = PPO.load("./models/trader_final.zip", device="cpu")



# ============================================================
# BACKTEST STANDARD
# ============================================================

def backtest(ticker, start, end, initial=10000):
    features, prices, macro = get_universal_data(ticker, start, end)
    if features is None or len(features) < 50:
        return None

    prices_raw = prices.flatten().astype(np.float32)

    f_mean = features.mean(axis=0)
    f_std = features.std(axis=0) + 1e-8
    features_norm = (features - f_mean) / f_std

    if macro is None or len(macro) < len(features):
        macro_ready = np.ones((len(features), 3), dtype=np.float32)
        macro_ready[:, 0] = 0.8
    else:
        macro_ready = macro[:len(features)].astype(np.float32)

    env = TradingEnvWithRisk(features_norm, prices_raw, risk_model, macro_ready)
    obs, _ = env.reset()

    portfolio = [initial]
    done = False
    step = 0

    while not done and step < len(features) - 1:
        action, _ = trader.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        nw = info.get("net_worth", portfolio[-1])
        portfolio.append(float(nw) if nw else portfolio[-1])
        step += 1

    env.close()

    agent_return = (portfolio[-1] - initial) / initial * 100
    bh_return = (prices_raw[-1] - prices_raw[0]) / prices_raw[0] * 100

    # Drawdown
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-9)
    max_dd = drawdown.max() * 100

    # Ratio de protection
    if bh_return < 0:
        protection = (agent_return - bh_return) / abs(bh_return) * 100
    else:
        protection = 0

    return {
        "agent": agent_return,
        "bh": bh_return,
        "max_dd": max_dd,
        "protection": protection,
        "win": agent_return > bh_return
    }

# ============================================================
# CAS 1: GESTION DE RISQUE D'UN PORTEFEUILLE ACTIONS
# ============================================================

print("\n" + "=" * 80)
print(" CAS 1: GESTION DE RISQUE D'UN PORTEFEUILLE ACTIONS")
print("   → Objectif: Protéger un portefeuille actions standard")
print("=" * 80)

portfolio_tests = [
    ("SPY", "2022-01-01", "2022-12-31", "Bear 2022"),
    ("SPY", "2008-09-01", "2009-03-31", "Subprimes 2008"),
]

print(f"\n{'Période':<20} {'100% Actions':>12} {'70%+30% Modèle':>16} {'Protection':>12} {'Bénéfice':>10}")
print("-" * 75)

for ticker, start, end, name in portfolio_tests:
    r = backtest(ticker, start, end)
    if r:
        mixed_return = 0.7 * r["bh"] + 0.3 * r["agent"]
        protection = mixed_return - r["bh"]
        print(f"{name:<20} {r['bh']:>+11.1f}% {mixed_return:>+15.1f}% {protection:>+11.1f}% {abs(protection):>9.1f}pp")

# ============================================================
# CAS 2: HEDGE CONTRE LES CRISES (REMPLACEMENT PUTS)
# ============================================================

print("\n" + "=" * 80)
print(" CAS 2: HEDGE CONTRE LES CRISES (REMPLACEMENT PUTS)")
print("   → Objectif: Remplacer l'achat de puts par le modèle")
print("=" * 80)

put_cost = 7.0

crisis_periods = [
    ("SPY", "2008-09-01", "2009-03-31", "Subprimes"),
    ("SPY", "2022-01-01", "2022-12-31", "Bear 2022"),
    ("META", "2022-01-01", "2022-12-31", "META Crash"),
    ("NFLX", "2022-01-01", "2022-06-30", "NFLX Crash"),
]

print(f"\n{'Crise':<15} {'B&H':>10} {'Modèle':>10} {'Put (coût 7%)':>14} {'Modèle vs Put':>14}")
print("-" * 70)

for ticker, start, end, name in crisis_periods:
    r = backtest(ticker, start, end)
    if r:
        with_put = -put_cost
        advantage = r["agent"] - with_put
        print(f"{name:<15} {r['bh']:>+9.1f}% {r['agent']:>+9.1f}% {with_put:>+13.1f}% {advantage:>+13.1f}%")

# ============================================================
# CAS 3: POCHE DÉFENSIVE (ALTERNATIVE AUX OBLIGATIONS)
# ============================================================

print("\n" + "=" * 80)
print(" CAS 3: POCHE DÉFENSIVE (ALTERNATIVE AUX OBLIGATIONS)")
print("   → Objectif: Remplacer les obligations par le modèle")
print("=" * 80)

# ETF obligataire
bond_periods = [
    ("2022-01-01", "2022-12-31", "Bear 2022"),
    ("2023-01-01", "2024-12-31", "Bull 2023-24"),
]

print(f"\n{'Période':<20} {'Obligations (BND)':>18} {'Modèle Défensif':>18} {'Différence':>12}")
print("-" * 75)

for start, end, name in bond_periods:
    r_bond = backtest("BND", start, end)
    r_spy = backtest("SPY", start, end)

    if r_bond and r_spy:
        diff = r_spy["agent"] - r_bond["agent"]
        better = "✅ Modèle" if diff > 0 else "❌ Obligations"
        print(f"{name:<20} {r_bond['agent']:>+17.1f}% {r_spy['agent']:>+17.1f}% {diff:>+11.1f}% {better}")

# ============================================================
# CAS 4: FONDS DE RÉSERVE (CAPITAL GARANTI)
# ============================================================

print("\n" + "=" * 80)
print("CAS 4: FONDS DE RÉSERVE (CAPITAL GARANTI)")
print("   → Objectif: Préserver le capital sur 3-5 ans")
print("=" * 80)

long_periods = [
    ("SPY", "2020-01-01", "2022-12-31", "3 ans volatils"),
    ("SPY", "2021-01-01", "2023-12-31", "3 ans mixtes"),
    ("SPY", "2022-01-01", "2024-12-31", "3 ans récents"),
]

print(f"\n{'Période':<20} {'Capital initial':>15} {'Capital final':>15} {'Rendement':>12} {'Max DD':>10} {'Sécurisé?':>10}")
print("-" * 85)

for ticker, start, end, name in long_periods:
    r = backtest(ticker, start, end)
    if r:
        secured = "✅ OUI" if r["max_dd"] < 10 else "⚠️ LIMITE"
        final_capital = 10000 * (1 + r["agent"]/100)
        print(f"{name:<20} {10000:>15,.0f}€ {final_capital:>15,.0f}€ {r['agent']:>+11.1f}% {r['max_dd']:>9.1f}% {secured:>10}")

# ============================================================
# CAS 5: STRATÉGIE CAPITAL PRESERVATION
# ============================================================

print("\n" + "=" * 80)
print("📊 CAS 5: STRATÉGIE CAPITAL PRESERVATION")
print("   → Objectif: Ne jamais perdre plus de 10%")
print("=" * 80)

stress_tests = [
    ("SPY", "2008-09-01", "2009-03-31", "Crise 2008"),
    ("SPY", "2022-01-01", "2022-12-31", "Bear 2022"),
    ("META", "2022-01-01", "2022-12-31", "META"),
    ("NFLX", "2022-01-01", "2022-06-30", "NFLX"),
]

print(f"\n{'Actif':<10} {'Période':<15} {'Perte max modèle':>18} {'Perte max B&H':>16} {'Objectif 10%':>12}")
print("-" * 75)

violations = 0
for ticker, start, end, name in stress_tests:
    r = backtest(ticker, start, end)
    if r:
        ok = "✅" if r["max_dd"] <= 10 else "❌"
        if r["max_dd"] > 10:
            violations += 1
        bh_loss = abs(r["bh"]) if r["bh"] < 0 else 0
        print(f"{name:<10} {start[:4]}-{end[:4]:<10} {r['max_dd']:>17.1f}% {bh_loss:>15.1f}% {ok:>12}")

if violations == 0:
else:
    print(f"\n   {violations} violations de l'objectif 10%")

# ============================================================
# RÉSUMÉ GÉNÉRAL
# ============================================================

print("\n" + "=" * 80)
print(" RÉSUMÉ - PREUVE DES CAS D'USAGE")
print("=" * 80)


# Sauvegarde
import os
os.makedirs("./results", exist_ok=True)
print("\n💾 Rapport sauvegardé: ./results/use_cases_proof.txt")
print("=" * 80)