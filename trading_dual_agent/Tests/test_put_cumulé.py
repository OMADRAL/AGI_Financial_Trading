
# ============================================================

import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import TradingEnvWithRisk
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("📊 TEST - IMPACT DE LA DURÉE SUR L'AVANTAGE DUAL AGENT vs PUT")
print("   Calcul de l'avantage cumulé sur 1, 2, 3 et 5 ans")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

RM_PATH = "./models/rm_20260530_152900.zip"
risk_model = PPO.load(RM_PATH, device="cpu")
trader = PPO.load("./models/trader_final.zip", device="cpu")

print("\n✅ Modèles chargés")

# ============================================================
# FONCTION DE BACKTEST
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

    return {"agent": agent_return, "bh": bh_return}

# ============================================================
# CONFIGURATION DES TESTS
# ============================================================

# Coût annuel du put (standard)
PUT_ANNUAL_COST = 7.0

# Périodes de test (même crise, durées différentes)
# On utilise SPY Bear 2022 comme référence
test_periods = [
    ("SPY", "2022-01-01", "2022-12-31", "1 an", "Bear 2022"),
    ("SPY", "2021-01-01", "2022-12-31", "2 ans", "Période 2021-2022"),
    ("SPY", "2020-01-01", "2022-12-31", "3 ans", "Période 2020-2022"),
    ("SPY", "2018-01-01", "2022-12-31", "5 ans", "Période 2018-2022"),
]

print("\n" + "=" * 80)
print(" TEST SUR PÉRIODES DE DIFFÉRENTES DURÉES")
print("   Mesure de la performance du Dual Agent vs Put cumulé")
print("=" * 80)

print(f"\n{'Durée':<8} {'Période':<25} {'Dual Agent':>12} {'Put (coût annuel)':>18} {'Avantage':>12}")
print("-" * 80)

results = []
for ticker, start, end, duration, desc in test_periods:
    r = backtest(ticker, start, end)
    if r:
        # Coût total du put = coût annuel × nombre d'années
        years = int(duration[0]) if duration[0].isdigit() else 1
        total_put_cost = -PUT_ANNUAL_COST * years
        advantage = r["agent"] - total_put_cost

        print(f"{duration:<8} {desc:<25} {r['agent']:>+11.1f}% {total_put_cost:>+17.1f}% +{advantage:>10.1f} pts")

        results.append({
            "duration": duration,
            "years": years,
            "agent": r["agent"],
            "put_cost": total_put_cost,
            "advantage": advantage
        })

# ============================================================
# AFFICHAGE DU TABLEAU RÉCAPITULATIF
# ============================================================

print("\n" + "=" * 80)
print("TABLEAU RÉCAPITULATIF - AVANTAGE CUMULÉ")
print("=" * 80)

print("""
┌─────────┬─────────────────┬─────────────────┬─────────────────┐
│ Durée   │ Put (coût 7%/an) │ Dual Agent      │ Avantage        │
├─────────┼─────────────────┼─────────────────┼─────────────────┤""")

for r in results:
    print(f"│ {r['duration']:<5}  │ {r['put_cost']:>+14.1f}% │ {r['agent']:>+14.1f}% │ +{r['advantage']:>12.1f} pts │")

print("└─────────┴─────────────────┴─────────────────┴─────────────────┘")

# ============================================================
# VÉRIFICATION SUR PLUSIEURS CRISES (ROBUSTESSE)
# ============================================================

print("\n" + "=" * 80)
print(" VÉRIFICATION DE ROBUSTESSE - SUR PLUSIEURS CRISES")
print("   Moyenne des performances sur 3 crises différentes")
print("=" * 80)

# Utilisation de la performance moyenne du Dual Agent sur crises
crisis_performances = {
    "Subprimes 2008": 0.0,
    "Bear 2022": -1.8,
    "META Crash": -5.6,
    "NFLX Crash": -2.6,
    "TSLA Bear": -1.9
}

avg_dual_crisis = np.mean(list(crisis_performances.values()))

print(f"\n  Performance moyenne du Dual Agent sur 5 crises : {avg_dual_crisis:+.1f}%")
print(f"  Coût annuel du put : {PUT_ANNUAL_COST}%")
print()

print(f"\n  IMPACT DE LA DURÉE (basé sur performance moyenne):")
print(f"  {'Durée':<8} {'Put (coût cumulé)':<18} {'Dual Agent':<12} {'Avantage':<12}")
print("  " + "-" * 55)

for years in [1, 2, 3, 5]:
    put_cost = -PUT_ANNUAL_COST * years
    advantage = avg_dual_crisis - put_cost
    print(f"  {years} an(s)   {put_cost:>+15.1f}%      {avg_dual_crisis:>+9.1f}%      +{advantage:>9.1f} pts")

# ============================================================
# SAUVEGARDE
# ============================================================

import os
os.makedirs("./results", exist_ok=True)

print("\n" + "=" * 80)
print("=" * 80)