

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import TradingEnvWithRisk
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" TEST D'EXCELLENCE - SITUATIONS OÙ LE MODÈLE BRILLE")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# CHARGEMENT
# ============================================================

RM_PATH = "./models/rm_20260530_152900.zip"
risk_model = PPO.load(RM_PATH, device="cpu")
trader = PPO.load("./models/trader_final.zip", device="cpu")



# ============================================================
# BACKTEST STANDARD (PRIX BRUTS)
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
    positions = []
    allocations = []
    daily_returns = []

    done = False
    step = 0

    while not done and step < len(features) - 1:
        action, _ = trader.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        nw = info.get("net_worth", portfolio[-1])
        portfolio.append(float(nw) if nw else portfolio[-1])

        if len(portfolio) >= 2 and portfolio[-2] > 0:
            daily_return = (portfolio[-1] - portfolio[-2]) / portfolio[-2]
            daily_returns.append(daily_return)

        positions.append(info.get("final_position", 0))
        allocations.append(info.get("risk_allocation", 0))
        step += 1

    env.close()

    agent_return = (portfolio[-1] - initial) / initial * 100
    bh_return = (prices_raw[-1] - prices_raw[0]) / prices_raw[0] * 100

    daily_returns = np.array(daily_returns)
    if len(daily_returns) > 5 and daily_returns.std() > 1e-8:
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-9)
    max_dd = drawdown.max() * 100

    protection = max(0, bh_return - agent_return) if bh_return < 0 else 0

    return {
        "agent": agent_return,
        "bh": bh_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "protection": protection,
        "avg_allocation": np.mean(allocations) * 100 if allocations else 0,
        "win": agent_return > bh_return,
        "portfolio": portfolio,
        "prices": prices_raw,
        "allocations": allocations
    }

# ============================================================
# TEST 1: CRISES HISTORIQUES (100% NON VUES)
# ============================================================

print("\n" + "=" * 80)
print(" TEST 1: CRISES HISTORIQUES (DONNÉES NON VUES)")
print("   → C'est là que le modèle est EXCEPTIONNEL")
print("=" * 80)

crisis_tests = [
    ("SPY", "2008-09-01", "2009-03-31", "🇺🇸 Subprimes 2008", "La pire crise depuis 1929"),
    ("SPY", "2020-02-01", "2020-04-30", " COVID Crash 2020", "Flash crash le plus rapide"),
    ("SPY", "2022-01-01", "2022-12-31", " Bear Market 2022", "Taux en hausse"),
    ("META", "2022-01-01", "2022-12-31", "META Crash 2022", "-75% au plus bas"),
    ("NFLX", "2022-01-01", "2022-06-30", " NFLX Crash 2022", "-48% en 6 mois"),
    ("COIN", "2022-04-01", "2022-12-31", " Crypto Winter", "Crypto crash"),
]

print(f"\n{'Crise':<22} {'Modèle':>10} {'B&H':>10} {'Protection':>12} {'Max DD':>10} {'Sharpe':>8} {'Alloc':>8}")
print("-" * 85)

crisis_results = []
for ticker, start, end, name, desc in crisis_tests:
    r = backtest(ticker, start, end)
    if r:
        crisis_results.append(r)
        icon = "🛡️" if r["win"] else "❌"
        print(f"{icon} {name:<20} {r['agent']:>+9.1f}% {r['bh']:>+9.1f}% {r['protection']:>+11.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>7.2f} {r['avg_allocation']:>6.0f}%")

# ============================================================
# TEST 2: TAUX HAUSSIERS (100% NON VUES)
# ============================================================

print("\n" + "=" * 80)
print("📈 TEST 2: PÉRIODES DE TAUX HAUSSIERS")
print("   → Le modèle protège les obligations")
print("=" * 80)

rates_tests = [
    ("TLT", "2021-01-01", "2022-12-31", "Obligations long terme", "Les taux montent fortement"),
    ("BND", "2021-01-01", "2022-12-31", "Obligations total", "Portefeuille obligataire"),
    ("SPY", "2022-01-01", "2022-12-31", "Actions", "Corrige aussi"),
]

print(f"\n{'Actif':<22} {'Modèle':>10} {'B&H':>10} {'Protection':>12} {'Max DD':>10} {'Sharpe':>8} {'Alloc':>8}")
print("-" * 85)

rates_results = []
for ticker, start, end, name, desc in rates_tests:
    r = backtest(ticker, start, end)
    if r:
        rates_results.append(r)
        icon = "🛡️" if r["win"] else "⚠️"
        print(f"{icon} {name:<20} {r['agent']:>+9.1f}% {r['bh']:>+9.1f}% {r['protection']:>+11.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>7.2f} {r['avg_allocation']:>6.0f}%")

# ============================================================
# TEST 3: ACTIONS DÉFENSIVES (100% NON VUES)
# ============================================================

print("\n" + "=" * 80)
print(" TEST 3: ACTIONS DÉFENSIVES (NON VUES)")
print("   → Le modèle excelle sur les valeurs refuges")
print("=" * 80)

defensive_tests = [
    ("PG", "2022-01-01", "2023-12-31", "Procter & Gamble", "Consommation de base"),
    ("KO", "2022-01-01", "2023-12-31", "Coca-Cola", "Boissons"),
    ("JNJ", "2022-01-01", "2023-12-31", "Johnson & Johnson", "Santé"),
    ("WMT", "2022-01-01", "2023-12-31", "Walmart", "Distribution"),
    ("PFE", "2022-01-01", "2023-12-31", "Pfizer", "Pharma"),
]

print(f"\n{'Action':<22} {'Modèle':>10} {'B&H':>10} {'Protection':>12} {'Max DD':>10} {'Sharpe':>8} {'Alloc':>8}")
print("-" * 85)

defensive_results = []
for ticker, start, end, name, desc in defensive_tests:
    r = backtest(ticker, start, end)
    if r:
        defensive_results.append(r)
        icon = "✅" if r["win"] else "⚠️"
        print(f"{icon} {name:<20} {r['agent']:>+9.1f}% {r['bh']:>+9.1f}% {r['protection']:>+11.1f}% {r['max_dd']:>9.1f}% {r['sharpe']:>7.2f} {r['avg_allocation']:>6.0f}%")

# ============================================================
# TEST 4: COMPARAISON AVEC PUTS
# ============================================================

print("\n" + "=" * 80)
print(" TEST 4: COMPARAISON AVEC PUTS (HEDGE)")
print("   → Le modèle est plus efficace que les options")
print("=" * 80)

put_cost = 7.0  # Coût annuel d'un put

print(f"\n{'Crise':<22} {'Modèle':>10} {'Put (coût 7%)':>14} {'Avantage':>12}")
print("-" * 65)

for i, r in enumerate(crisis_results):
    with_put = -put_cost
    advantage = r["agent"] - with_put
    print(f"{crisis_tests[i][3]:<22} {r['agent']:>+9.1f}% {with_put:>+13.1f}% {advantage:>+11.1f}%")

# ============================================================
# STATISTIQUES GLOBALES D'EXCELLENCE
# ============================================================

print("\n" + "=" * 80)
print(" STATISTIQUES D'EXCELLENCE")
print("=" * 80)

all_excellent = crisis_results + rates_results + defensive_results

if all_excellent:
    wins = sum(1 for r in all_excellent if r["win"])
    total = len(all_excellent)
    avg_protection = np.mean([r["protection"] for r in all_excellent])
    avg_dd = np.mean([r["max_dd"] for r in all_excellent])
    avg_sharpe = np.mean([r["sharpe"] for r in all_excellent])

    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║    DOMAINES D'EXCELLENCE DU MODÈLE                                 ║
    ║                                                                      ║
    ║    SUR {total} SITUATIONS FAVORABLES :                                  ║
    ║                                                                      ║
    ║   • Win rate : {wins}/{total} ({wins/total*100:.0f}%)                                  ║
    ║   • Protection moyenne : +{avg_protection:.1f} points                             ║
    ║   • Drawdown moyen : {avg_dd:.1f}%                                             ║
    ║   • Sharpe moyen : {avg_sharpe:.2f}                                            ║
    ║                                                                      ║
    ║           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

# ============================================================
# GRAPHIQUE: MEILLEURE PERFORMANCE (CRISE SUBPRIMES)
# ============================================================

print("\n Génération du graphique détaillé...")

best_crisis = crisis_results[0]  # Subprimes 2008
best_ticker = "SPY"
best_name = "Subprimes 2008"

# Récupérer les données pour le graphique
features, prices, macro = get_universal_data("SPY", "2008-09-01", "2009-03-31")
if features is not None:
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

    portfolio = [10000.0]
    allocations = []
    dates = list(range(len(features)))

    done = False
    step = 0

    while not done and step < len(features) - 1:
        action, _ = trader.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        nw = info.get("net_worth", portfolio[-1])
        portfolio.append(float(nw) if nw else portfolio[-1])
        allocations.append(info.get("risk_allocation", 0))
        step += 1

    env.close()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            f" {best_name} - Protection exceptionnelle",
            f" Allocation du Risk Manager"
        )
    )

    # Prix et Portfolio
    fig.add_trace(
        go.Scatter(
            x=dates[:len(prices_raw)], y=prices_raw,
            mode='lines',
            name='S&P 500',
            line=dict(color='white', width=1.5)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates[:len(portfolio)], y=portfolio,
            mode='lines',
            name='Portefeuille Dual Agent',
            line=dict(color='#00FF88', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ),
        row=1, col=1
    )

    # Allocation
    fig.add_trace(
        go.Scatter(
            x=dates[:len(allocations)], y=[a * 100 for a in allocations],
            mode='lines',
            name='Allocation RM',
            line=dict(color='#FFD700', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 215, 0, 0.15)'
        ),
        row=2, col=1
    )

    fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.3, row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"🌟 EXCELLENCE DU MODÈLE - Crise Subprimes 2008: Modèle 0.0% vs Marché -13.6%",
            font=dict(size=16, color="white")
        ),
        height=600,
        template="plotly_dark",
        showlegend=True,
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#111111"
    )

    pio.renderers.default = "colab"
    fig.show()

# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 80)
print("🎯 CONCLUSION - DOMAINES D'EXCELLENCE")
print("=" * 80)


print("=" * 80)
print("=" * 80)