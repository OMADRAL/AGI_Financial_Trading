# test_bear_market_excellence

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
print("TEST SPÉCIFIQUE - PERFORMANCE SUR MARCHÉS BAISSIERS")
print("   Domaine d'excellence du modèle : PROTECTION EN CRISE")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# CHARGEMENT
# ============================================================

RM_PATH = "./models/rm_20260530_152900.zip"
risk_model = PPO.load(RM_PATH, device="cpu")
trader = PPO.load("./models/trader_final.zip", device="cpu")



# ============================================================
# BACKTEST (PRIX BRUTS)
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

    protection = (bh_return - agent_return) if bh_return < 0 else 0

    return {
        "agent": agent_return,
        "bh": bh_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "protection": protection,
        "avg_allocation": np.mean(allocations) * 100 if allocations else 0,
        "win": agent_return > bh_return
    }

# ============================================================
# SCÉNARIOS DE MARCHÉ BAISSIER (100% NON VUS)
# ============================================================

bear_scenarios = [
    # Crises historiques majeures
    ("SPY", "2000-03-01", "2002-10-31", "Dot-com Bubble 2000-2002"),
    ("SPY", "2007-10-01", "2009-03-31", "Financial Crisis 2007-2009"),
    ("SPY", "2020-02-01", "2020-04-30", "COVID Crash 2020"),
    ("SPY", "2022-01-01", "2022-12-31", "Bear Market 2022"),

    # Krachs sectoriels
    ("META", "2022-01-01", "2022-12-31", "META Crash 2022"),
    ("NFLX", "2022-01-01", "2022-06-30", "NFLX Crash 2022"),
    ("COIN", "2022-04-01", "2022-12-31", "Crypto Winter 2022"),
    ("TSLA", "2022-01-01", "2022-12-31", "TSLA Bear 2022"),

    # Marchés baissiers internationaux
    ("EWZ", "2010-01-01", "2016-12-31", "Brazil Lost Decade"),
    ("FXI", "2021-01-01", "2024-06-30", "China Bear Market"),
    ("EWJ", "1990-01-01", "2012-12-31", "Japan Lost Decades"),
]

print("\n" + "=" * 80)
print("TEST SUR MARCHÉS BAISSIERS (Domaine d'excellence)")
print("=" * 80)

print(f"\n{'Scénario':<35} {'Dual%':>10} {'B&H%':>10} {'Protection':>12} {'MaxDD%':>8} {'Sharpe':>7} {'Alloc%':>7}")
print("-" * 95)

results = []
for ticker, start, end, name in bear_scenarios:
    r = backtest(ticker, start, end)
    if r:
        r["name"] = name
        results.append(r)

        symbol = "🛡️" if r["win"] else "⚠️"
        protection_str = f"+{r['protection']:.1f}" if r['protection'] > 0 else f"{r['protection']:.1f}"
        print(f"{symbol} {name:<33} {r['agent']:>+9.1f}% {r['bh']:>+9.1f}% {protection_str:>11} pts {r['max_dd']:>7.1f}% {r['sharpe']:>6.2f} {r['avg_allocation']:>6.0f}%")

# ============================================================
# STATISTIQUES GLOBALES - MARCHÉS BAISSIERS
# ============================================================

print("\n" + "=" * 80)
print("STATISTIQUES - PERFORMANCE EN MARCHÉ BAISSIER")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    wins = sum(1 for r in results if r["win"])
    total = len(results)
    avg_protection = np.mean([r["protection"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])
    avg_agent = np.mean([r["agent"] for r in results])
    avg_bh = np.mean([r["bh"] for r in results])

 

# ============================================================
# TABLEAU RÉCAPITULATIF PAR TYPE DE CRISE
# ============================================================

print("\n" + "=" * 80)
print(" CLASSEMENT PAR PROTECTION OBTENUE")
print("=" * 80)

sorted_results = sorted(results, key=lambda x: x["protection"], reverse=True)
print(f"\n{'Scénario':<35} {'Protection':>12} {'Dual%':>10} {'B&H%':>10}")
print("-" * 70)
for r in sorted_results[:5]:
    print(f"  {r['name']:<33} +{r['protection']:>10.1f} pts {r['agent']:>+9.1f}% {r['bh']:>+9.1f}%")

# ============================================================
# GRAPHIQUE DE SYNTHÈSE
# ============================================================

print("\n Génération du graphique de synthèse...")

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Performance comparée sur marchés baissiers",
        "Protection générée (points)"
    ),
    vertical_spacing=0.15
)

names = [r["name"][:25] for r in results]
agent_returns = [r["agent"] for r in results]
bh_returns = [r["bh"] for r in results]
protections = [r["protection"] for r in results]

# Graphique 1: Performances
fig.add_trace(
    go.Bar(name='Dual Agent', x=names, y=agent_returns, marker_color='#00C8FF'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(name='Buy & Hold', x=names, y=bh_returns, marker_color='#FF6B6B'),
    row=1, col=1
)

# Graphique 2: Protection
colors = ['#00FF88' if p > 0 else '#FF4444' for p in protections]
fig.add_trace(
    go.Bar(x=names, y=protections, marker_color=colors),
    row=2, col=1
)

fig.update_layout(
    title=dict(
        text=f" EXCELLENCE DU MODÈLE - Protection en marché baissier",
        font=dict(size=18, color="white")
    ),
    height=800,
    template="plotly_dark",
    showlegend=True,
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#111111",
    barmode='group'
)

pio.renderers.default = "colab"
fig.show()

# ============================================================
# VERDICT FINAL
# ============================================================

print("\n" + "=" * 80)
print("=" * 80)


print("=" * 80)
print("=" * 80)