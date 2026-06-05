
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
print(" TEST ÉLARGI - PÉRIODE DE TAUX HAUSSIERS (2021-2022)")
print("   12 actifs testés pour valider la protection du modèle")
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
    daily_returns = []
    allocations = []

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

    if bh_return < 0:
        protection = agent_return - bh_return
    else:
        protection = 0

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
# ACTIFS TESTÉS (12 actifs)
# ============================================================

print("\n" + "=" * 80)
print("📊 ACTIFS TESTÉS - PÉRIODE DE TAUX HAUSSIERS (2021-2022)")
print("=" * 80)

assets = [
    # OBLIGATIONS (les plus sensibles aux taux)
    ("TLT", "2021-01-01", "2022-12-31", "Obligations LT", "iShares 20+ Year Treasury Bond", "Obligations"),
    ("IEF", "2021-01-01", "2022-12-31", "Obligations MT", "iShares 7-10 Year Treasury Bond", "Obligations"),
    ("SHY", "2021-01-01", "2022-12-31", "Obligations CT", "iShares 1-3 Year Treasury Bond", "Obligations"),
    ("BND", "2021-01-01", "2022-12-31", "Obligations Total", "Vanguard Total Bond Market", "Obligations"),
    ("LQD", "2021-01-01", "2022-12-31", "Corporate Bonds", "iShares Investment Grade Corp Bonds", "Obligations"),
    ("HYG", "2021-01-01", "2022-12-31", "High Yield", "iShares High Yield Corporate Bond", "Obligations"),

    # ACTIONS
    ("SPY", "2021-01-01", "2022-12-31", "S&P 500", "Large Cap US", "Actions"),
    ("QQQ", "2021-01-01", "2022-12-31", "Nasdaq", "Tech Growth", "Actions"),
    ("DIA", "2021-01-01", "2022-12-31", "Dow Jones", "Industrial", "Actions"),
    ("IWM", "2021-01-01", "2022-12-31", "Russell 2000", "Small Caps", "Actions"),

    # SECTEURS SENSIBLES AUX TAUX
    ("XLU", "2021-01-01", "2022-12-31", "Utilities", "Secteur défensif", "Secteur"),
    ("XLRE", "2021-01-01", "2022-12-31", "Real Estate", "Immobilier", "Secteur"),
]

print("\n" + "=" * 100)
print("📈 RÉSULTATS - PÉRIODE DE TAUX HAUSSIERS (2021-2022)")
print("=" * 100)

print(f"\n{'Ticker':<8} {'Nom':<22} {'Type':<14} {'Dual%':>8} {'B&H%':>8} {'Protection':>10} {'Sharpe':>7} {'MaxDD%':>8} {'Alloc%':>7} {'Win':>4}")
print("-" * 110)

results = []
for ticker, start, end, name, desc, asset_type in assets:
    r = backtest(ticker, start, end)
    if r:
        r["ticker"] = ticker
        r["name"] = name
        r["asset_type"] = asset_type
        results.append(r)

        symbol = "✅" if r["win"] else "❌"
        protection_str = f"+{r['protection']:.1f}" if r['protection'] > 0 else f"{r['protection']:.1f}"
        print(f"{ticker:<8} {name:<22} {asset_type:<14} {r['agent']:>+7.1f}% {r['bh']:>+7.1f}% {protection_str:>9} pts {r['sharpe']:>6.2f} {r['max_dd']:>7.1f}% {r['avg_allocation']:>6.0f}% {symbol:>4}")

# ============================================================
# STATISTIQUES PAR CATÉGORIE
# ============================================================

print("\n" + "=" * 80)
print(" STATISTIQUES PAR CATÉGORIE D'ACTIF")
print("=" * 80)

if results:
    df = pd.DataFrame(results)

    print("\n   OBLIGATIONS (6 actifs):")
    bonds = df[df['asset_type'] == 'Obligations']
    if len(bonds) > 0:
        print(f"     Win rate: {bonds['win'].sum()}/{len(bonds)} ({bonds['win'].sum()/len(bonds)*100:.0f}%)")
        print(f"     Perf Dual moyenne: {bonds['agent'].mean():+.1f}%")
        print(f"     Perf B&H moyenne: {bonds['bh'].mean():+.1f}%")
        print(f"     Protection moyenne: +{bonds['protection'].mean():.1f} points")
        print(f"     Drawdown moyen: {bonds['max_dd'].mean():.1f}%")

    print("\n   ACTIONS (4 actifs):")
    stocks = df[df['asset_type'] == 'Actions']
    if len(stocks) > 0:
        print(f"     Win rate: {stocks['win'].sum()}/{len(stocks)} ({stocks['win'].sum()/len(stocks)*100:.0f}%)")
        print(f"     Perf Dual moyenne: {stocks['agent'].mean():+.1f}%")
        print(f"     Perf B&H moyenne: {stocks['bh'].mean():+.1f}%")
        print(f"     Protection moyenne: +{stocks['protection'].mean():.1f} points")
        print(f"     Drawdown moyen: {stocks['max_dd'].mean():.1f}%")

    print("\n   SECTEURS SENSIBLES (2 actifs):")
    sectors = df[df['asset_type'] == 'Secteur']
    if len(sectors) > 0:
        print(f"     Win rate: {sectors['win'].sum()}/{len(sectors)} ({sectors['win'].sum()/len(sectors)*100:.0f}%)")
        print(f"     Perf Dual moyenne: {sectors['agent'].mean():+.1f}%")
        print(f"     Perf B&H moyenne: {sectors['bh'].mean():+.1f}%")
        print(f"     Protection moyenne: +{sectors['protection'].mean():.1f} points")
        print(f"     Drawdown moyen: {sectors['max_dd'].mean():.1f}%")

# ============================================================
# STATISTIQUES GLOBALES
# ============================================================

print("\n" + "=" * 80)
print(" STATISTIQUES GLOBALES - HAUSSE DES TAUX")
print("=" * 80)

if results:
    wins = sum(1 for r in results if r["win"])
    total = len(results)
    avg_protection = np.mean([r["protection"] for r in results])
    avg_dd = np.mean([r["max_dd"] for r in results])
    avg_agent = np.mean([r["agent"] for r in results])
    avg_bh = np.mean([r["bh"] for r in results])

 

# ============================================================
# TOP PROTECTIONS
# ============================================================

print("\n" + "=" * 80)
print(" TOP 5 MEILLEURES PROTECTIONS")
print("=" * 80)

if results:
    sorted_by_protection = sorted(results, key=lambda x: x["protection"], reverse=True)
    print(f"\n{'Actif':<12} {'Nom':<22} {'Protection':>12} {'Dual%':>10} {'B&H%':>10}")
    print("-" * 70)
    for r in sorted_by_protection[:5]:
        print(f"{r['ticker']:<12} {r['name']:<22} +{r['protection']:>10.1f} pts {r['agent']:>+9.1f}% {r['bh']:>+9.1f}%")

# ============================================================
# GRAPHIQUE DE SYNTHÈSE
# ============================================================

print("\n Génération du graphique de synthèse...")

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Performance comparée pendant la hausse des taux (2021-2022)",
        "Protection générée par le Dual Agent (points)"
    ),
    vertical_spacing=0.12
)

tickers = [r["ticker"] for r in results]
agent_returns = [r["agent"] for r in results]
bh_returns = [r["bh"] for r in results]
protections = [r["protection"] for r in results]

# Graphique 1: Performances comparées
fig.add_trace(
    go.Bar(name='Dual Agent', x=tickers, y=agent_returns, marker_color='#00C8FF'),
    row=1, col=1
)
fig.add_trace(
    go.Bar(name='Buy & Hold', x=tickers, y=bh_returns, marker_color='#FF6B6B'),
    row=1, col=1
)

# Graphique 2: Protection
colors = ['#00FF88' if p > 0 else '#FF4444' for p in protections]
fig.add_trace(
    go.Bar(x=tickers, y=protections, marker_color=colors),
    row=2, col=1
)

fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5, row=2, col=1)

fig.update_layout(
    title=dict(
        text=f" Protection du Dual Agent pendant la hausse des taux (2021-2022) - Win rate: {wins}/{total} ({wins/total*100:.0f}%)",
        font=dict(size=16, color="white")
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

print("\n" + "=" * 80)
print(" Test terminé - 12 actifs validés")
print("=" * 80)