
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import TradingEnvWithRisk
import warnings
import os
warnings.filterwarnings('ignore')

print("=" * 80)
print("DUAL AGENT vs PUT OPTION - BULL MARKET COMPARISON (EXTENDED)")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# LOAD MODELS
# ============================================================

RM_PATH = "./models/rm_20260530_152900.zip"
risk_model = PPO.load(RM_PATH, device="cpu")
trader = PPO.load("./models/trader_final.zip", device="cpu")

print("\n✅ Models loaded")

# ============================================================
# BACKTEST FUNCTION
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
# CONFIGURATION
# ============================================================

PUT_COST = 7.0

# Extended list of bull market scenarios
bull_markets = [
    # US Indices
    ("SPY", "2023-01-01", "2024-12-31", "Bull Market 2023-24 (S&P 500)"),
    ("QQQ", "2023-01-01", "2024-12-31", "Tech Bull 2023-24 (Nasdaq)"),
    ("DIA", "2023-01-01", "2024-12-31", "Industrial Bull 2023-24 (Dow Jones)"),
    ("IWM", "2023-01-01", "2024-12-31", "Small Cap Bull 2023-24 (Russell 2000)"),
    
    # Tech Giants
    ("NVDA", "2023-01-01", "2024-12-31", "AI Bull Run (NVIDIA)"),
    ("META", "2023-01-01", "2024-12-31", "Social Media Bull (Meta)"),
    ("MSFT", "2023-01-01", "2024-12-31", "Tech Bull (Microsoft)"),
    ("GOOGL", "2023-01-01", "2024-12-31", "Tech Bull (Google)"),
    ("AMZN", "2023-01-01", "2024-12-31", "E-commerce Bull (Amazon)"),
    ("AAPL", "2023-01-01", "2024-12-31", "Tech Bull (Apple)"),
    ("TSLA", "2023-01-01", "2024-12-31", "EV Bull (Tesla)"),
    
    # Other Growth Stocks
    ("SHOP", "2023-01-01", "2024-12-31", "E-commerce Bull (Shopify)"),
    ("UBER", "2023-01-01", "2024-12-31", "Ride-sharing Bull (Uber)"),
    ("SNOW", "2023-01-01", "2024-12-31", "Cloud Bull (Snowflake)"),
    ("CRM", "2023-01-01", "2024-12-31", "SaaS Bull (Salesforce)"),
    ("ADBE", "2023-01-01", "2024-12-31", "Software Bull (Adobe)"),
    ("NFLX", "2023-01-01", "2024-12-31", "Streaming Bull (Netflix)"),
]

# ============================================================
# RUN TESTS
# ============================================================

print("\n" + "=" * 80)
print(f"RESULTS: DUAL AGENT vs PUT OPTION ({PUT_COST}% annual cost)")
print(f"Testing {len(bull_markets)} bull market scenarios")
print("=" * 80)

print(f"\n{'Market':<35} {'Dual Agent':>12} {'Put (7%)':>14} {'Advantage':>12} {'Verdict':>10}")
print("-" * 95)

results = []
for ticker, start, end, name in bull_markets:
    r = backtest(ticker, start, end)
    if r:
        put_return = -PUT_COST
        advantage = r["agent"] - put_return
        verdict = "✅ Dual Agent" if advantage > 0 else "❌ Put"
        
        print(f"{name:<35} {r['agent']:>+11.1f}% {put_return:>+13.1f}% +{advantage:>10.1f} pts {verdict:>10}")
        
        results.append({
            "ticker": ticker,
            "period": name,
            "dual_agent": r["agent"],
            "put": put_return,
            "advantage": advantage,
            "market_return": r["bh"]
        })

# ============================================================
# STATISTICS BY CATEGORY
# ============================================================

print("\n" + "=" * 80)
print("STATISTICS BY CATEGORY")
print("=" * 80)

# US Indices
indices = [r for r in results if "S&P" in r["period"] or "Nasdaq" in r["period"] or "Dow" in r["period"] or "Russell" in r["period"]]
if indices:
    avg_adv = np.mean([r["advantage"] for r in indices])
    print(f"\n US INDICES ({len(indices)} scenarios):")
    print(f"   Average advantage: +{avg_adv:.1f} points")

# Tech Giants
tech = [r for r in results if any(x in r["period"] for x in ["NVIDIA", "Meta", "Microsoft", "Google", "Amazon", "Apple", "Tesla"])]
if tech:
    avg_adv = np.mean([r["advantage"] for r in tech])
    print(f"\n TECH GIANTS ({len(tech)} scenarios):")
    print(f"   Average advantage: +{avg_adv:.1f} points")

# Growth Stocks
growth = [r for r in results if any(x in r["period"] for x in ["Shopify", "Uber", "Snowflake", "Salesforce", "Adobe", "Netflix"])]
if growth:
    avg_adv = np.mean([r["advantage"] for r in growth])
    print(f"\n GROWTH STOCKS ({len(growth)} scenarios):")
    print(f"   Average advantage: +{avg_adv:.1f} points")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

if results:
    wins = sum(1 for r in results if r["advantage"] > 0)
    total = len(results)
    avg_advantage = np.mean([r["advantage"] for r in results])
    avg_dual = np.mean([r["dual_agent"] for r in results])
    avg_market = np.mean([r["market_return"] for r in results])
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\n OVERALL PERFORMANCE ({total} scenarios):")
    print(f"   • Win rate vs Put: {wins}/{total} ({wins/total*100:.0f}%)")
    print(f"   • Average Dual Agent return: {avg_dual:+.1f}%")
    print(f"   • Average Market return: {avg_market:+.1f}%")
    print(f"   • Average advantage over Put: +{avg_advantage:.1f} points")
    
    if wins/total >= 0.8:
        print(f"\n CONCLUSION: Dual Agent consistently outperforms put strategy")
        print(f"   The model captures upside potential while put loses its premium")
    elif wins/total >= 0.6:
        print(f"\n CONCLUSION: Dual Agent generally outperforms put strategy")
        print(f"   Performance varies but remains positive on average")
    else:
        print(f"\n CONCLUSION: Dual Agent struggles in some bull scenarios")
        print(f"   Model may be too defensive for certain growth stocks")

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("./results", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("./results/dual_vs_put_bull_extended.csv", index=False)
print(f"\n Results saved: ./results/dual_vs_put_bull_extended.csv")
print("=" * 80)