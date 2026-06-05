import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from data_prep import get_universal_data
from env import UniversalTradingEnv

SAVE_PATH = "plots/"
os.makedirs(SAVE_PATH, exist_ok=True)

model = PPO.load("models/universal_agent")
TRAINING_TICKERS = ['AAPL','MSFT','TSLA','NVDA','JPM','AMZN','META','SPY','QQQ','GLD',
                    'KO','PG','JNJ','WMT','V','NFLX','SNAP','UBER','BABA','TSM']
TEST_TICKERS = ["TSLA", "HOOD", "PLTR", "MCD", "XOM", "DIS"]
results_summary = []
for TICKER in TEST_TICKERS:
    print(f"\n{'='*50}\nTesting: {TICKER} ({'SEEN' if TICKER in TRAINING_TICKERS else 'UNSEEN'})")
    s, p, df_ohlc = get_universal_data(TICKER, "2023-01-01", "2025-01-01", seq_len=30)
    if s is None: continue
    env = UniversalTradingEnv(s, p)
    obs, _ = env.reset()

    net_worths, positions, daily_returns, drawdowns = [], [], [], []
    peak = 10000.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)

        nw = info['net_worth']
        if nw > peak: peak = nw
        drawdown = (nw - peak) / peak * 100

        net_worths.append(nw)
        # Append the true discretized position for the chart
        positions.append(info['position'])
        drawdowns.append(drawdown)
        if len(net_worths) > 1:
            ret = (net_worths[-1] - net_worths[-2]) / (net_worths[-2] + 1e-9) * 100
            daily_returns.append(ret)
        else:
            daily_returns.append(0.0)

        if done: break

    dates = df_ohlc.index[1:len(net_worths)+1]
    bh_return = (p[1:len(net_worths)+1] / p[0]) * 10000
    final_ia, final_bh = net_worths[-1], bh_return[-1]
    beat = final_ia > final_bh

    total_return = (final_ia - 10000) / 10000 * 100
    bh_total = (final_bh - 10000) / 10000 * 100
    max_drawdown = min(drawdowns)
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)
    win_rate = sum(1 for r in daily_returns if r > 0) / max(1, len(daily_returns)) * 100
    
    long_days = sum(1 for p_ in positions if p_ > 0.5)
    short_days = sum(1 for p_ in positions if p_ < -0.5)
    flat_days = len(positions) - long_days - short_days

    results_summary.append({
        'ticker': TICKER, 'final_ia': final_ia, 'final_bh': final_bh, 'beat': beat,
        'seen': TICKER in TRAINING_TICKERS, 'sharpe': sharpe, 'max_dd': max_drawdown, 'win_rate': win_rate
    })

    print(f"IA: ${final_ia:,.2f} | B&H: ${final_bh:,.2f} | Sharpe: {sharpe:.2f}")

    # Plotting
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.40, 0.20, 0.20, 0.20])

    fig.add_trace(go.Scatter(x=dates, y=net_worths, name="IA Net Worth", line=dict(color='#00FF00', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bh_return, name="Buy & Hold", line=dict(color='gray', dash='dash')), row=1, col=1)

    pos_colors = ['#00CC00' if v > 0.5 else '#CC0000' if v < -0.5 else '#888888' for v in positions]
    fig.add_trace(go.Bar(x=dates, y=positions, name="Position", marker_color=pos_colors), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=dates, y=drawdowns, name="Drawdown", fill='tozeroy', line=dict(color='#FF4444')), row=3, col=1)
    
    ret_colors = ['#00CC00' if r > 0 else '#CC0000' for r in daily_returns]
    fig.add_trace(go.Bar(x=dates, y=daily_returns, name="Daily Return %", marker_color=ret_colors), row=4, col=1)

    fig.update_layout(title=f"AGI Model: {TICKER} | IA: ${final_ia:,.2f} vs B&H: ${final_bh:,.2f}", 
                      template="plotly_dark", height=1000)
    fig.write_html(f"{SAVE_PATH}backtest_{TICKER}.html")

# Summary
print(f"\n{'='*75}\n{'AGI MODEL — GENERALIZATION RESULTS':^75}\n{'='*75}")
for r in results_summary:
    print(f"{r['ticker']:<8} {'SEEN' if r['seen'] else 'UNSEEN':<12} ${r['final_ia']:>9,.0f} ${r['final_bh']:>9,.0f} {r['sharpe']:>8.2f} {r['max_dd']:>7.1f}% {r['win_rate']:>8.1f}% {'Beat' if r['beat'] else 'Lost':>8}")

unseen = [r for r in results_summary if not r['seen']]
gen_rate = sum(1 for r in unseen if r['beat']) / max(1, len(unseen))
print(f"\nGeneralization rate on UNSEEN stocks: {gen_rate*100:.0f}%")