import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from data_prep import get_universal_data
from env import UniversalTradingEnv

# CONFIGURATION
MODEL_PATH = "models/universal_agent"
# Uniquement les actifs UNSEEN (jamais vus pendant l'entraînement)
UNSEEN_TICKERS = ["HOOD", "PLTR", "MCD", "XOM", "DIS"]

def analyze_robustness(ticker):
    print(f"\n" + "="*50)
    print(f"DIAGNOSTIC ROBUSTESSE (UNSEEN) : {ticker}")
    print("="*50)
    
    # 1. Charger les données (Période de test pure : 2023-2025)
    s, p, df_ohlc = get_universal_data(ticker, "2023-01-01", "2025-01-01", seq_len=30)
    if s is None: 
        print(f"Erreur : Impossible de charger {ticker}")
        return None
    
    model = PPO.load(MODEL_PATH)
    env = UniversalTradingEnv(s, p)
    obs, _ = env.reset()
    
    history = []
    done = False
    
    # 2. Simulation pas à pas
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        step = env.current_step - 1
        mkt_ret = (p[step+1] - p[step]) / p[step]
        
        # Volatilité locale sur 10 jours pour définir le risque
        volatility = df_ohlc['Close'].pct_change().rolling(10).std().iloc[step+30]
        
        history.append({
            "mkt_ret": mkt_ret,
            "ia_pos": info['position'],
            "ia_ret": info['position'] * mkt_ret,
            "volatility": volatility
        })

    df = pd.DataFrame(history).dropna()

    # 3. Définition des seuils de régime
    # Bull : Marché monte significativement (> 0.2%)
    # Bear : Marché descend significativement (< -0.2%)
    # High Risk : Volatilité dans le top 25% historique de l'actif
    vol_threshold = df['volatility'].quantile(0.75)
    
    df['regime'] = 'Neutral'
    df.loc[df['mkt_ret'] > 0.002, 'regime'] = 'Bull'
    df.loc[df['mkt_ret'] < -0.002, 'regime'] = 'Bear'
    df['is_high_risk'] = df['volatility'] > vol_threshold

    # 4. Synthèse des résultats
    report = []
    for name, group in df.groupby('regime'):
        capture = (group['ia_ret'].sum() / group['mkt_ret'].sum()) * 100 if group['mkt_ret'].sum() != 0 else 0
        report.append({
            "Régime": name,
            "Exposition IA (%)": group['ia_pos'].mean() * 100,
            "Capture Mouvement (%)": capture,
            "Win Rate J. (%)": (group['ia_ret'] > 0).sum() / len(group) * 100
        })
    
    # Analyse de la réaction face au risque (Haute Volatilité)
    risk_group = df[df['is_high_risk']]
    report.append({
        "Régime": "RISQUE ÉLEVÉ (VOL)",
        "Exposition IA (%)": risk_group['ia_pos'].mean() * 100,
        "Capture Mouvement (%)": (risk_group['ia_ret'].sum() / risk_group['mkt_ret'].sum()) * 100,
        "Win Rate J. (%)": (risk_group['ia_ret'] > 0).sum() / len(risk_group) * 100
    })

    summary_df = pd.DataFrame(report)
    print(summary_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    return summary_df

# Exécution
for t in UNSEEN_TICKERS:
    analyze_robustness(t)