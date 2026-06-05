import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from data_prep import get_universal_data
from env import UniversalTradingEnv
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ANALYSE D'OVERFITTING - AGENT UNIQUE")
print("=" * 60)

# --- 1. CONFIGURATION ---
TRAIN_PERIODS = [
    ("2018-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2023-06-30")
]
#  3 actions vues à l'entraînement (In-Sample)
TRAIN_TICKERS = ["AAPL", "MSFT", "JPM"] 

# Période de test hors-échantillon
TEST_PERIOD = ("2023-07-01", "2024-06-30")
#  3 actions non vues à l'entraînement (Out-of-Sample)
TEST_TICKERS = ["MCD", "XOM", "DIS"] 
MODEL_PATH = "models/universal_agent" 
#2. FONCTION DE BACKTEST
def run_backtest(model, ticker, start, end, seq_len=30):
    s, p, _ = get_universal_data(ticker, start, end, seq_len=seq_len)
    if s is None or len(s) < 50:
        return None
        
    env = UniversalTradingEnv(s, p)
    obs, _ = env.reset()
    
    net_worths = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        net_worths.append(info['net_worth'])
        
    nw = np.array(net_worths)
    returns = np.diff(nw) / nw[:-1]
    
    total_return = (nw[-1] - 10000) / 10000 * 100
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0
    
    return total_return, sharpe

#3. EXÉCUTION DE L'ANALYSE 
try:
    model = PPO.load(MODEL_PATH, device="cpu")
    print(f"Modele '{MODEL_PATH}' charge avec succes.\n")
except Exception as e:
    print(f"Erreur lors du chargement du modele : {e}")
    exit()

train_returns = []
train_sharpes = []
test_returns = []
test_sharpes = []

print("--- EVALUATION IN-SAMPLE (Donnees d'entrainement vues) ---")
for ticker in TRAIN_TICKERS:
    for start, end in TRAIN_PERIODS:
        res = run_backtest(model, ticker, start, end)
        if res:
            ret, sh = res
            train_returns.append(ret)
            train_sharpes.append(sh)
            print(f"{ticker} [{start[:4]}-{end[:4]}] : Rendement = {ret:>+7.1f}% | Sharpe = {sh:>5.2f}")

print("\n--- EVALUATION OUT-OF-SAMPLE (Donnees de test NON VUES) ---")
for ticker in TEST_TICKERS:
    res = run_backtest(model, ticker, TEST_PERIOD[0], TEST_PERIOD[1])
    if res:
        ret, sh = res
        test_returns.append(ret)
        test_sharpes.append(sh)
        print(f"{ticker} [{TEST_PERIOD[0][:4]}-{TEST_PERIOD[1][:4]}] : Rendement = {ret:>+7.1f}% | Sharpe = {sh:>5.2f}")

# 4. DIAGNOSTIC D'OVERFITTING 
if not train_returns or not test_returns:
    print("\nErreur : Pas assez de donnees pour effectuer le diagnostic.")
    exit()

mean_train_ret = np.mean(train_returns)
mean_train_sh = np.mean(train_sharpes)
mean_test_ret = np.mean(test_returns)
mean_test_sh = np.mean(test_sharpes)

# La perte de performance (Degradation Ratio)
degradation_ret = ((mean_train_ret - mean_test_ret) / abs(mean_train_ret)) * 100 if mean_train_ret != 0 else 0
degradation_sh = ((mean_train_sh - mean_test_sh) / abs(mean_train_sh)) * 100 if mean_train_sh != 0 else 0

print("\n" + "=" * 60)
print("DIAGNOSTIC D'OVERFITTING (SURAPPRENTISSAGE)")
print("=" * 60)
print(f"Moyenne In-Sample (Entrainement)   : Rendement = {mean_train_ret:>+7.1f}% | Sharpe = {mean_train_sh:>5.2f}")
print(f"Moyenne Out-of-Sample (Test UNSEEN): Rendement = {mean_test_ret:>+7.1f}% | Sharpe = {mean_test_sh:>5.2f}")
print("-" * 60)
print(f"Degradation du rendement : {degradation_ret:.1f}%")
print(f"Degradation du Sharpe    : {degradation_sh:.1f}%")
print("-" * 60)

# Interprétation automatique
if mean_train_ret > 150 and mean_test_ret < 0:
    print("VERDICT : OVERFITTING MASSIF DETECTE (Le modele a appris par coeur).")
elif degradation_sh > 50:
    print("VERDICT : OVERFITTING LEGER A MODERE (Le modele peine a generaliser le meme niveau de risque).")
elif degradation_sh < 30 and mean_test_ret > 0:
    print("VERDICT : AUCUN OVERFITTING DETECTE. Le modele generalise parfaitement (60% de reussite legitime).")
else:
    print("VERDICT : COMPORTEMENT NORMAL EN FINANCE. Baisse de performance attendue hors-echantillon, mais modele robuste.")
print("=" * 60)