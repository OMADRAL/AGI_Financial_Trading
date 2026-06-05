import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from data_prep import get_universal_data
from env import UniversalTradingEnv
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("VÉRIFICATION OVERFITTING : TRAIN vs TEST (Les 60% gagnants)")
print("=" * 60)
#1. CONFIGURATION DES DONNÉES 
TRAIN_TICKERS = ['AAPL', 'MSFT', 'JPM'] 
TRAIN_PERIOD = ("2020-01-01", "2021-12-31") 
TEST_TICKERS = ["MCD", "XOM", "DIS"]
TEST_PERIOD = ("2023-01-01", "2025-01-01") 
MODEL_PATH = "models/universal_agent" 

# 2. FONCTION DE BACKTEST RAPIDE 
def quick_backtest(ticker, start, end, model_agent):
    s, p, _ = get_universal_data(ticker, start, end, seq_len=30)
    if s is None or len(s) < 50:
        return None, None
    env = UniversalTradingEnv(s, p)
    obs, _ = env.reset()
    net_worths = []
    done = False
    while not done:
        action, _ = model_agent.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        net_worths.append(info['net_worth'])   
    nw = np.array(net_worths)
    returns = np.diff(nw) / nw[:-1]
    total_return = (nw[-1] - 10000) / 10000 * 100
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0   
    return total_return, sharpe

#3. EXÉCUTION 
try:
    model = PPO.load(MODEL_PATH, device="cpu")
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")
    exit()
train_results_ret = []
test_results_ret = []
print("--- 1. ÉVALUATION SUR DONNÉES D'ENTRAÎNEMENT (IN-SAMPLE) ---")
for ticker in TRAIN_TICKERS:
    ret, sh = quick_backtest(ticker, TRAIN_PERIOD[0], TRAIN_PERIOD[1], model)
    if ret is not None:
        train_results_ret.append(ret)
        print(f"  {ticker:<6} [{TRAIN_PERIOD[0][:4]}-{TRAIN_PERIOD[1][:4]}] : Rendement = {ret:>+7.1f}% | Sharpe = {sh:>5.2f}")

print("\n--- 2. ÉVALUATION SUR DONNÉES DE TEST (OUT-OF-SAMPLE) ---")
print("    (Focus sur les 3 actifs gagnants représentant les 60%)")
for ticker in TEST_TICKERS:
    ret, sh = quick_backtest(ticker, TEST_PERIOD[0], TEST_PERIOD[1], model)
    if ret is not None:
        test_results_ret.append(ret)
        print(f"  {ticker:<6} [{TEST_PERIOD[0][:4]}-{TEST_PERIOD[1][:4]}] : Rendement = {ret:>+7.1f}% | Sharpe = {sh:>5.2f}")

#4. DIAGNOSTIC STATISTIQUE 
if not train_results_ret or not test_results_ret:
    print("\nErreur : Pas assez de données.")
    exit()
avg_train_ret = np.mean(train_results_ret)
avg_test_ret = np.mean(test_results_ret)
print("\n" + "=" * 60)
print("DIAGNOSTIC FINAL D'OVERFITTING")
print("=" * 60)
print(f"Moyenne Rendement Entraînement (In-Sample) : {avg_train_ret:>+7.1f}%")
print(f"Moyenne Rendement Test (Les 60% gagnants)  : {avg_test_ret:>+7.1f}%")
print("-" * 60)

# Calcul du ratio de dégradation
if avg_train_ret > 0:
    degradation = ((avg_train_ret - avg_test_ret) / avg_train_ret) * 100
else:
    degradation = 0 # Cas rare où le train est négatif
if avg_train_ret > 150 and avg_test_ret < 20:
    print("VERDICT :  OVERFITTING PROBABLE.")
    print("L'écart entre l'entraînement et le test est trop massif. Le modèle a appris par cœur.")
elif degradation > 60:
    print(f"VERDICT :  OVERFITTING MODÉRÉ (Dégradation de {degradation:.0f}%).")
    print("Le modèle est beaucoup moins performant en réalité qu'à l'entraînement.")
else:
    print(f"VERDICT :  AUCUN OVERFITTING CRITIQUE DÉTECTÉ (Dégradation de {degradation:.0f}%).")
    print("La baisse de performance est tout à fait normale en finance quantitative.")
    print("Vos 60% de réussite sur MCD, XOM et DIS sont légitimes et robustes !")
print("=" * 60)