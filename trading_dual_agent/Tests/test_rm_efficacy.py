# ============================================================
# test_rm_efficacy.py
# Vérifie que le Risk Manager fonctionne correctement
# avant de lancer train.py
# ============================================================

import numpy as np
import glob, os
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import RiskManagerEnv

print("=" * 65)
print("🧪 TEST D'EFFICACITÉ — RISK MANAGER")
print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

# ============================================================
# 1. CHARGER LE MODÈLE
# ============================================================

rm_files = (glob.glob("./models/rm_*.zip") +
            glob.glob("./best_model/rm/*.zip") +
            ["./models/rm_20260530_152900.zip"])
rm_files  = [f for f in rm_files if os.path.exists(f)]

if not rm_files:
    print("❌ Aucun modèle RM trouvé.")
    print("   Lance d'abord : python train_rm.py")
    exit()

latest = max(rm_files, key=os.path.getmtime)
rm     = PPO.load(latest)
print(f"✅ Modèle chargé : {latest}\n")


# ============================================================
# UTILITAIRE
# ============================================================

def get_mean_alloc(model, vix_val, spy_val=0.001, n=120):
    """Allocation moyenne sur une séquence de marché simulée."""
    macro = np.zeros((200, 3), dtype=np.float32)
    macro[:, 0] = vix_val
    macro[:, 1] = spy_val
    env    = RiskManagerEnv(macro)
    obs, _ = env.reset(seed=0)
    allocs = []
    for _ in range(n):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
        allocs.append(info["allocation"])
        if done:
            break
    return np.mean(allocs), np.std(allocs)


# ============================================================
# TEST 1 — ALLOCATION PAR ZONE DE MARCHÉ
# ============================================================

print("─" * 65)
print("TEST 1 — Allocation par zone de marché")
print("─" * 65)

zones = [
    ("🟢 Très calme",  0.35,  0.003,  0.65, 1.00),
    ("🟢 Calme",       0.55,  0.002,  0.50, 0.85),
    ("🟡 Normal",      0.80,  0.001,  0.30, 0.65),
    ("🟠 Agité",       1.10, -0.002,  0.15, 0.45),
    ("🔴 Crise",       1.50, -0.015,  0.05, 0.25),
    ("🔴 Crash",       2.00, -0.040,  0.00, 0.15),
]

passed_t1 = 0
print(f"\n  {'Zone':<18} {'VIX':>5}  {'Alloc':>7}  {'Std':>5}  "
      f"{'Attendu':>12}  Résultat")
print("  " + "-" * 60)

for label, vix, spy, lo, hi in zones:
    mean, std = get_mean_alloc(rm, vix, spy)
    ok        = lo <= mean <= hi
    if ok:
        passed_t1 += 1
    print(f"  {label:<18} {vix:>5.2f}  {mean:>6.1%}  {std:>4.1%}  "
          f"[{lo:.0%}–{hi:.0%}]  {'✅' if ok else '❌'}")

score_t1 = passed_t1 / len(zones)
print(f"\n  Score : {passed_t1}/{len(zones)}  "
      f"({'✅ OK' if score_t1 >= 0.67 else '❌ Insuffisant'})")


# ============================================================
# TEST 2 — CORRÉLATION VIX / ALLOCATION
# ============================================================

print("\n" + "─" * 65)
print("TEST 2 — Corrélation VIX → Allocation (doit être négative)")
print("─" * 65)

vix_sweep   = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5]
alloc_sweep = [get_mean_alloc(rm, v)[0] for v in vix_sweep]

print(f"\n  {'VIX norm':>9}  {'VIX réel':>9}  {'Allocation':>10}")
print("  " + "-" * 35)
for v, a in zip(vix_sweep, alloc_sweep):
    print(f"  {v:>9.2f}  {v*20:>9.1f}  {a:>10.1%}")

corr     = np.corrcoef(vix_sweep, alloc_sweep)[0, 1]
corr_ok  = corr < -0.70
print(f"\n  Corrélation : {corr:.4f}  "
      f"({'✅ OK (< -0.70)' if corr_ok else '❌ Insuffisante'})")


# ============================================================
# TEST 3 — RÉACTIVITÉ AUX CHOCS
# ============================================================

print("\n" + "─" * 65)
print("TEST 3 — Réactivité aux chocs de marché")
print("─" * 65)

n      = 350
macro  = np.zeros((n, 3), dtype=np.float32)
# Phase calme
macro[:100, 0] = 0.5;  macro[:100, 1]  =  0.003
# Phase crash
macro[100:160, 0] = 1.9; macro[100:160, 1] = -0.04
# Phase rebond
macro[160:, 0] = 0.6;  macro[160:, 1]  =  0.002

env    = RiskManagerEnv(macro)
obs, _ = env.reset(seed=0)
allocs = []
phases = []

for i in range(min(320, n - 10)):
    act, _ = rm.predict(obs, deterministic=True)
    obs, _, done, _, info = env.step(act)
    allocs.append(info["allocation"])
    phases.append(0 if i < 100 else 1 if i < 160 else 2)
    if done:
        break

calme  = np.mean([a for a, p in zip(allocs, phases) if p == 0])
crash  = np.mean([a for a, p in zip(allocs, phases) if p == 1])
rebond = np.mean([a for a, p in zip(allocs, phases) if p == 2])

react_ok = calme > crash and rebond > crash

print(f"\n  Phase calme  (VIX=10) : {calme:.1%}")
print(f"  Phase crash  (VIX=38) : {crash:.1%}")
print(f"  Phase rebond (VIX=12) : {rebond:.1%}")
print(f"\n  Réduction au crash   : {(calme-crash)*100:+.1f} pts")
print(f"  Récupération rebond  : {(rebond-crash)*100:+.1f} pts")
print(f"\n  {'✅ OK' if react_ok else '❌ Pas assez réactif'}")


# ============================================================
# TEST 4 — DONNÉES RÉELLES (Bear 2022 vs Bull 2023)
# ============================================================

print("\n" + "─" * 65)
print("TEST 4 — Données réelles : Bear 2022 vs Bull 2023")
print("─" * 65)

real_ok     = False
bear_alloc  = None
bull_alloc  = None

for label, start, end in [
    ("Bear 2022", "2022-01-01", "2022-12-31"),
    ("Bull 2023", "2023-01-01", "2024-01-01"),
]:
    try:
        _, _, macro_r = get_universal_data("SPY", start, end)
        if macro_r is None or len(macro_r) < 50:
            print(f"  ⚠️  {label} : données non disponibles")
            continue

        env_r  = RiskManagerEnv(macro_r)
        obs, _ = env_r.reset(seed=0)
        a_buf, v_buf = [], []

        for _ in range(min(200, len(macro_r) - 5)):
            act, _ = rm.predict(obs, deterministic=True)
            obs, _, done, _, info = env_r.step(act)
            a_buf.append(info["allocation"])
            v_buf.append(info["vix"])
            if done:
                break

        mean = np.mean(a_buf)
        print(f"\n  {label}")
        print(f"    VIX moyen    : {np.mean(v_buf):.2f} "
              f"(réel ~{np.mean(v_buf)*20:.0f})")
        print(f"    Allocation   : {mean:.1%}  "
              f"(min {np.min(a_buf):.0%} / max {np.max(a_buf):.0%})")

        if label == "Bear 2022":
            bear_alloc = mean
        else:
            bull_alloc = mean

    except Exception as e:
        print(f"  ⚠️  {label} : erreur — {e}")

if bear_alloc is not None and bull_alloc is not None:
    real_ok = bull_alloc > bear_alloc
    print(f"\n  Bull > Bear : {bull_alloc:.1%} > {bear_alloc:.1%}  "
          f"{'✅ OK' if real_ok else '❌ Inversé'}")


# ============================================================
# TEST 5 — VARIABILITÉ (ne doit pas être figé)
# ============================================================

print("\n" + "─" * 65)
print("TEST 5 — Variabilité (ne doit pas être figé sur une valeur)")
print("─" * 65)

# Séquence mixte : VIX oscille
n_v    = 400
macro_v = np.zeros((n_v, 3), dtype=np.float32)
for i in range(n_v):
    macro_v[i, 0] = 0.5 + 1.2 * abs(np.sin(i / 40))
    macro_v[i, 1] = 0.001 * np.cos(i / 20)

env_v  = RiskManagerEnv(macro_v)
obs, _ = env_v.reset(seed=0)
a_var  = []

for _ in range(350):
    act, _ = rm.predict(obs, deterministic=True)
    obs, _, done, _, info = env_v.step(act)
    a_var.append(info["allocation"])
    if done:
        break

std_var = np.std(a_var)
amp_var = max(a_var) - min(a_var)
var_ok  = std_var > 0.08 and amp_var > 0.20

print(f"\n  Std des allocations  : {std_var:.3f}  "
      f"({'✅' if std_var > 0.08 else '❌'} seuil > 0.08)")
print(f"  Amplitude min→max    : {min(a_var):.1%} → {max(a_var):.1%}  "
      f"({'✅' if amp_var > 0.20 else '❌'} amplitude > 20 pts)")


# ============================================================
# VERDICT FINAL
# ============================================================

print("\n" + "=" * 65)
print("🎯 VERDICT FINAL")
print("=" * 65)

tests = [
    ("Zones de marché",   score_t1 >= 0.67, f"{passed_t1}/{len(zones)} zones"),
    ("Corrélation VIX",   corr_ok,          f"corr = {corr:.3f}"),
    ("Réactivité chocs",  react_ok,         "calme > crash < rebond"),
    ("Données réelles",   real_ok,          "bull > bear"),
    ("Variabilité",       var_ok,           f"std = {std_var:.3f}"),
]

total_ok = sum(1 for _, ok, _ in tests if ok)
print()
for name, ok, detail in tests:
    print(f"  {'✅' if ok else '❌'}  {name:<22} {detail}")

print(f"\n  Score total : {total_ok}/5")
print()

if total_ok >= 4:
    print("  ✅✅ RISK MANAGER VALIDÉ")
    print("     → Tu peux lancer train.py")
elif total_ok == 3:
    print("  ⚠️  RISK MANAGER ACCEPTABLE")
    print("     → Résultats corrects mais pas optimaux")
    print("     → Relancer train_rm.py avec +100k steps")
    print("       OU continuer avec ce RM")
else:
    print("  ❌ RISK MANAGER INSUFFISANT")
    print("     → Ne pas lancer train.py")
    print("     → Relancer train_rm.py depuis le début")
    print("     → Vérifier que risk_manager.py est la bonne version")

print("=" * 65)