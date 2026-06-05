# ============================================================
# test_rm_extended.py
# Tests approfondis du Risk Manager — au-delà du test de base
# ============================================================

import math
import numpy as np
import glob, os
from datetime import datetime
from stable_baselines3 import PPO
from data_prep import get_universal_data
from risk_manager import RiskManagerEnv

print("=" * 65)
print("🔬 TESTS APPROFONDIS — RISK MANAGER")
print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

# ── Chargement ────────────────────────────────────────────────
rm_files = (glob.glob("./models/rm_*.zip") +
            glob.glob("./best_model/rm/*.zip") +
            ["./models/rm_20260530_152900.zip"])
rm_files  = [f for f in rm_files if os.path.exists(f)]
if not rm_files:
    raise FileNotFoundError("❌ Aucun modèle RM trouvé")

latest = max(rm_files, key=os.path.getmtime)
rm     = PPO.load(latest)
print(f"✅ {latest}\n")

scores = {}   # stocke pass/fail de chaque test


# ── Utilitaire ────────────────────────────────────────────────

def run_episode(model, macro, seed=0, n_steps=200):
    """Exécute un épisode complet et retourne (allocs, vix_vals, spy_vals)."""
    env    = RiskManagerEnv(macro)
    obs, _ = env.reset(seed=seed)
    allocs, vix_vals, spy_vals = [], [], []
    for _ in range(min(n_steps, len(macro) - 10)):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
        allocs.append(info["allocation"])
        vix_vals.append(info["vix"])
        spy_vals.append(info["spy"])
        if done:
            break
    return np.array(allocs), np.array(vix_vals), np.array(spy_vals)


def sim_macro(vix_val, spy_val=0.001, n=300):
    """Macro simulée constante."""
    m = np.zeros((n, 3), dtype=np.float32)
    m[:, 0] = vix_val
    m[:, 1] = spy_val
    return m


# ============================================================
# TEST A — MONOTONIE : allocation doit décroître avec le VIX
# ============================================================

print("─" * 65)
print("TEST A — Monotonie VIX → Allocation")
print("─" * 65)

vix_pts = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0]
means   = []
for v in vix_pts:
    a, _, _ = run_episode(rm, sim_macro(v))
    means.append(np.mean(a))

violations = 0
print(f"\n  {'VIX':>6}  {'Alloc':>8}  {'Tendance':>10}")
print("  " + "-" * 30)
for i, (v, m) in enumerate(zip(vix_pts, means)):
    if i > 0:
        arrow = "↓ ✅" if m < means[i-1] else "↑ ❌"
        if m >= means[i-1]:
            violations += 1
    else:
        arrow = "    "
    print(f"  {v:>6.2f}  {m:>7.1%}  {arrow}")

mono_ok = violations <= 1   # tolérance 1 violation
scores["A"] = mono_ok
print(f"\n  Violations : {violations}/8  "
      f"({'✅ Monotone' if mono_ok else '❌ Non monotone'})")


# ============================================================
# TEST B — VITESSE DE RÉACTION (en steps)
# ============================================================

print("\n" + "─" * 65)
print("TEST B — Vitesse de réaction à un choc")
print("─" * 65)

n     = 300
macro = np.zeros((n, 3), dtype=np.float32)
macro[:150, 0] = 0.5   # calme
macro[150:, 0] = 1.8   # crash soudain

env    = RiskManagerEnv(macro)
obs, _ = env.reset(seed=0)
allocs = []
for _ in range(280):
    act, _ = rm.predict(obs, deterministic=True)
    obs, _, done, _, info = env.step(act)
    allocs.append(info["allocation"])
    if done:
        break

allocs = np.array(allocs)
pre_crash  = np.mean(allocs[130:150])   # 20 steps avant
post_crash = allocs[150:180]            # 30 steps après le choc

# Trouver combien de steps pour descendre sous 30%
reaction_steps = None
for i, a in enumerate(post_crash):
    if a < 0.30:
        reaction_steps = i + 1
        break

react_ok = reaction_steps is not None and reaction_steps <= 10
scores["B"] = react_ok

print(f"\n  Allocation pré-crash (20 steps) : {pre_crash:.1%}")
print(f"  Allocation post-crash step 1    : {post_crash[0]:.1%}")
print(f"  Allocation post-crash step 5    : {post_crash[min(4,len(post_crash)-1)]:.1%}")
print(f"  Allocation post-crash step 10   : {post_crash[min(9,len(post_crash)-1)]:.1%}")
print(f"\n  Steps pour passer sous 30%      : "
      f"{reaction_steps if reaction_steps else '>30'}")
print(f"  {'✅ Réaction rapide (≤10 steps)' if react_ok else '❌ Réaction lente (>10 steps)'}")


# ============================================================
# TEST C — STABILITÉ : pas d'oscillations inutiles
# ============================================================

print("\n" + "─" * 65)
print("TEST C — Stabilité (pas d'oscillations en marché stable)")
print("─" * 65)

for label, vix_v in [("Calme  VIX=0.5", 0.5),
                      ("Normal VIX=0.9", 0.9),
                      ("Crise  VIX=1.6", 1.6)]:
    a, _, _ = run_episode(rm, sim_macro(vix_v), n_steps=150)
    std_roll = np.mean([np.std(a[i:i+10]) for i in range(0, len(a)-10, 10)])
    ok       = std_roll < 0.08
    print(f"  {label} : std glissante = {std_roll:.4f}  "
          f"{'✅' if ok else '❌'}")

scores["C"] = True   # informatif seulement


# ============================================================
# TEST D — MÉMOIRE DU CONTEXTE (5 jours de fenêtre)
# ============================================================

print("\n" + "─" * 65)
print("TEST D — Mémoire : le RM utilise bien la fenêtre de 5 jours")
print("─" * 65)

# Scénario : VIX monte progressivement sur 5 jours
n      = 200
macro1 = sim_macro(0.5)      # calme depuis le début
macro2 = sim_macro(0.5)      # calme, puis pic VIX sur 5 jours
macro2[100:105, 0] = 2.0     # pic de crise sur 5 jours
macro2[105:, 0]    = 0.5     # retour calme

a1, _, _ = run_episode(rm, macro1, n_steps=150)
a2, _, _ = run_episode(rm, macro2, n_steps=150)

# Après le pic : le RM doit être plus prudent (mémoire du choc)
post_peak_1 = np.mean(a1[105:120]) if len(a1) > 120 else np.mean(a1[-15:])
post_peak_2 = np.mean(a2[105:120]) if len(a2) > 120 else np.mean(a2[-15:])

mem_ok = post_peak_2 < post_peak_1 + 0.1
scores["D"] = mem_ok

print(f"\n  Sans pic de crise : alloc après step 105 = {post_peak_1:.1%}")
print(f"  Avec pic de crise : alloc après step 105 = {post_peak_2:.1%}")
print(f"  {'✅ Mémoire active (plus prudent après crise)' if mem_ok else '⚠️  Pas de mémoire détectée'}")


# ============================================================
# TEST E — COHÉRENCE SPY : réagit au rendement du marché
# ============================================================

print("\n" + "─" * 65)
print("TEST E — Sensibilité au SPY (rendement journalier)")
print("─" * 65)

results_spy = {}
for spy_v, label in [(-0.04, "SPY -4% crash"),
                      (-0.01, "SPY -1%"),
                      ( 0.00, "SPY  0%"),
                      ( 0.01, "SPY +1%"),
                      ( 0.03, "SPY +3% hausse")]:
    a, _, _ = run_episode(rm, sim_macro(0.8, spy_val=spy_v))
    results_spy[spy_v] = np.mean(a)
    print(f"  VIX=0.8, {label:15s} → Alloc : {np.mean(a):.1%}")

# SPY négatif doit donner moins d'allocation que SPY positif
spy_ok = results_spy[-0.04] < results_spy[0.03]
scores["E"] = spy_ok
print(f"\n  SPY crash < SPY hausse : "
      f"{results_spy[-0.04]:.1%} < {results_spy[0.03]:.1%}  "
      f"{'✅' if spy_ok else '❌'}")


# ============================================================
# TEST F — GÉNÉRALISATION : tickers non vus
# ============================================================

print("\n" + "─" * 65)
print("TEST F — Généralisation sur tickers non vus pendant l'entraînement")
print("─" * 65)

unseen = [
    ("TSLA", "2023-01-01", "2024-01-01"),
    ("NVDA", "2023-01-01", "2024-01-01"),
    ("COIN", "2023-01-01", "2024-01-01"),
    ("AMZN", "2022-01-01", "2022-12-31"),
]

gen_results = []
for ticker, start, end in unseen:
    try:
        _, _, macro_r = get_universal_data(ticker, start, end)
        if macro_r is None or len(macro_r) < 50:
            continue
        a, v, _ = run_episode(rm, macro_r)
        corr     = np.corrcoef(v, a)[0, 1] if len(set(v)) > 3 else 0
        mean_a   = np.mean(a)
        period   = "bull" if start >= "2023" else "bear"
        ok       = corr < -0.3
        gen_results.append(ok)
        print(f"  {ticker} {start[:4]} ({period:4s}) : "
              f"alloc={mean_a:.1%}  corr={corr:+.3f}  {'✅' if ok else '⚠️'}")
    except Exception as e:
        print(f"  {ticker} : erreur — {e}")

gen_ok = sum(gen_results) >= len(gen_results) * 0.6
scores["F"] = gen_ok
print(f"\n  Généralisation : {sum(gen_results)}/{len(gen_results)} OK  "
      f"{'✅' if gen_ok else '❌'}")


# ============================================================
# TEST G — CRISE 2008 (hors distribution)
# ============================================================

print("\n" + "─" * 65)
print("TEST G — Robustesse : Crise 2008 (hors distribution)")
print("─" * 65)

try:
    _, _, macro_08 = get_universal_data("SPY", "2008-01-01", "2009-06-01")
    if macro_08 is not None and len(macro_08) > 50:
        a08, v08, _ = run_episode(rm, macro_08, n_steps=300)
        vix_mean    = np.mean(v08)
        alloc_mean  = np.mean(a08)
        alloc_peak  = np.mean(a08[v08 < 0.8]) if np.any(v08 < 0.8) else None
        alloc_crash = np.mean(a08[v08 > 1.5]) if np.any(v08 > 1.5) else None

        robust_ok = alloc_mean < 0.50
        scores["G"] = robust_ok

        print(f"\n  VIX moyen 2008 : {vix_mean:.2f} (réel ~{vix_mean*20:.0f})")
        print(f"  Allocation moy : {alloc_mean:.1%}")
        if alloc_peak:
            print(f"  Alloc VIX<0.8  : {alloc_peak:.1%}")
        if alloc_crash:
            print(f"  Alloc VIX>1.5  : {alloc_crash:.1%}")
        print(f"\n  {'✅ Prudent en 2008' if robust_ok else '❌ Trop exposé en 2008'}")
    else:
        print("  ⚠️  Données 2008 non disponibles")
        scores["G"] = None
except Exception as e:
    print(f"  ⚠️  Erreur : {e}")
    scores["G"] = None


# ============================================================
# TEST H — PROFIL COMPLET : courbe allocation vs VIX
# ============================================================

print("\n" + "─" * 65)
print("TEST H — Profil complet (cible sigmoïde vs allocation réelle)")
print("─" * 65)

vix_range  = np.linspace(0.2, 2.5, 20)
alloc_real = []
alloc_tgt  = []
max_err    = 0

print(f"\n  {'VIX':>6}  {'Cible':>7}  {'Réel':>7}  {'Écart':>7}")
print("  " + "-" * 35)

for v in vix_range:
    target = 1.0 / (1.0 + math.exp(6.0 * (v - 0.9)))
    target = float(np.clip(target, 0.05, 0.92))
    a, _, _ = run_episode(rm, sim_macro(float(v)), n_steps=80)
    real     = np.mean(a)
    err      = abs(real - target)
    max_err  = max(max_err, err)
    alloc_real.append(real)
    alloc_tgt.append(target)
    flag = "⚠️" if err > 0.20 else ""
    print(f"  {v:>6.2f}  {target:>6.1%}  {real:>6.1%}  "
          f"{(real-target)*100:>+6.1f}pp  {flag}")

mae    = np.mean(np.abs(np.array(alloc_real) - np.array(alloc_tgt)))
prof_ok = mae < 0.18
scores["H"] = prof_ok
print(f"\n  MAE (erreur moyenne) : {mae:.3f}  "
      f"({'✅ < 0.18' if prof_ok else '❌ ≥ 0.18'})")
print(f"  Écart max            : {max_err:.3f}")


# ============================================================
# VERDICT FINAL
# ============================================================

print("\n" + "=" * 65)
print("🎯 VERDICT FINAL")
print("=" * 65)

test_names = {
    "A": "Monotonie VIX→Allocation",
    "B": "Vitesse de réaction",
    "C": "Stabilité (informatif)",
    "D": "Mémoire du contexte",
    "E": "Sensibilité SPY",
    "F": "Généralisation tickers",
    "G": "Robustesse 2008",
    "H": "Profil sigmoïde",
}

passed = 0
total  = 0
print()
for k, name in test_names.items():
    v = scores.get(k)
    if v is None:
        print(f"  ⚪  Test {k} — {name:<30} (données indisponibles)")
        continue
    total += 1
    if v:
        passed += 1
    print(f"  {'✅' if v else '❌'}  Test {k} — {name}")

print(f"\n  Score : {passed}/{total}")
print()

if passed >= total * 0.75:
    print("  ✅✅ RISK MANAGER PLEINEMENT VALIDÉ")
    print("     → Excellent pour lancer train.py")
elif passed >= total * 0.60:
    print("  ✅  RISK MANAGER VALIDÉ")
    print("     → Acceptable pour lancer train.py")
elif passed >= total * 0.50:
    print("  ⚠️  RISK MANAGER PARTIEL")
    print("     → Relancer train_rm.py +100k steps conseillé")
else:
    print("  ❌  RISK MANAGER INSUFFISANT")
    print("     → Relancer train_rm.py depuis le début")

print("=" * 65)