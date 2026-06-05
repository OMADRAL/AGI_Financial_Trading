# ============================================================
# train_rm.py — VERSION CORRIGÉE
# Données : marché calme 2010-2019 + crises 2020-2023
# ============================================================

import numpy as np, os, time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_prep import get_universal_data
from risk_manager import RiskManagerEnv

os.makedirs("./models", exist_ok=True)

# ── DONNÉES : équilibre calme + crise ────────────────────────
STOCKS = ["AAPL", "MSFT", "JPM", "KO", "GS",
          "JNJ",  "CVX",  "HD",  "MCD", "MMM"]

PERIODS = [
    ("2010-01-01", "2015-12-31"),  # ← marché calme long (VIX bas)
    ("2016-01-01", "2019-12-31"),  # ← bull market calme
    ("2020-01-01", "2021-06-01"),  # ← COVID crash
    ("2021-11-01", "2022-12-31"),  # ← bear market
    ("2023-01-01", "2024-06-01"),  # ← rebond bull
]

print("📥 Chargement données...")
macro_list = []
for tk in STOCKS:
    for s, e in PERIODS:
        _, _, m = get_universal_data(tk, s, e)
        if m is not None and len(m) > 100:
            macro_list.append(m)
            print(f"  ✅ {tk} {s[:4]}-{e[:4]} : {len(m)} jours")

# Concaténer DIRECTEMENT — pas de moyenne
# La moyenne écrase la variabilité VIX dont le RM a besoin
macro_all = np.vstack(macro_list).astype(np.float32)
print(f"\n✅ Total : {len(macro_all):,} jours macro")

# Stats pour vérifier la diversité
vix_vals = macro_all[:, 0]
print(f"   VIX moyen : {vix_vals.mean():.2f} "
      f"| min : {vix_vals.min():.2f} "
      f"| max : {vix_vals.max():.2f}")
print(f"   % jours VIX < 0.7 (calme) : "
      f"{(vix_vals < 0.7).mean():.0%}")
print(f"   % jours VIX > 1.3 (crise) : "
      f"{(vix_vals > 1.3).mean():.0%}")

# ── Validation sur période récente ───────────────────────────
_, _, macro_val = get_universal_data("SPY", "2023-01-01", "2024-06-01")
if macro_val is None:
    macro_val = macro_all[-500:]

# ── Entraînement ─────────────────────────────────────────────
print("\n🚀 Entraînement Risk Manager (400k steps)...")

N_ENV   = 4
env_trn = DummyVecEnv([
    (lambda: lambda: RiskManagerEnv(macro_all))()
    for _ in range(N_ENV)
])

rm_model = PPO(
    "MlpPolicy", env_trn,
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 512,
    n_epochs      = 10,
    ent_coef      = 0.05,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    verbose       = 0,
    device        = "cuda",
    policy_kwargs = dict(net_arch=[128, 128, 64]),
)

print(f"📐 Params : "
      f"{sum(p.numel() for p in rm_model.policy.parameters()):,}")

# Monitoring toutes les 50k steps
from stable_baselines3.common.callbacks import BaseCallback

class MonitorCallback(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % 50_000 == 0:
            allocs = {}
            for vix_t in [0.4, 0.7, 1.0, 1.5, 2.0]:
                macro_t = np.zeros((200, 3), dtype=np.float32)
                macro_t[:, 0] = vix_t
                macro_t[:, 1] = 0.001
                env_t   = RiskManagerEnv(macro_t)
                obs, _  = env_t.reset(seed=0)
                a_buf   = []
                for _ in range(80):
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _, info = env_t.step(act)
                    a_buf.append(info["allocation"])
                    if done: break
                allocs[vix_t] = np.mean(a_buf)
            print(f"  [{self.num_timesteps:>7,}] "
                  f"VIX=0.4→{allocs[0.4]:.0%} | "
                  f"VIX=0.7→{allocs[0.7]:.0%} | "
                  f"VIX=1.0→{allocs[1.0]:.0%} | "
                  f"VIX=1.5→{allocs[1.5]:.0%} | "
                  f"VIX=2.0→{allocs[2.0]:.0%}")
        return True

rm_model.learn(
    total_timesteps = 400_000,
    callback        = MonitorCallback(),
    progress_bar    = True,
)
env_trn.close()

ts   = time.strftime("%Y%m%d_%H%M%S")
path = f"./models/rm_{ts}"
rm_model.save(path)
print(f"\n💾 Sauvegardé : {path}.zip")
print("➡️  Lancer : python test_rm_efficacy.py")