

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, BaseCallback, CallbackList
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from data_prep import get_universal_data
from risk_manager import RiskManagerEnv, TradingEnvWithRisk

print("=" * 60)
print("DUAL AGENT — ENTRAÎNEMENT FINAL")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# ── N_ENVS selon le device ────────────────────────────────────
# CPU Colab : 1 pour éviter deadlock + AssertionError set_env
# GPU       : 4 pour la parallélisation
N_ENVS = 1 if device == "cpu" else 4
print(f"N_ENVS : {N_ENVS}")

# ============================================================
# ÉTAPE 1 — ARCHITECTURE TCN+GRU
# Définie en premier pour pouvoir charger les modèles
# ============================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=0, dilation=dilation
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1   = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2   = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm    = nn.LayerNorm(channels)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.dropout(h)
        h = self.conv2(h) + x
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)
        return self.act(h)


class TCNGRUExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        n_features:    int   = 13,
        n_extras:      int   = 3,
        tcn_channels:  int   = 32,
        tcn_dilations        = (1, 2, 4),
        tcn_kernel:    int   = 3,
        gru_hidden:    int   = 64,
        dropout:       float = 0.2,
        features_dim:  int   = 128,
    ):
        super().__init__(observation_space, features_dim)
        self.n_features = n_features
        self.n_extras   = n_extras

        self.input_proj = nn.Linear(n_features, tcn_channels)
        self.tcn = nn.Sequential(*[
            TCNBlock(tcn_channels, tcn_kernel, d, dropout)
            for d in tcn_dilations
        ])
        self.gru = nn.GRU(
            tcn_channels, gru_hidden,
            num_layers=1, batch_first=True,
            dropout=0.0
        )
        self.norm_gru = nn.LayerNorm(gru_hidden)
        self.drop_out = nn.Dropout(dropout)

        in_head   = gru_hidden + max(n_extras, 0)
        self.head = nn.Sequential(
            nn.Linear(in_head, features_dim),
            nn.GELU(),
        )

        for name, p in self.gru.named_parameters():
            if 'weight_ih' in name:  nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name: nn.init.orthogonal_(p)
            elif 'bias' in name:      nn.init.zeros_(p)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat  = obs[:, :self.n_features]
        extra = obs[:, self.n_features:]
        x = feat.unsqueeze(1)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        z    = self.norm_gru(h.squeeze(0))
        z    = self.drop_out(z)
        if self.n_extras > 0:
            z = torch.cat([z, extra], dim=-1)
        return self.head(z)


# policy_kwargs partagé — utilisé pour créer ET charger le modèle
def make_policy_kwargs(n_features, n_extras):
    return dict(
        features_extractor_class  = TCNGRUExtractor,
        features_extractor_kwargs = dict(
            n_features    = n_features,
            n_extras      = n_extras,
            tcn_channels  = 32,
            tcn_dilations = (1, 2, 4),
            tcn_kernel    = 3,
            gru_hidden    = 64,
            dropout       = 0.2,
            features_dim  = 128,
        ),
        net_arch      = [64, 64],
        activation_fn = nn.GELU,
    )

# ============================================================
# ÉTAPE 2 — CHARGER LE RM FIGÉ
# ============================================================

print("\nChargement Risk Manager...")

RM_PATH = "./models/rm_20260530_152900.zip"
if not os.path.exists(RM_PATH):
    rm_files = sorted(glob.glob("./models/rm_*.zip"))
    if not rm_files:
        raise FileNotFoundError(
            "Aucun rm_*.zip trouvé.\n"
            "Lance d'abord : train_rm.py"
        )
    RM_PATH = rm_files[-1]
    print(f"  RM auto-détecté : {RM_PATH}")

risk_model = PPO.load(RM_PATH, device=device)
print(f"   RM chargé : {RM_PATH}")
print(f"   risk_model.learn() ne sera JAMAIS appelé")

# ============================================================
# ÉTAPE 3 — DONNÉES
# ============================================================

print("\nChargement données...")

STOCKS_TRAIN = [
    "AAPL", "MSFT", "JPM",  "KO",   "GS",
    "JNJ",  "NVDA", "HD",   "CVX",  "MCD",
    "TSLA", "GOOGL","AMZN", "META", "XOM",
    "WMT",  "V",    "MA",   "UNH",  "PG",
]

# Stocks non vus — JAMAIS dans STOCKS_TRAIN
STOCKS_VAL = [
    "NFLX", "PYPL", "COIN", "HOOD",
    "PLTR", "DIS",  "SCHW", "NKE",
]

PERIODS_TRAIN = [
    ("2013-01-01", "2015-12-31"),
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
    ("2023-01-01", "2024-06-30"),
]

PERIOD_VAL    = [("2022-01-01", "2023-12-31")]
PERIOD_BULL   = [
    ("2013-01-01", "2015-12-31"),
    ("2016-01-01", "2019-12-31"),
    ("2023-01-01", "2024-06-30"),
]
PERIOD_CRISIS = [
    ("2020-01-01", "2021-06-01"),
    ("2021-11-01", "2022-12-31"),
]

BULL_STOCKS   = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "V",    "MA",   "UNH",   "HD",
]
CRISIS_STOCKS = ["AAPL", "MSFT", "JPM", "KO", "GS", "META"]


def collect_data(tickers, periods, label=""):
    fl, pl, ml = [], [], []
    for tk in tickers:
        for s, e in periods:
            f, p, m = get_universal_data(tk, s, e)
            if f is not None and len(f) > 100:
                fl.append(f)
                pl.append(p)
                ml.append(
                    m if m is not None
                    else np.zeros((len(f), 3), dtype=np.float32)
                )
                print(f"  OK {tk:6s} {s[:4]}-{e[:4]} : {len(f)}j")
    if not fl:
        raise ValueError(f"Aucune donnée [{label}]")
    all_f = np.vstack(fl).astype(np.float32)
    all_p = np.concatenate(pl).astype(np.float32)
    all_m = np.vstack(ml).astype(np.float32)
    print(f"  => {label} : {len(all_f):,}j | {all_f.shape[1]} features\n")
    return all_f, all_p, all_m


print("Train :")
train_f, train_p, train_m = collect_data(
    STOCKS_TRAIN, PERIODS_TRAIN, "Train"
)
print("Validation (stocks non vus) :")
val_f, val_p, val_m = collect_data(
    STOCKS_VAL, PERIOD_VAL, "Val"
)

# ============================================================
# ÉTAPE 4 — DIMENSIONS AUTO
# ============================================================

_probe  = TradingEnvWithRisk(train_f, train_p, risk_model, train_m)
OBS_DIM = _probe.observation_space.shape[0]
N_FEAT  = train_f.shape[1]
N_EXTRA = OBS_DIM - N_FEAT
print(f"obs_dim={OBS_DIM}  N_FEAT={N_FEAT}  N_EXTRA={N_EXTRA}")
try:
    _probe.close()
except Exception:
    pass

# ============================================================
# ÉTAPE 5 — CALLBACKS
# ============================================================

class OverfitMonitor(BaseCallback):
    def __init__(self, check_freq=20_000):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0 \
                and self.num_timesteps > 0:
            infos  = self.locals.get("infos", [{}])
            worths = [i.get("net_worth", 10_000) for i in infos]
            avg_w  = np.mean(worths)
            ret    = (avg_w - 10_000) / 10_000 * 100
            print(f"\n  [{self.num_timesteps:>7,}] "
                  f"net_worth: ${avg_w:,.0f}  ({ret:+.1f}%)")
            if ret > 300:
                print("  ⚠️  Return > 300% → SUSPICION OVERFITTING")
        return True


# ============================================================
# UTILITAIRE DummyVecEnv — closure correcte
# ============================================================

def make_env_fn(f, p, rm, m):
    """
    Capture explicite des variables pour éviter le bug
    de closure Python dans DummyVecEnv.
    """
    _f, _p, _rm, _m = f, p, rm, m
    def _init():
        return TradingEnvWithRisk(_f, _p, _rm, _m)
    return _init


# ============================================================
# ÉTAPE 6 — CONSTRUCTION ENVS PRINCIPAUX
# ============================================================

os.makedirs("./models",            exist_ok=True)
os.makedirs("./best_model/trader", exist_ok=True)

train_env = DummyVecEnv([
    make_env_fn(train_f, train_p, risk_model, train_m)
    for _ in range(N_ENVS)
])
val_env = DummyVecEnv([
    make_env_fn(val_f, val_p, risk_model, val_m)
])

# ============================================================
# ÉTAPE 7 — CRÉER LE MODÈLE PPO + TCN+GRU
# ============================================================

policy_kwargs = make_policy_kwargs(N_FEAT, N_EXTRA)

trader = PPO(
    "MlpPolicy",
    train_env,
    learning_rate  = 3e-4,
    n_steps        = 512,
    batch_size     = 128,
    n_epochs       = 8,
    ent_coef       = 0.05,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    verbose        = 1,
    device         = device,
    policy_kwargs  = policy_kwargs,
    seed           = 42,
)

n_params = sum(p.numel() for p in trader.policy.parameters())
print(f"\nParamètres TCN+GRU : {n_params:,}")

# ============================================================
# PHASE 1 — ENTRAÎNEMENT PRINCIPAL
# ============================================================

eval_cb = EvalCallback(
    val_env,
    best_model_save_path = "./best_model/trader/",
    eval_freq            = max(25_000 // N_ENVS, 1000),
    n_eval_episodes      = 5,
    deterministic        = True,
    verbose              = 1,
)
overfit_cb = OverfitMonitor(check_freq=20_000)

TOTAL_STEPS = 300_000
print(f"\n{'='*60}")
print(f"PHASE 1 — Principal ({TOTAL_STEPS:,} steps)")
print(f"  N_ENVS   : {N_ENVS}")
print(f"  Val      : {STOCKS_VAL}")
print(f"  Dropout  : 0.2")
print(f"{'='*60}")

trader.learn(
    total_timesteps = TOTAL_STEPS,
    callback        = CallbackList([eval_cb, overfit_cb]),
    progress_bar    = True,
)
train_env.close()
val_env.close()
trader.save("./models/trader_phase1")
print(" Phase 1 : ./models/trader_phase1.zip")

# ============================================================
# PHASE 2 — FINE-TUNING BULL
# ============================================================

print(f"\n{'='*60}")
print("PHASE 2 — Fine-tuning Bull (100k steps)")
print(f"{'='*60}")

bull_f, bull_p, bull_m = collect_data(
    BULL_STOCKS, PERIOD_BULL, "Bull"
)

bull_env = DummyVecEnv([
    make_env_fn(bull_f, bull_p, risk_model, bull_m)
    for _ in range(N_ENVS)
])

trader.set_env(bull_env)
trader.learn(
    total_timesteps     = 100_000,
    reset_num_timesteps = False,
    progress_bar        = True,
)
bull_env.close()
trader.save("./models/trader_bull")
print(" Phase 2 Bull : ./models/trader_bull.zip")

# ============================================================
# PHASE 3 — RAPPEL CRISIS
# Évite l'oubli catastrophique après le fine-tuning bull
# ============================================================

print(f"\n{'='*60}")
print("PHASE 3 — Rappel Crisis (30k steps)")
print(f"{'='*60}")

crisis_f, crisis_p, crisis_m = collect_data(
    CRISIS_STOCKS, PERIOD_CRISIS, "Crisis"
)

crisis_env = DummyVecEnv([
    make_env_fn(crisis_f, crisis_p, risk_model, crisis_m)
    for _ in range(N_ENVS)
])

trader.set_env(crisis_env)
trader.learn(
    total_timesteps     = 30_000,
    reset_num_timesteps = False,
    progress_bar        = True,
)
crisis_env.close()

trader.save("./models/trader_final")
print(" Phase 3 Crisis : ./models/trader_final.zip")

# ============================================================
# TEST ANTI-OVERFIT FINAL
# ============================================================

print(f"\n{'='*60}")
print("VÉRIFICATION ANTI-OVERFIT")
print(f"{'='*60}")


def test_correct(ticker, start, end):
    """
    Test correct :
    - Prix bruts (pas log-normalisés)
    - net_worth depuis info (pas reward)
    """
    f, p, m = get_universal_data(ticker, start, end)
    if f is None or len(f) < 50:
        return None, None
    if m is None:
        m = np.zeros((len(f), 3), dtype=np.float32)

    env      = TradingEnvWithRisk(f, p, risk_model, m)
    obs, _   = env.reset()
    portfolio = [10_000.0]
    done      = False

    while not done:
        act, _ = trader.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
        portfolio.append(float(info.get("net_worth", portfolio[-1])))

    ret = (portfolio[-1] - 10_000) / 10_000 * 100
    bh  = (float(p[-1]) / float(p[0]) - 1) * 100
    return ret, bh


test_cases = [
    # Seen
    ("AAPL", "2023-01-01", "2024-01-01", "SEEN",   "BULL"),
    ("MSFT", "2022-01-01", "2022-12-31", "SEEN",   "BEAR"),
    ("TSLA", "2022-01-01", "2022-12-31", "SEEN",   "BEAR"),
    # Unseen
    ("NFLX", "2022-01-01", "2022-12-31", "UNSEEN", "BEAR"),
    ("PYPL", "2022-01-01", "2022-12-31", "UNSEEN", "BEAR"),
    ("COIN", "2022-01-01", "2022-12-31", "UNSEEN", "BEAR"),
    ("HOOD", "2023-01-01", "2024-01-01", "UNSEEN", "BULL"),
    ("PLTR", "2023-01-01", "2024-01-01", "UNSEEN", "BULL"),
    ("META", "2022-01-01", "2022-12-31", "UNSEEN", "BEAR"),
]

results_seen, results_unseen = [], []
wins_seen, wins_unseen = 0, 0
total_seen, total_unseen = 0, 0

print(f"\n  {'Ticker':<6} {'Status':<8} {'Type':<6} "
      f"{'Dual':>8} {'B&H':>8}  Résultat")
print("  " + "-" * 52)

for ticker, start, end, status, mtype in test_cases:
    ret, bh = test_correct(ticker, start, end)
    if ret is None:
        continue
    win  = ret > bh
    flag = " ⚠️ OVERFIT?" if abs(ret) > 500 else ""
    print(f"  {'✅' if win else '❌'} {ticker:<6} {status:<8} "
          f"{mtype:<6} {ret:>+7.1f}% {bh:>+7.1f}%{flag}")
    if status == "SEEN":
        results_seen.append(ret)
        total_seen += 1
        if win: wins_seen += 1
    else:
        results_unseen.append(ret)
        total_unseen += 1
        if win: wins_unseen += 1

# ── Diagnostic ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DIAGNOSTIC OVERFITTING")
print(f"{'='*60}")

if results_seen and results_unseen:
    avg_seen   = np.mean(results_seen)
    avg_unseen = np.mean(results_unseen)
    gap        = avg_seen - avg_unseen
    gen_rate   = wins_unseen / total_unseen * 100

    print(f"\n  SEEN   : {wins_seen}/{total_seen} victoires "
          f"| Return moy: {avg_seen:+.1f}%")
    print(f"  UNSEEN : {wins_unseen}/{total_unseen} victoires "
          f"| Return moy: {avg_unseen:+.1f}%")
    print(f"  Gap    : {gap:+.1f}%")
    print()

    if gap > 80:
        print("   OVERFITTING (gap > 80%)")
    elif gap > 40:
        print("    GAP MODÉRÉ — surveiller")
    else:
        print("   PAS D'OVERFITTING")

    print(f"\n  Taux généralisation UNSEEN : "
          f"{wins_unseen}/{total_unseen} ({gen_rate:.0f}%)")
    if gen_rate >= 60:
        print("   Bonne généralisation AGI !")
    elif gen_rate >= 40:
        print("    Généralisation moyenne")
    else:
        print("   Généralisation insuffisante")

# ── Résumé ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("RÉSUMÉ FINAL")
print(f"{'='*60}")
print(f"  RM      : {RM_PATH}")
print(f"  Trader  : ./models/trader_final.zip")
print(f"  Meilleur: ./best_model/trader/best_model.zip")
print(f"  N_ENVS  : {N_ENVS}")
print(f"  Dropout : 0.2")
print(f"{'='*60}")
print("Lance : python test_proof_final.py")