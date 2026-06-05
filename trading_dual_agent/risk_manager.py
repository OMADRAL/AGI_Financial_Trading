# ============================================================
# risk_manager.py — VERSION FINALE CORRIGÉE
# - RiskManagerController avec état interne
# - Clipping des rewards à ±2%
# - Limitation de la position maximale à 20% (réaliste)
# ============================================================

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class RiskManagerEnv(gym.Env):
    """Environnement d'entraînement du Risk Manager"""

    def __init__(self, macro_features):
        super().__init__()
        self.macro   = np.array(macro_features, dtype=np.float32)
        self.n       = len(self.macro)
        self.n_mac   = self.macro.shape[1]
        self.window  = 5
        self.alloc_hist = deque(maxlen=50)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window * self.n_mac + 3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(self.n - 252 - 1, self.window + 1)
        self.t    = np.random.randint(self.window, max_start)
        self.end  = min(self.t + 252, self.n - 1)
        self.alloc_hist.clear()
        return self._obs(), {}

    def _obs(self):
        s = max(0, self.t - self.window + 1)
        w = self.macro[s:self.t + 1]
        if len(w) < self.window:
            pad = np.zeros((self.window - len(w), self.n_mac), dtype=np.float32)
            w   = np.vstack([pad, w])
        base  = w.flatten()
        amean = np.mean(list(self.alloc_hist)) if self.alloc_hist else 0.5
        vt    = float(self.macro[self.t][0]) - \
                float(self.macro[max(0, self.t - 5)][0])
        s5    = float(np.sum(self.macro[max(0, self.t - 5):self.t, 1]))
        return np.concatenate([base, [amean, vt, s5]]).astype(np.float32)

    def step(self, action):
        raw        = float(action[0])
        allocation = float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))
        self.alloc_hist.append(allocation)

        idx = min(self.t, self.n - 1)
        vix = float(self.macro[idx][0])
        spy = float(self.macro[idx][1])

        # Cible sigmoïde lisse
        target = 1.0 / (1.0 + math.exp(6.0 * (vix - 0.9)))
        target = float(np.clip(target, 0.05, 0.92))

        # Reward gaussienne autour de la cible
        error  = allocation - target
        reward = 6.0 * math.exp(-8.0 * error ** 2)

        # Bonus SPY
        if spy > 0.005:
            reward += allocation * 2.0
        elif spy < -0.015:
            reward += (1.0 - allocation) * 2.0

        # Pénalité stagnation
        if len(self.alloc_hist) > 10:
            std = float(np.std(list(self.alloc_hist)[-10:]))
            if std < 0.02:
                reward -= 3.0

        self.t += 1
        done = self.t >= self.end

        return self._obs(), reward, done, False, {
            "allocation": allocation,
            "vix":        vix,
            "spy":        spy,
            "target":     target,
        }


class RiskManagerController:
    """
    Contrôleur pour le Risk Manager - CORRIGÉ
    Maintient son propre état interne pour que l'allocation soit cohérente
    """

    def __init__(self, model, macro_features):
        self.model = model
        self.macro = np.array(macro_features, dtype=np.float32) if macro_features is not None else None
        self.alloc_history = deque(maxlen=50)
        self.current_step = 0
        self.window = 5
        self.n_mac = self.macro.shape[1] if self.macro is not None and len(self.macro) > 0 else 3

    def _get_obs(self, step):
        """Construit l'observation pour le RM au step donné"""
        if self.macro is None or len(self.macro) == 0:
            # Pas de données macro → allocation par défaut
            return np.zeros(self.window * self.n_mac + 3, dtype=np.float32)
        
        idx = min(step, len(self.macro) - 1)
        
        # Fenêtre des macros
        s = max(0, idx - self.window + 1)
        w = self.macro[s:idx + 1]
        
        if len(w) < self.window:
            pad = np.zeros((self.window - len(w), self.n_mac), dtype=np.float32)
            w = np.vstack([pad, w])
        
        base = w.flatten()
        
        # Moyenne des allocations récentes
        amean = np.mean(list(self.alloc_history)) if self.alloc_history else 0.5
        
        # Variation du VIX sur 5 jours
        vix_prev = float(self.macro[max(0, idx - 5)][0]) if idx >= 5 else 0.0
        vt = float(self.macro[idx][0]) - vix_prev
        
        # Somme SPY sur 5 jours
        s5 = float(np.sum(self.macro[max(0, idx - 5):idx, 1])) if idx >= 5 else 0.0
        
        return np.concatenate([base, [amean, vt, s5]]).astype(np.float32)

    def get_allocation(self, current_step):
        """
        Retourne l'allocation pour le step courant
        """
        if self.macro is None or len(self.macro) == 0:
            return 0.5
        
        idx = min(current_step, len(self.macro) - 1)
        
        # Construire l'observation
        obs = self._get_obs(current_step)
        
        # Prédire l'action du RM
        action, _ = self.model.predict(obs, deterministic=True)
        raw = float(action[0])
        
        # Convertir [-1, 1] → [0, 1]
        allocation = float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))
        
        # Mettre à jour l'historique
        self.alloc_history.append(allocation)
        self.current_step = current_step
        
        return allocation

    def reset(self):
        """Reset l'état interne du contrôleur"""
        self.alloc_history.clear()
        self.current_step = 0


class TradingEnvWithRisk(gym.Env):
    """
    Environnement de trading avec Risk Manager intégré
    VERSION FINALE CORRIGÉE:
    - RM réinitialisé correctement
    - Clipping des rewards à ±2%
    - Limitation de la position maximale à 20% (réaliste)
    """

    def __init__(self, features, prices, risk_model, macro_features,
                 initial_capital=10_000):
        super().__init__()
        from env import UniversalTradingEnv

        self.base = UniversalTradingEnv(features, prices, initial_capital)
        self.rm = RiskManagerController(risk_model, macro_features)
        self.t = 0
        
        # 🔧 CORRECTION: Limiter la position maximale à 20% du capital (réaliste)
        self.max_position = 0.2  # ← MODIFIÉ: 0.5 → 0.2

        self.observation_space = self.base.observation_space
        self.action_space = self.base.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.base.reset(seed=seed)
        self.t = 0
        self.rm.reset()
        return obs.astype(np.float32), info

    def step(self, trader_action):
        # Récupérer l'allocation du Risk Manager
        risk = self.rm.get_allocation(self.t)
        
        raw_pos = float(trader_action[0])
        
        # 🔧 CORRECTION 1: Limiter la position finale à max_position
        final_pos = float(np.clip(raw_pos * risk, -self.max_position, self.max_position))

        # Exécuter l'ordre
        obs, reward, done, trunc, info = self.base.step([final_pos])
        
        # 🔧 CORRECTION 2: Clipping des rewards à ±2% par jour
        reward = np.clip(reward, -0.02, 0.02)
        
        self.t += 1

        # Ajouter les informations de risque
        info["risk_allocation"] = risk
        info["raw_position"] = raw_pos
        info["final_position"] = final_pos
        info["max_position_limit"] = self.max_position

        return obs.astype(np.float32), reward, done, trunc, info