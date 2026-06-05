# Deep Multi-Agent Reinforcement Learning for Autonomous Financial Trading

This repository presents a Deep Reinforcement Learning (DRL) framework for autonomous financial trading. The project investigates the limitations of traditional monolithic trading agents and proposes a hierarchical Multi-Agent architecture designed to improve robustness, risk management, and long-term performance during periods of market stress and macroeconomic uncertainty.

## 📌 Project Overview

Traditional DRL-based trading systems often struggle to balance two conflicting objectives:

* Maximizing short-term returns.
* Preserving capital during periods of extreme market volatility.

To address this challenge, two architectures were developed and evaluated:

### 1. Single Agent (Baseline)

A monolithic trading agent based on:

* Conv1D for local pattern extraction.
* GRU for temporal dependency modeling.
* PPO (Proximal Policy Optimization) for policy learning.

Although the model demonstrates strong generalization capabilities on stable assets (60% zero-shot generalization), it remains vulnerable to major market crashes due to the absence of explicit macroeconomic awareness.

### 2. Dual Agent (Proposed Architecture)

A hierarchical Multi-Agent Reinforcement Learning framework that separates trading decisions from risk management.

#### Trader Agent

* TCN (Temporal Convolutional Network)
* GRU (Gated Recurrent Unit)
* PPO optimization

The Trader focuses on identifying local market opportunities and generating trading signals.

#### Risk Manager Agent

* Independent MLP controller
* Driven by a normalized VIX index

The Risk Manager dynamically adjusts the Trader's maximum market exposure according to global market stress conditions, improving capital preservation during crises.

---

## 📂 Repository Structure

```text
.
├── Single_Agent/
│   ├── data_prep.py
│   ├── env.py
│   ├── rnn_agent.py
│   ├── train.py
│   └── test.py
│
└── trading_dual_agent/
    ├── risk_manager.py
    ├── models.py
    ├── train_rm.py
    ├── train.py
    └── evaluate_dual_*.py
```

### Single_Agent/

Baseline DRL implementation:

* `data_prep.py` — Feature engineering (log returns, relative volatility).
* `env.py` — Custom Gymnasium trading environment.
* `rnn_agent.py` — Conv1D + GRU feature extractor.
* `train.py` — Training pipeline.
* `test.py` — Evaluation and backtesting.

### trading_dual_agent/

Multi-Agent architecture:

* `risk_manager.py` — Risk Manager environment and controller.
* `models.py` — TCN + GRU architecture with causal dilations.
* `train_rm.py` — Risk Manager training.
* `train.py` — Trader training under risk constraints.
* `evaluate_dual_*.py` — Backtesting and stress-testing scripts.

---

## 🚀 Key Results

The proposed Dual Agent architecture demonstrated strong robustness across diverse market regimes:

* **80% Win Rate** against the Buy & Hold benchmark across a panel of 10 financial assets.
* **Average Maximum Drawdown reduced to 3.0%** during major market crises.
* **Successful navigation of the 2008 Subprime Crisis and the 2020 COVID-19 Crash.**
* **+2090% cumulative return on AAPL (2015–2025)** compared with **+796%** for Buy & Hold.
* **Minimal overfitting**, with only a **1.7 percentage point gap** between in-sample and out-of-sample performance.

---

## 🛠️ Installation

Install the required dependencies:

```bash
pip install torch numpy pandas yfinance stable-baselines3 gymnasium plotly
```

## ▶️ Usage

Train the baseline model:

```bash
python Single_Agent/train.py
```

Train the Risk Manager:

```bash
python trading_dual_agent/train_rm.py
```

Train the Dual-Agent system:

```bash
python trading_dual_agent/train.py
```

Run evaluation:

```bash
python trading_dual_agent/evaluate_dual.py
```

---

## 📊 Research Focus

* Deep Reinforcement Learning
* Algorithmic Trading
* Multi-Agent Systems
* Risk-Aware Portfolio Management
* Temporal Convolutional Networks (TCN)
* Gated Recurrent Units (GRU)
* Proximal Policy Optimization (PPO)

---

## 👥 Authors

* Oumaima Dribi Alaoui
* Fatima-Zahrae Farfari


