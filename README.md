Deep Multi-Agent Reinforcement Learning for Autonomous Financial Trading
This repository contains the source code for our trading project. It explores the application of Deep Reinforcement Learning (DRL) to algorithmic trading, aiming to solve the vulnerability of monolithic DRL agents to macroeconomic shocks and systemic market crashes.
📌 Project Overview
Traditional RL agents often struggle to balance the contradictory goals of short-term profit maximization and long-term capital preservation. To address this, we developed and compared two architectures:
Single Agent (Baseline): A monolithic agent using Conv1D + GRU and PPO. While it generalizes well on stable assets (60% zero-shot generalization), it suffers from severe drawdowns (up to -46%) during high market volatility due to "macroeconomic blindness".
Dual Agent (Proposed Architecture): A hierarchical Multi-Agent system. It decouples the trading signal from risk management:
The Trader: Uses a TCN + GRU network to extract local micro-economic trends.
The Risk Manager: An independent MLP network that dynamically scales the Trader's maximum exposure based on global macroeconomic stress (normalized VIX index).
📂 Repository Structure
Single_Agent/ : Contains the baseline model codebase.
data_prep.py: Feature engineering (Log returns, relative volatility).
env.py: Custom Gymnasium environment.
rnn_agent.py: Conv1D + GRU feature extractor.
train.py & test.py: Training and evaluation scripts.
trading_dual_agent/ : Contains the proposed Multi-Agent architecture.
risk_manager.py: VIX-driven Risk Manager environment and controller.
models.py: TCN + GRU architecture avoiding look-ahead bias via causal dilations.
train_rm.py: Independent training script for the Risk Manager.
train.py: Training script for the Trader under RM constraints.
evaluate_dual_...py: Comprehensive backtesting and extreme stress-testing scripts.
🚀 Key Results
The Dual Agent architecture proved to be highly robust and "production-ready", achieving:
80% Win Rate against the Buy & Hold benchmark across a diverse panel of 10 assets.
Exceptional Capital Preservation: Survived the 2008 Subprime and 2020 COVID crashes, reducing the average Maximum Drawdown to just 3.0%.
Massive Long-Term Outperformance: Generated +2090% on AAPL over a 10-year backtest (2015-2025) compared to +796% for the market.
Zero Overfitting: Statistical validation confirmed a minimal 1.7pp gap between In-Sample and Out-of-Sample performance.
🛠️ Requirements & Installation
To run this project, you need the following libraries:
code
Bash
pip install torch numpy pandas yfinance stable-baselines3 gymnasium plotly   modify engineering graduation model
