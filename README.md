# Regime-Aware Forecasting of EUR 10Y Swap Spreads (HMM + LSTM)

This project implements a regime-sensitive framework for forecasting the EUR 10-year swap spread and its volatility using a hybrid Hidden Markov Model (HMM) and LSTM approach.

## Problem Definition

The EUR 10Y swap spread is defined as:

10Y EUR Interest Rate Swap – 10Y German Bund Yield

The spread reflects credit risk, liquidity conditions, and macro-financial stress.  
Its dynamics are highly non-stationary and exhibit structural regime changes, making them difficult to model with traditional linear time-series approaches.

## Objectives

This project studies two related forecasting tasks:

1. **Swap Spread Level Forecasting**
2. **Swap Spread Volatility Forecasting**

## Methodology

The proposed framework combines regime detection with nonlinear forecasting:

- **Hidden Markov Model (HMM)** → identifies latent macro-financial regimes  
- **LSTM Neural Network** → captures nonlinear temporal structure within regimes  
- **Macro-financial variables** → provide economic context and improve predictive stability

## Benchmarks

The regime-aware model is compared against standard approaches:

- **AR model**
- **ARX model (macro-augmented autoregression)**
- **Vanilla LSTM (without regime structure)**

This allows evaluation of whether explicitly modeling regime shifts improves predictive performance.

## Data

The original research used institutional-grade data sourced from Bloomberg, which cannot be redistributed.

This repository therefore includes a small illustrative dataset located in 'data_sample/sample.csv'.


The sample preserves the structure of the original dataset while allowing the pipeline to run without proprietary data.

The code is fully **data-agnostic** and can be applied to any dataset with the same schema.

## Repository Structure

```text
src/
    preprocessing.py
    hmm_lstm.py
    benchmarks.py
    volatility_forecast.py
    evaluate.py
    trade_strategy.py
    run_demo.py

data_sample/
    sample.csv
```

## Running the Demo

```bash
pip install -r requirements.txt
```
Run the reproducible example:
```bash
python src/run_demo.py
```
## Context

This repository is derived from a Master's thesis in Quantitative Finance focused on regime-aware modeling of EUR swap spread dynamics under structural market change.

The purpose of this repository is to present the modeling framework in a clean, reproducible format suitable for integration with proprietary datasets.

