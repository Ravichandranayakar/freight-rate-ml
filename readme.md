# US Freight Rate Prediction Engine

An end-to-end machine learning system for predicting commercial spot freight shipping rates across US logistics lanes. The project implements robust data preprocessing, logistics feature engineering, chronological validation, and an optimized LightGBM regressor to accurately estimate posted shipment rates.

---

## 1. Project Overview

Predicting dynamic spot rates in commercial freight is critical for freight brokers, carriers, and shippers. Freight rates are influenced by trip distance, cargo weight, equipment requirements, market density, and regional supply/demand imbalances.

This project delivers a complete machine learning pipeline:
- **Data Quality & Hygiene:** Handling anomalous cargo weights and imputing missing market indices.
- **Logistics Feature Engineering:** Synthesizing cost proxies (`est_cost`), market pressure ratios, equipment encodings, and temporal signals.
- **Leakage-Free Evaluation:** Enforcing a chronological time-series holdout split to emulate real-world production forecasting.
- **Model Optimization:** Tuning LightGBM with an L1 regression loss to minimize Mean Absolute Error (MAE).

---

## 2. Dataset & Preprocessing

The historical dataset contains 48,000 commercial shipment records covering January through October 2025 across key US shipping corridors.

### Data Cleaning Highlights:
- **Cargo Weight Anomalies:** Resolved negative cargo weight values by applying domain-based magnitude corrections.
- **Missing Value Imputation:** Imputed missing values for cargo weights and regional market indices using median values derived strictly from the training partition to prevent lookahead data leakage.
- **Geospatial Features:** Leveraged precise origin/destination coordinates to capture spatial variance rather than sparse, noisy city labels.

---

## 3. Methodology & Architecture

1. **Exploratory Data Analysis (EDA):** Analyzed feature correlations, rate distributions, seasonal trends, and distance-to-rate linearity.
2. **Feature Engineering:**
   - `est_cost`: Distance and weight cost proxy.
   - `market_pressure`: Interaction between market indices and shipment volume.
   - `is_short_haul`: Binary indicator for local trips under 150 miles.
   - Temporal attributes: Extracted month, day of week, and quarterly indicators.
3. **Validation Strategy:** Chronological split (Jan–Aug 2025 for training, Sep–Oct 2025 for validation) ensuring the model is evaluated on future unseen market conditions.
4. **Model Architecture:** Gradient Boosted Decision Trees (LightGBM) optimized for speed and tabular accuracy.

---

## 4. Benchmark & Results

Evaluating models on the internal chronological holdout partition (9,523 shipments):

| Model | Objective / Loss | Mean Absolute Error (MAE) | R² Score |
|---|---|---:|---:|
| Baseline LightGBM | L2 (MSE) | $149.99 | 0.8207 |
| **Tuned LightGBM** | **L1 (`regression_l1`)** | **$114.36** | **0.8264** |

> **Key Finding:** Switching from L2 loss to L1 loss and tuning leaf parameters reduced prediction error by **23.8%**, bringing the average prediction error down to **$114.36**.

---

## 5. Repository Structure

```text
├── data/
│   └── december-chart-inputs.csv   # Route forecasting inputs
├── notebooks/
│   └── freight_rate_prediction.ipynb # Complete EDA, feature engineering & model training
├── requirements.txt                # Project dependencies
├── validation_predictions.csv      # Out-of-sample predictions
└── README.md                       # Project documentation
