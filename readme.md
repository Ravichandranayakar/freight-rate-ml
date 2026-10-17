# US Freight Rate Prediction Engine

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/RavichandraNayakar/Freight-Rate-Prediction)
[![Model](https://img.shields.io/badge/Model-LightGBM%20(L1%20Loss)-teal)](https://github.com/Ravichandranayakar/freight-rate-ml)
[![Holdout MAE](https://img.shields.io/badge/MAE-$114.36-brightgreen)](https://github.com/Ravichandranayakar/freight-rate-ml)

> **Live Interactive Application:** [https://huggingface.co/spaces/RavichandraNayakar/Freight-Rate-Prediction](https://huggingface.co/spaces/RavichandraNayakar/Freight-Rate-Prediction)

An end-to-end machine learning system for predicting commercial spot freight shipping rates across US logistics lanes. The project focuses on data hygiene, domain-driven feature engineering, chronological validation, and LightGBM regression to accurately estimate posted shipment rates for unseen loads.


---

## 1. Project Overview

Predicting dynamic spot rates in commercial freight is critical for freight brokers, carriers, and shippers. Spot rates fluctuate significantly based on transit distance, cargo weight, equipment type, regional market demand, and supply imbalances.

The objective of this project is to predict `posted_rate` (freight rate in USD) using shipment attributes and market signals.

The solution implements a production-grade machine learning workflow:
- **Exploratory Data Analysis (EDA):** Identified core price drivers and distributional anomalies.
- **Data Preprocessing & Cleaning:** Rectified weight anomalies and handled missing market indicators without data leakage.
- **Feature Engineering:** Synthesized cost proxies, market pressure ratios, short-haul flags, and calendar features.
- **Chronological Validation:** Implemented time-based train/validation splitting to emulate live forward-looking deployment.
- **Model Training & Tuning:** Optimized LightGBM regressor with L1 loss, achieving superior MAE performance.

---

## 2. Dataset & Preprocessing

The historical dataset contains 48,000 commercial shipment records spanning January through October 2025 across diverse US shipping corridors. The target variable is `posted_rate`.

### Data Quality & Preprocessing Decisions:
- **Cargo Weight Anomalies:** Identified negative cargo weight values in the source records. Absolute values were applied as a domain-grounded correction.
- **Missing Value Imputation:** Missing values in cargo weight and market index were imputed using median values computed strictly from the training partition to avoid lookahead leakage.
- **Geospatial Features:** Retained origin and destination coordinates (latitude/longitude) rather than raw text city names to capture continuous spatial variance and lane directionality.

---

## 3. Methodology & Architecture

The workflow follows a disciplined 5-step ML development lifecycle:

1. **Exploratory Data Analysis:** Evaluated feature correlations, missingness patterns, skewness, and the strong linear/non-linear relationship between distance, cargo weight, and rate.
2. **Data Cleaning:** Imputed missing values and normalized numerical columns.
3. **Feature Engineering:**
   - `est_cost`: Distance and cargo weight cost proxy.
   - `market_pressure`: Ratio capturing regional market index vs. volume demand.
   - `is_short_haul`: Binary indicator flag for local runs under 150 miles.
   - Temporal features: Extracted month, day of week, and seasonality indicators.
4. **Categorical Encodings:** Encoded equipment types (e.g., Dry Van, Reefer, Flatbed) for tree-based split optimization.
5. **Time-Series Validation:** Structured a chronological holdout split ensuring historical shipments predict future shipments.

---

## 4. Benchmark & Validation Results

LightGBM was selected for its high execution speed, efficiency with tabular data, and native handling of non-linear interactions.

The dataset was split chronologically:
| Partition | Period | Records |
|---|---|---:|
| Training Set | January – August 2025 | 38,477 |
| Validation Holdout | September – October 2025 | 9,523 |

### Model Performance Comparison:

| Model | Objective / Loss Function | MAE | R² Score |
|---|---|---:|---:|
| Baseline LightGBM | L2 (MSE) | $149.99 | 0.8207 |
| **Tuned LightGBM** | **L1 (`regression_l1`)** | **$114.36** | **0.8264** |

> **Key Takeaway:** Switching from L2 loss to L1 regression loss aligned the optimization objective directly with the business metric (Mean Absolute Error), reducing prediction error by **23.8%** and achieving a final validation MAE of **$114.36**.

---

## 5. Repository Structure

```text
├── data/
│   ├── december-chart-inputs.csv       # Route forecasting inputs
│   └── train-test.csv                  # Historical shipment data (local)
├── notebooks/
│   └── freight_rate_prediction.ipynb   # Complete analysis, feature engineering & model training
├── scorer_results/
│   ├── candidate_december.png          # December rate forecast visualization
│   └── feature_importance.png          # Top predictive feature rankings
├── requirements.txt                    # Python runtime dependencies
├── validation_predictions.csv          # Out-of-sample holdout predictions
└── README.md                           # Project documentation
