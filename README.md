# Sales Prediction — Linear Regression

Predicting product sales from advertising spend (TV, Radio, Newspaper) using Ordinary Least Squares Linear Regression. The project covers the full pipeline: data exploration, model training, metric evaluation, and diagnostic visualizations.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-informational?style=flat-square)

---

## Problem Statement

A company runs advertising campaigns across three channels — TV, Radio, and Newspaper — and wants to understand how each channel influences sales. This model quantifies those relationships and produces sales predictions given a budget allocation.

---

## Dataset

**Source:** [Advertising dataset](https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv)

| Column | Description |
|--------|-------------|
| `TV` | Budget spent on TV advertising (thousands USD) |
| `Radio` | Budget spent on Radio advertising (thousands USD) |
| `Newspaper` | Budget spent on Newspaper advertising (thousands USD) |
| `Sales` | Units sold (thousands) — target variable |

---

## Model Performance

| Metric | Value |
|--------|-------|
| R-squared (R²) | ~0.90 |
| Mean Absolute Error (MAE) | ~1.05 |
| Root Mean Squared Error (RMSE) | ~1.40 |

The model explains approximately **90% of the variance** in sales, indicating a strong linear relationship between advertising spend and sales volume.

---

## Project Structure

```
Sales-Prediction-LinearRegression/
└── Analisis de ventas con sklearn.py   # Full analysis script
```

---

## Analysis Pipeline

1. **Data loading** — read CSV from remote URL
2. **Exploratory summary** — shape, head, descriptive statistics
3. **Train/test split** — 80/20 with fixed random state
4. **Model training** — `sklearn.linear_model.LinearRegression`
5. **Evaluation** — R², MAE, RMSE
6. **Coefficient analysis** — feature importance by coefficient magnitude
7. **Diagnostic plots** — Actual vs Predicted, Residuals, Residual Distribution, Coefficient Bar Chart

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and exploration |
| `scikit-learn` | Model training and evaluation |
| `numpy` | Numerical computations |
| `matplotlib` | Diagnostic visualizations |

---

## How to Run

```bash
git clone https://github.com/JulianDataScienceExplorerV2/Sales-Prediction-LinearRegression.git
cd Sales-Prediction-LinearRegression

pip install pandas scikit-learn numpy matplotlib

python "Analisis de ventas con sklearn.py"
```

---

## Key Insight

TV advertising has the strongest positive coefficient, making it the most impactful channel for driving sales. Newspaper spend shows minimal marginal contribution, suggesting budget reallocation toward TV and Radio would improve ROI.

---

## Author

**Julian David Urrego** — Data Analyst
Python · Scikit-learn · Pandas · Statistical Modeling

[![GitHub](https://img.shields.io/badge/GitHub-JulianDataScienceExplorerV2-181717?style=flat-square&logo=github)](https://github.com/JulianDataScienceExplorerV2)