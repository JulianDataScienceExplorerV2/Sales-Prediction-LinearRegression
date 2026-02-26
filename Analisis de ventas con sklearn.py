"""
Sales Prediction with Linear Regression
========================================
Dataset : Advertising (TV, Radio, Newspaper spend vs Sales)
Model   : Ordinary Least Squares Linear Regression
Metrics : R-squared, MAE, RMSE
Author  : JulianDataScienceExplorerV2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv"
df = pd.read_csv(URL)

print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head())
print("\nDescriptive statistics:")
print(df.describe().round(2))

# ---------------------------------------------------------------------------
# 2. Feature / target split
# ---------------------------------------------------------------------------
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET   = "Sales"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 3. Train model
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 40)
print("MODEL PERFORMANCE")
print("=" * 40)
print(f"  R-squared (R2) : {r2:.4f}")
print(f"  MAE            : {mae:.4f}")
print(f"  RMSE           : {rmse:.4f}")
print("=" * 40)

print("\nFeature Coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:<12}: {coef:.4f}")
print(f"  {'Intercept':<12}: {model.intercept_:.4f}")

# ---------------------------------------------------------------------------
# 5. Diagnostic plots
# ---------------------------------------------------------------------------
residuals = y_test - y_pred

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Sales Prediction — Linear Regression Diagnostics", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# 5a. Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred, alpha=0.7, edgecolors="steelblue", facecolors="lightblue", linewidths=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1.5, label="Perfect fit")
ax1.set_xlabel("Actual Sales")
ax1.set_ylabel("Predicted Sales")
ax1.set_title("Actual vs Predicted")
ax1.legend(fontsize=9)
ax1.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# 5b. Residuals vs Predicted
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred, residuals, alpha=0.7, edgecolors="coral", facecolors="lightsalmon", linewidths=0.5)
ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax2.set_xlabel("Predicted Sales")
ax2.set_ylabel("Residuals")
ax2.set_title("Residuals vs Predicted")

# 5c. Residuals distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
ax3.axvline(0, color="black", linewidth=1.2, linestyle="--")
ax3.set_xlabel("Residual Value")
ax3.set_ylabel("Frequency")
ax3.set_title("Residuals Distribution")

# 5d. Feature coefficients
ax4 = fig.add_subplot(gs[1, 1])
colors = ["steelblue" if c >= 0 else "coral" for c in model.coef_]
ax4.barh(FEATURES, model.coef_, color=colors, edgecolor="white")
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_xlabel("Coefficient Value")
ax4.set_title("Feature Coefficients")

plt.savefig("model_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as model_diagnostics.png")