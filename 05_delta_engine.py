"""
===============================================================
PILLAR 4 — Real-Time KPT Delta Adjustment Engine
===============================================================
Purpose:
  Train a lightweight LightGBM model that takes the base KPT
  prediction and produces a real-time correction (Δ) using
  live signals from Pillars 1–3.

  Adjusted_KPT = Base_KPT + Δ

  The Delta Engine runs every 60 seconds per active order and
  is retrained weekly on fresh data.

Input features:
  - base_kpt_prediction       (from existing model, fixed)
  - for_credibility_score      (Pillar 1)
  - merchant_archetype_code    (Pillar 1)
  - kli                        (Pillar 2 — kitchen load index)
  - current_rider_wait_others  (live: other riders waiting here)
  - order_complexity_score     (item count × avg prep units)
  - time_since_accept          (minutes elapsed since order placed)
  - hour_of_day                (cyclical encoded)
  - day_of_week                (cyclical encoded)

Output: delta_minutes (positive = push KPT estimate up)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    LGB_AVAILABLE = False
    print("LightGBM not installed — using sklearn GradientBoosting instead")
    print("Install with: pip install lightgbm\n")

np.random.seed(42)

# ================================================================
# STEP 1 — GENERATE TRAINING DATA
# ================================================================
# In production: join orders table + FOR credibility scores +
#   KLI features + rider GPS wait times.
# Here: synthetic data with realistic correlations.

N = 15000
print(f"Generating {N:,} training samples...\n")

# True KPT (what we want to predict accurately)
true_kpt = np.random.gamma(shape=4.0, scale=3.0, size=N) + 3  # mean ~15 min

# Base model prediction (existing, noisy — underestimates)
base_noise = np.random.normal(loc=2.8, scale=2.2, size=N)
base_kpt   = np.clip(true_kpt - base_noise, 3, 40)

# FOR credibility score (0–1; low = suspicious merchant)
for_cred = np.random.beta(2, 1.5, N)

# Merchant archetype (0=Proactive, 1=Erratic, 2=RiderTriggered, 3=NonCompiler)
archetype = np.random.choice([0, 1, 2, 3], N, p=[0.15, 0.30, 0.40, 0.15])

# KLI (0–1; higher = busier kitchen)
kli = np.clip(np.random.beta(2, 3, N), 0, 1)

# Current rider wait (min) — other riders already waiting at this restaurant
rider_wait_others = np.random.exponential(2.0, N) * (kli > 0.6).astype(float)

# Order complexity (1–10)
complexity = np.random.choice(range(1, 11), N, p=[0.05,0.10,0.15,0.20,0.18,0.12,0.08,0.06,0.04,0.02])

# Time since order accepted (0 to predicted KPT minutes)
time_since_accept = np.random.uniform(0, base_kpt)

# Hour and day (cyclical features)
hour = np.random.randint(9, 23, N)
dow  = np.random.randint(0, 7, N)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin  = np.sin(2 * np.pi * dow / 7)
dow_cos  = np.cos(2 * np.pi * dow / 7)

# ── True Delta (what the model learns to predict) ──
# Delta = True_KPT − Base_KPT
# This is shaped by real patterns:
#   - Low FOR credibility → base_kpt was underestimated more
#   - High KLI → kitchen is busier, actual slower
#   - High rider wait at restaurant → kitchen is backed up
#   - High complexity → base model tends to underestimate

delta_true = (
    (1 - for_cred) * 2.5                         # low credibility → push up
    + kli * 3.0                                   # busy kitchen → push up
    + rider_wait_others * 0.8                     # others waiting → kitchen behind
    + (complexity > 6).astype(float) * 1.5        # complex orders → push up
    + (archetype == 2).astype(float) * 1.2        # rider-triggered archetype
    + np.random.normal(0, 1.0, N)                 # residual noise
)

# ================================================================
# STEP 2 — BUILD FEATURE MATRIX
# ================================================================

feature_names = [
    "base_kpt", "for_credibility", "archetype_code",
    "kli", "rider_wait_others", "complexity",
    "time_since_accept", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

X = np.column_stack([
    base_kpt, for_cred, archetype, kli, rider_wait_others,
    complexity, time_since_accept, hour_sin, hour_cos, dow_sin, dow_cos
])
y = delta_true

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================================================
# STEP 3 — TRAIN DELTA MODEL
# ================================================================

print("Training Delta Engine model...")

if LGB_AVAILABLE:
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05,
        max_depth=6, num_leaves=31,
        min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1
    )
else:
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, random_state=42
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ================================================================
# STEP 4 — EVALUATE
# ================================================================

# Without Delta Engine
kpt_pred_base     = X_test[:, 0]               # base_kpt column
kpt_pred_adjusted = kpt_pred_base + y_pred

true_kpt_test = kpt_pred_base + y_test         # true = base + actual delta

mae_base = mean_absolute_error(true_kpt_test, kpt_pred_base)
mae_adj  = mean_absolute_error(true_kpt_test, kpt_pred_adjusted)
rmse_adj = mean_squared_error(true_kpt_test, kpt_pred_adjusted)**0.5

print(f"\n{'='*50}")
print("  DELTA ENGINE — RESULTS")
print(f"{'='*50}")
print(f"  MAE without Delta Engine:  {mae_base:.2f} min")
print(f"  MAE with    Delta Engine:  {mae_adj:.2f} min")
print(f"  RMSE:                      {rmse_adj:.2f} min")
print(f"  Improvement:               {(1-mae_adj/mae_base)*100:.1f}%")

# Feature importance
if LGB_AVAILABLE:
    importances = model.feature_importances_
else:
    importances = model.feature_importances_

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df  = imp_df.sort_values("importance", ascending=False)
print(f"\n  Top feature importances:")
for _, row in imp_df.head(6).iterrows():
    bar = "█" * int(row["importance"] / importances.max() * 20)
    print(f"    {row['feature']:22s}  {bar}  {row['importance']:.0f}")

# ================================================================
# STEP 5 — INFERENCE EXAMPLE (LIVE ORDER)
# ================================================================

print(f"\n{'='*50}")
print("  LIVE INFERENCE EXAMPLE")
print(f"{'='*50}")

live_order = np.array([[
    14.0,   # base_kpt from existing model
    0.35,   # for_credibility (low — rider-triggered restaurant)
    2,      # archetype: RiderTriggered
    0.78,   # kli: kitchen is very busy right now
    3.2,    # rider_wait_others: 3.2 min wait already observed
    7,      # complexity: 7/10
    4.5,    # time_since_accept: 4.5 min into prep
    np.sin(2*np.pi*13/24), np.cos(2*np.pi*13/24),  # 1pm
    np.sin(2*np.pi*5/7),   np.cos(2*np.pi*5/7),    # Saturday
]])

delta_predicted = model.predict(live_order)[0]
adjusted_kpt    = live_order[0][0] + delta_predicted

print(f"\n  Input:  base_kpt = {live_order[0][0]:.1f} min  |  KLI = {live_order[0][3]:.2f}  |  rider_wait_others = {live_order[0][4]:.1f} min")
print(f"  Predicted Δ:         +{delta_predicted:.2f} min")
print(f"  Adjusted KPT:         {adjusted_kpt:.2f} min  ← rider dispatched based on this")
print(f"  (Without Delta:        {live_order[0][0]:.1f} min  ← would have dispatched too early)")

# ================================================================
# STEP 6 — VISUALISATIONS
# ================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Delta Engine — Training & Evaluation", fontsize=14, fontweight="bold")

# Plot 1: Feature importance
ax = axes[0]
colors = ["#1B3A6B" if i < 3 else "#2E86AB" if i < 6 else "#B0BEC5" for i in range(len(imp_df))]
ax.barh(imp_df["feature"][:8][::-1], imp_df["importance"][:8][::-1],
        color=colors[:8][::-1], edgecolor="white")
ax.set_xlabel("Feature Importance")
ax.set_title("Top Feature Importances", fontweight="bold")

# Plot 2: Predicted vs actual delta
ax = axes[1]
sample = np.random.choice(len(y_test), 800, replace=False)
ax.scatter(y_test[sample], y_pred[sample], alpha=0.3, s=12, color="#E8612A")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "k--", linewidth=1.5)
ax.set_xlabel("Actual Δ (minutes)")
ax.set_ylabel("Predicted Δ (minutes)")
r = np.corrcoef(y_test, y_pred)[0, 1]
ax.set_title(f"Predicted vs Actual Δ  (r={r:.3f})", fontweight="bold")

# Plot 3: KPT MAE comparison
ax = axes[2]
labels  = ["Base KPT\n(no Delta Engine)", "Adjusted KPT\n(with Delta Engine)"]
maes    = [mae_base, mae_adj]
colors2 = ["#E8612A", "#1A7A4A"]
bars    = ax.bar(labels, maes, color=colors2, width=0.45, edgecolor="white", linewidth=2)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f} min", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("MAE (minutes)")
ax.set_title("KPT Prediction MAE", fontweight="bold")
imp_pct = (1 - mae_adj/mae_base)*100
ax.annotate(f"↓{imp_pct:.0f}% improvement", xy=(1, mae_adj),
            xytext=(0.5, mae_base*0.6), ha="center", fontsize=11,
            color="#1A7A4A", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1A7A4A", lw=2))

plt.tight_layout()
plt.savefig("delta_engine_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: delta_engine_results.png")
print("Done.")
