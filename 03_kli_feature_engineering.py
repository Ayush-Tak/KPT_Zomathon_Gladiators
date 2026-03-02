"""
===============================================================
PILLAR 2 — Kitchen Load Index (KLI) Feature Engineering
===============================================================
Purpose:
  Build a composite Kitchen Load Index per restaurant, updated
  every 5 minutes, using publicly available + Zomato-internal
  proxy signals. This gives the KPT model visibility into total
  kitchen load (dine-in + competitor orders) without IoT hardware.

Signals used:
  1. Google Maps Real-Time Busyness  (public / Places API)
  2. Zomato Table Reservations       (Zomato-internal)
  3. Listing Page View Velocity      (Zomato-internal analytics)
  4. Historical Load Profile         (derived from past order data)
  5. Weather Load Modifier           (OpenWeatherMap / IMD)

Output:
  - kli_features.csv  — KLI per (restaurant_id, timestamp)
  - kli_analysis.png  — visualisation of KLI vs true load
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


N_RESTAURANTS = 20
N_HOURS       = 24 * 7     # one week of 5-min intervals
N_INTERVALS   = N_HOURS * 12

restaurant_ids = [f"R{str(i).zfill(4)}" for i in range(1, N_RESTAURANTS + 1)]
base_dt = datetime(2024, 6, 1, 0, 0, 0)
timestamps = [base_dt + timedelta(minutes=5 * i) for i in range(N_INTERVALS)]

records = []
for rid in restaurant_ids:
    # Each restaurant has a capacity (max orders/hour) and a personality
    capacity      = np.random.randint(8, 30)
    peak_hour_1   = np.random.randint(11, 14)   # lunch
    peak_hour_2   = np.random.randint(18, 21)   # dinner

    for ts in timestamps:
        hour   = ts.hour
        dow    = ts.weekday()  # 0=Mon
        is_wkd = dow >= 5

        # True kitchen load (ground truth — not available in prod, only for validation)
        base_load = 0.2
        lunch_bump  = 0.6 * np.exp(-0.5 * ((hour - peak_hour_1) / 1.5) ** 2)
        dinner_bump = 0.5 * np.exp(-0.5 * ((hour - peak_hour_2) / 1.5) ** 2)
        wkd_bump    = 0.15 if is_wkd else 0
        true_load   = min(1.0, base_load + lunch_bump + dinner_bump + wkd_bump
                          + np.random.normal(0, 0.05))
        true_load   = max(0, true_load)

        # ── Signal 1: Google Maps Busyness (0–100, noisy, cached 10-min) ──
        # Production: call Google Places API populartimes field
        # GET https://maps.googleapis.com/maps/api/place/details/json
        #     ?fields=current_opening_hours&place_id=<id>&key=<API_KEY>
        gmap_noise   = np.random.normal(0, 0.08)
        gmap_busyness = np.clip(true_load * 100 + gmap_noise * 100, 0, 100)

        # ── Signal 2: Zomato Reservations (count in next 30 min) ──
        # Production: SELECT COUNT(*) FROM reservations
        #             WHERE restaurant_id=? AND res_time BETWEEN NOW() AND NOW()+30min
        reservation_load = true_load * capacity * 0.3 * (1 + np.random.normal(0, 0.1))
        reservations_30m = max(0, int(reservation_load))

        # ── Signal 3: Listing Page View Velocity ratio vs 7-day avg ──
        # Production: analytics DB — events where event='restaurant_view'
        #             in last 5 min vs rolling avg
        pvr_base  = true_load * 50 + np.random.normal(0, 5)
        pv_ratio  = np.clip(pvr_base / 25.0, 0, 3)  # ratio vs baseline (1.0 = normal)

        # ── Signal 4: Historical load profile (this restaurant, this hour, this DOW) ──
        # Production: precomputed lookup table from 90-day order history
        hist_load = (base_load + lunch_bump + dinner_bump + wkd_bump)  # no noise
        hist_load = max(0, min(1, hist_load))

        # ── Signal 5: Weather modifier ──
        # Production: GET https://api.openweathermap.org/data/2.5/weather?q=Mumbai&appid=<key>
        # Rain → fewer dine-in, more delivery orders
        is_rain  = np.random.random() < 0.12
        weather_mod = 0.1 if is_rain else 0.0   # rain adds 10% to delivery load

        records.append({
            "restaurant_id":    rid,
            "timestamp":        ts,
            "true_load":        round(true_load, 3),
            "capacity":         capacity,
            "gmap_busyness":    round(gmap_busyness, 1),
            "reservations_30m": reservations_30m,
            "pv_ratio":         round(pv_ratio, 3),
            "hist_load":        round(hist_load, 3),
            "is_rain":          int(is_rain),
            "weather_mod":      weather_mod,
        })

df = pd.DataFrame(records)
print(f"Generated {len(df):,} signal records for {N_RESTAURANTS} restaurants over {N_HOURS} hours\n")

# ================================================================
# STEP 2 — COMPUTE KLI (KITCHEN LOAD INDEX)
# ================================================================
# KLI ∈ [0, 1] — normalised composite kitchen busyness estimate
#
# Weights are learned per merchant archetype in production
# (using ridge regression on historical true load vs signals).
# Here we use reasonable fixed weights for demonstration.

W1 = 0.30   # Google Maps busyness
W2 = 0.25   # Reservations
W3 = 0.20   # Page view velocity
W4 = 0.20   # Historical profile
W5 = 0.05   # Weather modifier

def compute_kli(row):
    s1 = row["gmap_busyness"] / 100.0
    s2 = min(1.0, row["reservations_30m"] / (row["capacity"] * 0.5 + 1e-6))
    s3 = min(1.0, row["pv_ratio"] / 3.0)
    s4 = row["hist_load"]
    s5 = row["weather_mod"]
    return min(1.0, W1*s1 + W2*s2 + W3*s3 + W4*s4 + W5*s5)

df["kli"] = df.apply(compute_kli, axis=1)
df["kli_error"] = abs(df["kli"] - df["true_load"])

# Also compute a naive baseline: just use historical profile alone
df["baseline_error"] = abs(df["hist_load"] - df["true_load"])

# ================================================================
# STEP 3 — STATISTICS
# ================================================================

print("=" * 55)
print("KITCHEN LOAD INDEX (KLI) — RESULTS")
print("=" * 55)
print(f"\nKLI vs True Load:")
print(f"  Pearson correlation:  {df['kli'].corr(df['true_load']):.3f}")
print(f"  MAE (KLI):            {df['kli_error'].mean():.4f}")
print(f"  MAE (hist alone):     {df['baseline_error'].mean():.4f}")
impr = (1 - df['kli_error'].mean() / df['baseline_error'].mean()) * 100
print(f"  Improvement over hist-only:  {impr:.1f}%")

print(f"\nSignal correlation with true load:")
for sig, col in [("Google Maps",     "gmap_busyness"),
                 ("Reservations",    "reservations_30m"),
                 ("Page View Ratio", "pv_ratio"),
                 ("Historical",      "hist_load")]:
    r = df[col].corr(df["true_load"])
    print(f"  {sig:20s} r = {r:.3f}")

# ================================================================
# STEP 4 — VISUALISATIONS
# ================================================================

rid_demo = restaurant_ids[0]
demo     = df[df["restaurant_id"] == rid_demo].copy()
demo_day = demo[demo["timestamp"].dt.date == base_dt.date() + timedelta(days=1)]

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle("Kitchen Load Index (KLI) — Signal Fusion Analysis", fontsize=15, fontweight="bold")

# Plot 1: KLI vs true load over a day
ax = fig.add_subplot(gs[0, :2])
hrs = [(t - demo_day["timestamp"].iloc[0]).total_seconds() / 3600 for t in demo_day["timestamp"]]
ax.fill_between(hrs, demo_day["true_load"], alpha=0.3, color="#1B3A6B", label="True Load")
ax.plot(hrs, demo_day["true_load"], color="#1B3A6B", linewidth=2)
ax.plot(hrs, demo_day["kli"], color="#E8612A", linewidth=2, label="KLI (our estimate)")
ax.plot(hrs, demo_day["hist_load"], color="#2E86AB", linewidth=1.5, linestyle="--", alpha=0.7, label="Historical only (baseline)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Normalised Kitchen Load")
ax.set_title(f"Restaurant {rid_demo} — KLI vs True Load (1 Day)", fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(0, 24)

# Plot 2: KLI error distribution
ax = fig.add_subplot(gs[0, 2])
ax.hist(df["baseline_error"], bins=50, alpha=0.6, color="#2E86AB", density=True, label=f"Hist-only  (MAE={df['baseline_error'].mean():.3f})")
ax.hist(df["kli_error"],      bins=50, alpha=0.6, color="#1A7A4A", density=True, label=f"KLI fused  (MAE={df['kli_error'].mean():.3f})")
ax.set_xlabel("Absolute Error vs True Load")
ax.set_ylabel("Density")
ax.set_title("Load Estimation Error", fontweight="bold")
ax.legend(fontsize=8)

# Plot 3: Scatter KLI vs true load
ax = fig.add_subplot(gs[1, 0])
sample = df.sample(1500, random_state=42)
ax.scatter(sample["true_load"], sample["kli"], alpha=0.3, s=15, color="#E8612A")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("True Kitchen Load")
ax.set_ylabel("KLI Estimate")
ax.set_title("KLI Scatter vs True Load", fontweight="bold")
r = df["kli"].corr(df["true_load"])
ax.text(0.05, 0.9, f"r = {r:.3f}", transform=ax.transAxes, fontsize=11, color="#1B3A6B", fontweight="bold")

# Plot 4: Signal weights contribution
ax = fig.add_subplot(gs[1, 1])
signals = ["Google Maps\nBusyness", "Reservations\n(30 min)", "Page View\nVelocity", "Historical\nProfile", "Weather\nModifier"]
weights = [W1, W2, W3, W4, W5]
colors  = ["#E8612A", "#2E86AB", "#1B3A6B", "#1A7A4A", "#9B59B6"]
bars = ax.barh(signals, weights, color=colors, edgecolor="white", linewidth=1.5)
ax.set_xlabel("Weight in KLI Formula")
ax.set_title("Signal Weights in KLI", fontweight="bold")
for bar, w in zip(bars, weights):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{w:.0%}", va="center", fontsize=10, fontweight="bold")

# Plot 5: Avg KLI by hour of day (aggregate pattern)
ax = fig.add_subplot(gs[1, 2])
df["hour"] = df["timestamp"].dt.hour
hourly = df.groupby("hour").agg(mean_kli=("kli","mean"), mean_true=("true_load","mean")).reset_index()
ax.fill_between(hourly["hour"], hourly["mean_true"], alpha=0.25, color="#1B3A6B")
ax.plot(hourly["hour"], hourly["mean_true"], "o-", color="#1B3A6B", linewidth=2, markersize=4, label="True Load")
ax.plot(hourly["hour"], hourly["mean_kli"],  "s--", color="#E8612A", linewidth=2, markersize=4, label="KLI")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Normalised Load")
ax.set_title("Average Load Pattern (All Restaurants)", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xticks(range(0, 24, 3))

plt.savefig("kli_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: kli_analysis.png")

# ================================================================
# STEP 5 — EXPORT
# ================================================================

out_cols = ["restaurant_id", "timestamp", "kli", "gmap_busyness",
            "reservations_30m", "pv_ratio", "hist_load", "is_rain"]
df[out_cols].to_csv("kli_features.csv", index=False)
print("\nExported: kli_features.csv  (KLI feature ready for KPT model)")
print("\nDone.")
