"""
===============================================================
PILLAR 1 — FOR Credibility Scoring Engine
===============================================================
Purpose:
  Compute a credibility score for every Food Order Ready (FOR)
  event by comparing FOR timestamp with rider GPS arrival time.
  Produce per-event weights and per-merchant compliance scores
  to clean the KPT model's training data.

Inputs (simulated here, replace with real DB queries):
  - orders: order_id, restaurant_id, accept_time, for_time, pickup_time
  - rider_arrivals: order_id, rider_arrival_time (GPS geofence hit)

Outputs:
  - orders_scored.csv  — events with credibility weights
  - merchant_mcs.csv   — Merchant Compliance Score per restaurant
  - analysis plots     — visualising the FOR delta distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────
np.random.seed(42)
random.seed(42)


N_ORDERS      = 5000
N_RESTAURANTS = 80

restaurant_ids = [f"R{str(i).zfill(4)}" for i in range(1, N_RESTAURANTS + 1)]

# Each restaurant is secretly one of 4 archetypes
archetype_map = {}
archetypes    = ["proactive", "rider_triggered", "erratic", "non_compiler"]
weights       = [0.15, 0.40, 0.30, 0.15]
for rid in restaurant_ids:
    archetype_map[rid] = np.random.choice(archetypes, p=weights)

def generate_order(order_id, restaurant_ids, archetype_map):
    rid       = random.choice(restaurant_ids)
    archetype = archetype_map[rid]

    base_time    = datetime(2024, 1, 1, 10, 0, 0) + timedelta(minutes=random.randint(0, 43200))
    true_kpt_min = max(5, np.random.gamma(shape=3.5, scale=3.5))  # realistic KPT distribution

    accept_time = base_time
    true_ready  = accept_time + timedelta(minutes=true_kpt_min)

    # Rider arrives around true ready time ± noise
    rider_offset_sec = np.random.normal(loc=60, scale=120)  # avg 1 min after ready
    rider_arrival    = true_ready + timedelta(seconds=rider_offset_sec)

    # FOR marking behaviour depends on archetype
    if archetype == "proactive":
        # Marks 2–8 min BEFORE rider arrives → genuine
        for_offset = -1 * abs(np.random.normal(loc=300, scale=120))
        for_time   = rider_arrival + timedelta(seconds=for_offset)
        marked_for = True

    elif archetype == "rider_triggered":
        # Marks within 0–90s of rider arriving → suspicious
        for_offset = abs(np.random.normal(loc=30, scale=25))
        for_time   = rider_arrival + timedelta(seconds=for_offset)
        marked_for = True

    elif archetype == "erratic":
        # Random: sometimes genuine, sometimes triggered
        if random.random() < 0.5:
            for_offset = -1 * abs(np.random.normal(loc=180, scale=200))
        else:
            for_offset = abs(np.random.normal(loc=60, scale=80))
        for_time   = rider_arrival + timedelta(seconds=for_offset)
        marked_for = True

    else:  # non_compiler
        # Only marks 20% of orders
        marked_for = random.random() < 0.20
        for_offset = np.random.normal(loc=0, scale=150)
        for_time   = rider_arrival + timedelta(seconds=for_offset) if marked_for else None

    pickup_time = rider_arrival + timedelta(seconds=abs(np.random.normal(loc=90, scale=45)))

    return {
        "order_id":       order_id,
        "restaurant_id":  rid,
        "archetype":      archetype,
        "accept_time":    accept_time,
        "true_kpt_min":   round(true_kpt_min, 2),
        "rider_arrival":  rider_arrival,
        "for_time":       for_time,
        "marked_for":     marked_for,
        "pickup_time":    pickup_time,
    }

print("Generating synthetic dataset...")
orders = pd.DataFrame([
    generate_order(f"ORD{str(i).zfill(6)}", restaurant_ids, archetype_map)
    for i in range(N_ORDERS)
])
print(f"  Generated {len(orders)} orders across {N_RESTAURANTS} restaurants\n")

# ================================================================
# STEP 2 — COMPUTE FOR DELTA & CREDIBILITY SCORE
# ================================================================

# Only work with orders that have FOR marked
scored = orders[orders["marked_for"] == True].copy()

# Delta = FOR_time − Rider_arrival_time (in seconds)
scored["delta_sec"] = (
    scored["for_time"] - scored["rider_arrival"]
).dt.total_seconds()

def credibility_weight(delta_sec):
    """
    Rules:
      delta < -60s  → FOR marked well before rider  → Genuine    → weight 1.0
      0 ≤ delta ≤ 90s → marked within 90s of rider → Suspicious  → weight 0.4
      delta > 90s   → rider waited, then FOR marked → Very suspicious → weight 0.15
      delta < -60s but extremely early (> 20 min early) → also slightly lower trust
    """
    if delta_sec < -60:
        return 1.0
    elif delta_sec <= 90:
        return 0.4
    else:
        return 0.15

scored["credibility_weight"] = scored["delta_sec"].apply(credibility_weight)

# Classify event type for analysis
def classify_event(delta_sec):
    if delta_sec < -60:
        return "Genuine"
    elif delta_sec <= 90:
        return "Rider-Triggered"
    else:
        return "Delayed/Suspicious"

scored["event_type"] = scored["delta_sec"].apply(classify_event)

# ================================================================
# STEP 3 — RETROACTIVE TRUE KPT LABEL RECONSTRUCTION
# ================================================================
# For suspicious/delayed events, reconstruct true KPT using pickup time
# True KPT = pickup_time − accept_time − handshake_time(90s)

HANDSHAKE_TIME_SEC = 90

scored["reconstructed_kpt_min"] = (
    (scored["pickup_time"] - scored["accept_time"]).dt.total_seconds()
    - HANDSHAKE_TIME_SEC
) / 60.0

scored["raw_for_kpt_min"] = (
    (scored["for_time"] - scored["accept_time"]).dt.total_seconds()
) / 60.0

# Final label: use reconstructed KPT for low-credibility events
scored["final_kpt_label"] = np.where(
    scored["credibility_weight"] >= 0.8,
    scored["raw_for_kpt_min"],
    scored["reconstructed_kpt_min"]
)

scored["label_error_raw"]   = abs(scored["raw_for_kpt_min"]   - scored["true_kpt_min"])
scored["label_error_fixed"] = abs(scored["final_kpt_label"] - scored["true_kpt_min"])

# ================================================================
# STEP 4 — MERCHANT COMPLIANCE SCORE (MCS)
# ================================================================

mcs = scored.groupby("restaurant_id").agg(
    total_for_events     = ("credibility_weight", "count"),
    mean_credibility     = ("credibility_weight", "mean"),
    pct_genuine          = ("event_type",         lambda x: (x == "Genuine").mean()),
    pct_rider_triggered  = ("event_type",         lambda x: (x == "Rider-Triggered").mean()),
    mean_delta_sec       = ("delta_sec",           "mean"),
).reset_index()

mcs["archetype_true"] = mcs["restaurant_id"].map(archetype_map)
mcs["merchant_tier"]  = pd.cut(
    mcs["mean_credibility"],
    bins=[0, 0.35, 0.60, 0.85, 1.01],
    labels=["Non-Compliant", "Low Trust", "Medium Trust", "High Trust"]
)

# ================================================================
# STEP 5 — PRINT SUMMARY STATISTICS
# ================================================================

print("=" * 60)
print("FOR CREDIBILITY ANALYSIS — SUMMARY")
print("=" * 60)

print(f"\nTotal orders with FOR marked: {len(scored)}")
print(f"Orders WITHOUT FOR (non-compliers): {(~orders['marked_for']).sum()}\n")

print("Event type breakdown:")
print(scored["event_type"].value_counts().to_string())

print(f"\nLabel error BEFORE correction (MAE): {scored['label_error_raw'].mean():.2f} min")
print(f"Label error AFTER  correction (MAE): {scored['label_error_fixed'].mean():.2f} min")
improvement = (1 - scored['label_error_fixed'].mean() / scored['label_error_raw'].mean()) * 100
print(f"Label accuracy improvement: {improvement:.1f}%")

print("\nMerchant Compliance Score distribution:")
print(mcs["merchant_tier"].value_counts().to_string())

print("\nArchetype vs detected compliance (validation):")
print(mcs.groupby("archetype_true")["mean_credibility"].mean().round(3).to_string())

# ================================================================
# STEP 6 — VISUALISATIONS
# ================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("FOR Credibility Scoring — Analysis Results", fontsize=16, fontweight="bold", y=0.98)

# Plot 1: Delta distribution
ax = axes[0, 0]
colors_map = {"Genuine": "#1A7A4A", "Rider-Triggered": "#E8612A", "Delayed/Suspicious": "#B91C1C"}
for etype, color in colors_map.items():
    subset = scored[scored["event_type"] == etype]["delta_sec"] / 60
    ax.hist(subset, bins=40, alpha=0.65, color=color, label=etype, density=True)
ax.axvline(x=-1, color="black", linestyle="--", linewidth=1.5, label="Threshold: −60s")
ax.axvline(x=1.5, color="grey", linestyle="--", linewidth=1.5, label="Threshold: +90s")
ax.set_xlabel("FOR Delta (minutes from rider arrival)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("FOR Delta Distribution by Event Type", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.set_xlim(-15, 15)

# Plot 2: Event type pie
ax = axes[0, 1]
counts = scored["event_type"].value_counts()
ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
       colors=["#1A7A4A", "#E8612A", "#B91C1C"], startangle=90,
       textprops={"fontsize": 9})
ax.set_title("Event Type Breakdown", fontsize=11, fontweight="bold")

# Plot 3: Label error before vs after correction
ax = axes[0, 2]
ax.hist(scored["label_error_raw"], bins=50, alpha=0.6, color="#E8612A", label="Before (Raw FOR)", density=True)
ax.hist(scored["label_error_fixed"], bins=50, alpha=0.6, color="#1A7A4A", label="After (Corrected)", density=True)
ax.set_xlabel("Absolute Label Error (minutes)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("KPT Label Error: Before vs After Correction", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.axvline(scored["label_error_raw"].mean(), color="#E8612A", linestyle="--", linewidth=2)
ax.axvline(scored["label_error_fixed"].mean(), color="#1A7A4A", linestyle="--", linewidth=2)

# Plot 4: MCS distribution per archetype
ax = axes[1, 0]
arch_colors = {"proactive": "#1A7A4A", "rider_triggered": "#E8612A", "erratic": "#2E86AB", "non_compiler": "#B91C1C"}
for arch, color in arch_colors.items():
    subset = mcs[mcs["archetype_true"] == arch]["mean_credibility"]
    ax.hist(subset, bins=15, alpha=0.7, color=color, label=arch.replace("_", " ").title())
ax.set_xlabel("Merchant Compliance Score (MCS)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("MCS by True Archetype (Validation)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# Plot 5: % rider-triggered by archetype
ax = axes[1, 1]
arch_order = ["proactive", "erratic", "rider_triggered", "non_compiler"]
means      = [mcs[mcs["archetype_true"] == a]["pct_rider_triggered"].mean() * 100 for a in arch_order]
bar_colors = [arch_colors[a] for a in arch_order]
bars       = ax.bar([a.replace("_", "\n").title() for a in arch_order], means, color=bar_colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("% Rider-Triggered Events", fontsize=10)
ax.set_title("FOR Contamination Rate by Archetype", fontsize=11, fontweight="bold")
ax.set_ylim(0, 100)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

# Plot 6: MAE improvement bar
ax = axes[1, 2]
categories = ["Raw FOR Labels\n(Baseline)", "Corrected Labels\n(Our System)"]
mae_vals   = [scored["label_error_raw"].mean(), scored["label_error_fixed"].mean()]
bars       = ax.bar(categories, mae_vals, color=["#E8612A", "#1A7A4A"], width=0.45, edgecolor="white", linewidth=2)
ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=10)
ax.set_title("KPT Label MAE Improvement", fontsize=11, fontweight="bold")
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{val:.2f} min", ha="center", fontsize=11, fontweight="bold")
ax.annotate(f"↓ {improvement:.1f}% reduction", xy=(1, mae_vals[1]), xytext=(0.5, mae_vals[0] * 0.6),
            fontsize=11, color="#1A7A4A", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1A7A4A"))

plt.tight_layout()
plt.savefig("for_credibility_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: for_credibility_analysis.png")

# ================================================================
# STEP 7 — EXPORT
# ================================================================

scored.to_csv("orders_scored.csv", index=False)
mcs.to_csv("merchant_mcs.csv", index=False)

print("\nExported:")
print("  orders_scored.csv  — per-event credibility weights and corrected labels")
print("  merchant_mcs.csv   — per-restaurant compliance scores")
print("  for_credibility_analysis.png — visualisation plots")
print("\nDone.")
