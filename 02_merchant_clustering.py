"""
===============================================================
PILLAR 1 — Merchant Behavioural Segmentation (Clustering)
===============================================================
Purpose:
  Cluster all restaurants into 4 behavioural archetypes based
  on their historical FOR marking patterns. The archetype label
  is a feature fed directly into the KPT model.

Input:  merchant_mcs.csv (from 01_for_credibility_scoring.py)
Output: merchant_archetypes.csv, archetype clustering plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ================================================================
# STEP 1 — LOAD & PREPARE FEATURES
# ================================================================

print("Loading merchant compliance data...")
mcs = pd.read_csv("merchant_mcs.csv")
print(f"  Loaded {len(mcs)} merchants\n")

# Features used for clustering
FEATURES = [
    "mean_credibility",
    "pct_genuine",
    "pct_rider_triggered",
    "mean_delta_sec",
    "total_for_events",
]

X_raw = mcs[FEATURES].fillna(0).values
scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)

# ================================================================
# STEP 2 — FIND OPTIMAL K (ELBOW + SILHOUETTE)
# ================================================================

k_range  = range(2, 9)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))

# ================================================================
# STEP 3 — FIT FINAL MODEL (k=4 from domain knowledge)
# ================================================================

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=15)
mcs["cluster_id"] = kmeans.fit_predict(X)

# Name clusters by their mean_credibility and pct_genuine
cluster_stats = mcs.groupby("cluster_id")[FEATURES].mean()
cluster_stats["size"] = mcs.groupby("cluster_id").size()

# Sort by mean_credibility to assign archetype names
sorted_clusters = cluster_stats["mean_credibility"].sort_values(ascending=False)
archetype_names = {
    sorted_clusters.index[0]: "Proactive Markers",
    sorted_clusters.index[1]: "Erratic Markers",
    sorted_clusters.index[2]: "Rider-Triggered",
    sorted_clusters.index[3]: "Non-Compilers",
}
mcs["archetype_detected"] = mcs["cluster_id"].map(archetype_names)

# ================================================================
# STEP 4 — CLUSTER PROFILE SUMMARY
# ================================================================

print("=" * 60)
print("MERCHANT ARCHETYPE CLUSTERING — RESULTS")
print("=" * 60)

profile = mcs.groupby("archetype_detected").agg(
    count             = ("restaurant_id", "count"),
    mean_mcs          = ("mean_credibility", "mean"),
    pct_genuine       = ("pct_genuine", "mean"),
    pct_rider_trig    = ("pct_rider_triggered", "mean"),
    avg_delta_sec     = ("mean_delta_sec", "mean"),
).round(3)

print(f"\nSilhouette Score (k=4): {silhouette_score(X, kmeans.labels_):.3f}  (>0.5 is good)\n")
print(profile.to_string())

print(f"\nModel Feature Encoding:")
for name in archetype_names.values():
    code = list(archetype_names.values()).index(name)
    print(f"  {name:25s} → archetype_code = {code}")

# ================================================================
# STEP 5 — VISUALISATIONS
# ================================================================

ARCH_COLORS = {
    "Proactive Markers":  "#1A7A4A",
    "Erratic Markers":    "#2E86AB",
    "Rider-Triggered":    "#E8612A",
    "Non-Compilers":      "#B91C1C",
}

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Merchant Archetype Segmentation", fontsize=15, fontweight="bold")

# Plot 1: Elbow + Silhouette
ax1 = axes[0, 0]
ax2 = ax1.twinx()
l1, = ax1.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=6, label="Inertia")
l2, = ax2.plot(list(k_range), sil_scores, "rs--", linewidth=2, markersize=6, label="Silhouette")
ax1.axvline(x=4, color="gray", linestyle=":", linewidth=2, label="Chosen k=4")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia", color="blue")
ax2.set_ylabel("Silhouette Score", color="red")
ax1.set_title("Elbow + Silhouette Method", fontweight="bold")
ax1.legend(handles=[l1, l2], loc="center right")

# Plot 2: PCA 2D cluster view
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
ax    = axes[0, 1]
for name, color in ARCH_COLORS.items():
    mask = mcs["archetype_detected"] == name
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.6, s=40, edgecolors="white", linewidths=0.5)
ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("Cluster View (PCA Projection)", fontweight="bold")
ax.legend(fontsize=8)

# Plot 3: Archetype size
ax = axes[1, 0]
sizes = mcs["archetype_detected"].value_counts()
bars  = ax.bar(sizes.index, sizes.values,
               color=[ARCH_COLORS[n] for n in sizes.index],
               edgecolor="white", linewidth=1.5)
ax.set_ylabel("Number of Restaurants")
ax.set_title("Merchant Count per Archetype", fontweight="bold")
ax.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, sizes.values):
    pct = val / len(mcs) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{pct:.0f}%", ha="center", fontsize=9)

# Plot 4: Radar-style bar cluster profile
ax   = axes[1, 1]
feat_labels = ["MCS Score", "% Genuine", "% Rider-Trig.", "Normalised δ"]
x    = np.arange(len(feat_labels))
w    = 0.18
i    = 0
for name, color in ARCH_COLORS.items():
    row = profile.loc[name]
    vals = [
        row["mean_mcs"],
        row["pct_genuine"],
        row["pct_rider_trig"],
        max(0, min(1, (row["avg_delta_sec"] + 300) / 600)),  # normalise delta to 0-1
    ]
    ax.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85)
    i += 1
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(feat_labels, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title("Archetype Feature Profiles", fontweight="bold")
ax.legend(fontsize=7, loc="upper right")
ax.set_ylabel("Normalised Score (0–1)")

plt.tight_layout()
plt.savefig("merchant_archetypes.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved: merchant_archetypes.png")

# ================================================================
# STEP 6 — EXPORT
# ================================================================

out = mcs[["restaurant_id", "archetype_detected", "mean_credibility",
           "pct_genuine", "pct_rider_triggered", "total_for_events"]].copy()
out["archetype_code"] = out["archetype_detected"].map({
    v: k for k, v in enumerate(ARCH_COLORS.keys())
})
out.to_csv("merchant_archetypes.csv", index=False)

print("\nExported:")
print("  merchant_archetypes.csv — archetype per restaurant (use as model feature)")
print("  merchant_archetypes.png — cluster visualisation")
print("\nDone.")
