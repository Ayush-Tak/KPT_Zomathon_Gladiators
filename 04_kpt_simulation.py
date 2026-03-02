"""
===============================================================
PILLAR 5 — KPT Accuracy Simulation (M/M/c Queueing Model)
===============================================================
Pure NumPy discrete-event simulation — no external deps beyond
numpy, matplotlib, scipy (all standard).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

np.random.seed(2024)

@dataclass
class SimConfig:
    n_stations:          int   = 3
    true_kpt_mean:       float = 12.0
    true_kpt_std:        float = 4.0
    arrival_rate:        float = 0.18
    sim_duration:        float = 480.0
    dispatch_lead:       float = 8.0
    rider_travel_mean:   float = 5.0
    rider_travel_std:    float = 1.5
    baseline_bias_mean:  float = 3.5
    baseline_bias_std:   float = 2.5
    enhanced_bias_mean:  float = 0.4
    enhanced_bias_std:   float = 1.2
    kli_correction:      float = 0.6

CFG = SimConfig()

def run_simulation(cfg, scenario, seed=0):
    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / cfg.arrival_rate, int(cfg.sim_duration * cfg.arrival_rate * 3))
    arrival_times  = np.cumsum(inter_arrivals)
    arrival_times  = arrival_times[arrival_times < cfg.sim_duration]
    n_orders       = len(arrival_times)

    shape    = (cfg.true_kpt_mean / cfg.true_kpt_std) ** 2
    scale    = cfg.true_kpt_std ** 2 / cfg.true_kpt_mean
    true_kpt = np.clip(rng.gamma(shape, scale, n_orders), 2.0, 60.0)
    rider_tr = np.clip(rng.normal(cfg.rider_travel_mean, cfg.rider_travel_std, n_orders), 1.0, 20.0)

    station_free = np.zeros(cfg.n_stations)
    rider_waits, eta_errors, is_delayed = [], [], []

    for i in range(n_orders):
        t      = arrival_times[i]
        kpt_t  = true_kpt[i]
        busy   = int(np.sum(station_free > t))

        if scenario == "baseline":
            kpt_p = max(3.0, kpt_t - rng.normal(cfg.baseline_bias_mean, cfg.baseline_bias_std))
        else:
            kpt_p = max(3.0, kpt_t - rng.normal(cfg.enhanced_bias_mean, cfg.enhanced_bias_std))
            if busy >= cfg.n_stations:
                kpt_p += (busy - cfg.n_stations + 1) * 1.2 * cfg.kli_correction

        eta_c         = t + kpt_p + cfg.rider_travel_mean
        idx           = np.argmin(station_free)
        cook_start    = max(t, station_free[idx])
        cook_done     = cook_start + kpt_t
        station_free[idx] = cook_done

        rider_arrives = max(t, t + kpt_p - cfg.dispatch_lead) + rider_tr[i]
        pickup_time   = max(cook_done, rider_arrives)

        rider_waits.append(max(0.0, cook_done - rider_arrives))
        eta_errors.append(abs(pickup_time + cfg.rider_travel_mean - eta_c))
        is_delayed.append(float((pickup_time + cfg.rider_travel_mean - eta_c) > 5.0))

    return np.array(rider_waits), np.array(eta_errors), np.array(is_delayed)

N_REPS = 12
print(f"Running {N_REPS} replications per scenario...")

def aggregate(scenario):
    rw, ee, dl = [], [], []
    for r in range(N_REPS):
        a, b, c = run_simulation(CFG, scenario, seed=r*100 + (0 if scenario=="baseline" else 500))
        rw.extend(a); ee.extend(b); dl.extend(c)
    return np.array(rw), np.array(ee), np.array(dl)

b_rw, b_ee, b_dl = aggregate("baseline")
e_rw, e_ee, e_dl = aggregate("enhanced")
print(f"  {len(b_rw):,} orders simulated per scenario\n")

metrics = [
    ("P50 ETA Error (min)",      np.percentile(b_ee,50), np.percentile(e_ee,50)),
    ("P90 ETA Error (min)",      np.percentile(b_ee,90), np.percentile(e_ee,90)),
    ("Avg Rider Wait (min)",     b_rw.mean(),             e_rw.mean()),
    ("P90 Rider Wait (min)",     np.percentile(b_rw,90),  np.percentile(e_rw,90)),
    ("Order Delay Rate (%)",     b_dl.mean()*100,         e_dl.mean()*100),
]

print(f"{'='*62}")
print(f"  {'METRIC':<28}  {'BASELINE':>9}  {'ENHANCED':>9}  {'IMPROVE':>8}")
print(f"{'='*62}")
for name, bv, ev in metrics:
    print(f"  {name:<28}  {bv:>9.2f}  {ev:>9.2f}  {(1-ev/bv)*100:>+7.1f}%")
print(f"{'='*62}")

t_ee, p_ee = stats.ttest_ind(b_ee, e_ee, equal_var=False)
t_rw, p_rw = stats.ttest_ind(b_rw, e_rw, equal_var=False)
print(f"\n  t-test ETA Error:   t={t_ee:.1f}, p={p_ee:.2e}  {'✓ p<0.001' if p_ee<0.001 else ''}")
print(f"  t-test Rider Wait:  t={t_rw:.1f}, p={p_rw:.2e}  {'✓ p<0.001' if p_rw<0.001 else ''}")

BL, EN = "#E8612A", "#1A7A4A"
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("KPT Simulation: Baseline vs Signal Enrichment Layer", fontsize=15, fontweight="bold")

ax = fig.add_subplot(gs[0, :2])
ax.hist(b_ee, bins=70, alpha=0.55, color=BL, density=True,
        label=f"Baseline  (P50={np.percentile(b_ee,50):.1f}  P90={np.percentile(b_ee,90):.1f} min)")
ax.hist(e_ee, bins=70, alpha=0.55, color=EN, density=True,
        label=f"Enhanced  (P50={np.percentile(e_ee,50):.1f}  P90={np.percentile(e_ee,90):.1f} min)")
ax.axvline(np.percentile(b_ee,90), color=BL, linestyle="--", lw=1.8, alpha=0.8)
ax.axvline(np.percentile(e_ee,90), color=EN, linestyle="--", lw=1.8, alpha=0.8)
ax.set_xlabel("ETA Prediction Error (minutes)"); ax.set_ylabel("Density")
ax.set_title("ETA Prediction Error Distribution", fontweight="bold"); ax.legend(fontsize=9); ax.set_xlim(0,35)

ax = fig.add_subplot(gs[0, 2:])
ax.hist(b_rw, bins=60, alpha=0.55, color=BL, density=True, label=f"Baseline  (avg={b_rw.mean():.2f} min)")
ax.hist(e_rw, bins=60, alpha=0.55, color=EN, density=True, label=f"Enhanced  (avg={e_rw.mean():.2f} min)")
ax.set_xlabel("Rider Wait Time at Restaurant (minutes)"); ax.set_ylabel("Density")
ax.set_title("Rider Wait Time Distribution", fontweight="bold"); ax.legend(fontsize=9)

bar_mets = [
    ("P50 ETA Error\n(minutes)", np.percentile(b_ee,50), np.percentile(e_ee,50), gs[1,0]),
    ("P90 ETA Error\n(minutes)", np.percentile(b_ee,90), np.percentile(e_ee,90), gs[1,1]),
    ("Avg Rider Wait\n(minutes)", b_rw.mean(),           e_rw.mean(),            gs[1,2]),
    ("Order Delay Rate\n(%)",    b_dl.mean()*100,        e_dl.mean()*100,        gs[1,3]),
]
for title, bv, ev, gsp in bar_mets:
    ax   = fig.add_subplot(gsp)
    imp  = (1-ev/bv)*100
    bars = ax.bar(["Baseline","Enhanced"], [bv,ev], color=[BL,EN], width=0.5, edgecolor="white", lw=2)
    ax.set_title(title, fontweight="bold", fontsize=10)
    for bar, val in zip(bars, [bv,ev]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+bv*0.015,
                f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.annotate(f"↓{imp:.0f}%", xy=(1,ev), xytext=(0.5,bv*0.5),
                fontsize=13, color=EN, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=EN, lw=2))
    ax.set_ylim(0, bv*1.35)

plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved: simulation_results.png")
print("Done.")
