# KPT Signal Enrichment — Code Submission
## Improving Kitchen Prep Time Prediction at Zomato Scale

---

## File Structure

```
01_for_credibility_scoring.py   — Pillar 1: Clean noisy FOR labels
02_merchant_clustering.py       — Pillar 1: Segment merchants into archetypes
03_kli_feature_engineering.py   — Pillar 2: Kitchen Load Index (proxy signals)
04_kpt_simulation.py            — Pillar 5: M/M/c queueing simulation (BONUS)
05_delta_engine.py              — Pillar 4: Real-time KPT adjustment model
```

---

## How to Run (in order)

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib scikit-learn simpy scipy lightgbm

# 2. Run each script in sequence
#    Each script generates CSVs used by the next

python 01_for_credibility_scoring.py
# Outputs: orders_scored.csv, merchant_mcs.csv, for_credibility_analysis.png

python 02_merchant_clustering.py
# Outputs: merchant_archetypes.csv, merchant_archetypes.png

python 03_kli_feature_engineering.py
# Outputs: kli_features.csv, kli_analysis.png

python 04_kpt_simulation.py
# Outputs: simulation_results.png  (key quantitative results)

python 05_delta_engine.py
# Outputs: delta_engine_results.png
```

---

## What Each Script Demonstrates

| Script | Evaluation Criterion Addressed |
|--------|-------------------------------|
| 01 | Signal Quality Improvement — de-biasing FOR labels |
| 02 | Signal Quality — merchant archetype as model feature |
| 03 | Restaurant Live Rush — capturing dine-in + competitor load |
| 04 | Simulation bonus — quantitative improvement proof |
| 05 | Real-time system — Delta Engine implementation |

---

## Key Results (from simulation — script 04)

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| P50 ETA Error | ~7 min | ~4 min | ~40% |
| P90 ETA Error | ~15 min | ~9 min | ~38% |
| Avg Rider Wait | ~5 min | ~2 min | ~52% |
| Order Delay Rate | ~18% | ~11% | ~38% |

---

## Architecture Note

All scripts use **synthetic data** that mirrors realistic distributions
based on the problem statement parameters. In production:

- `01` and `02`: replace synthetic data with Zomato orders DB query
- `03`: replace with Google Places API + Zomato reservations DB
- `04`: parameterise with real KPT distribution from Zomato's historical data
- `05`: replace with real feature joins from the Signal Enrichment Layer

The model architecture (existing KPT model) is **never touched**.
All improvements operate at the data / signal / post-processing layer.
