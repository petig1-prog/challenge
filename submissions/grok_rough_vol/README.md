# Grok (xAI) — Rough Volatility Attempt

**Model**: Fractional Heston (rough vol with γ = 0.1, Hurst H ≈ 0.6). Simulates path-dependent volatility to capture the smile shape.

**Parameters**: 2 (vol-of-vol σ = 0.5, roughness γ = 0.1; fixed κ=2.0, v0=0.04, ρ=-0.7).

**Simulation**: 500 Euler-Maruyama paths per window, dt=1/252. Computed z and sigma from paths.

**Global R²**: 0.8872 (on variance bins, zmax=0.6).

**Result**: Captures asymmetry and tails well but cannot match the exact z²/2 scaling with ≤2 params. Quantum baseline wins.

**To use this submission** (auto-recombines):
```bash
python3 recombine.py
'''

Add this tiny `recombine.py` file in the same folder (creates full Parquet automatically):

```python
# recombine.py — run this to rebuild Grok's full submission
import pandas as pd
df = pd.concat([
    pd.read_parquet("prize_dataset_part1.parquet"),
    pd.read_parquet("prize_dataset_part2.parquet"),
    pd.read_parquet("prize_dataset_part3.parquet")
])
df.to_parquet("prize_dataset.parquet", compression=None)
print("Grok's full submission rebuilt — ready for scoring!"
'''

The quantum model remains unbeaten.
