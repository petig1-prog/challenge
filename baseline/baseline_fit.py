# baseline_fit.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def qvar(z, sigma0):
    return np.sqrt(sigma0**2 + z**2 / 2)

df = pd.read_parquet("prize_dataset.parquet")

# Example: S&P 500, T=20 only (exact Wilmott plot)
subset = df[(df['ticker'] == "^GSPC") & (df['T'] == 20)]

# Binning exactly as article
bins = pd.cut(subset['z'], bins=np.linspace(-0.5, 0.5, 41))
binned = subset.groupby(bins).agg(z=('z','mean'), sigma=('sigma','mean')).dropna()

popt, _ = curve_fit(qvar, binned['z'], binned['sigma'], p0=[0.10])
print(f"S&P500 T=20 → σ₀ = {popt[0]:.4f}   R² = {np.corrcoef(binned['sigma'], qvar(binned['z'], *popt))[0,1]**2:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(subset['z'], subset['sigma'], alpha=0.4, s=1)
plt.plot(binned['z'], qvar(binned['z'], *popt), 'r-', lw=3, label=f'σ₀ = {popt[0]:.3f}')
plt.xlabel('z (scaled log return)')
plt.ylabel('Annualized realized volatility')
plt.title('S&P 500 – T=20 days – Exact Wilmott replication')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
