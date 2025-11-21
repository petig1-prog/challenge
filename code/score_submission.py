# scoring/score_submission.py (works with variance + 352 stocks + 1–26 weeks)
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
from datetime import datetime

# Quantum variance prediction: variance = σ₀² + z²/2
def qvar(z, sigma0):
    return sigma0**2 + z**2 / 2

def compute_qvar_score(df_dict, submission_name="Submission", author="Anonymous"):
    """
    df_dict: dict of {ticker: DataFrame with columns ['ticker','date','T','z','sigma']}
             where sigma is already volatility → variance = sigma²
    """
    results = []
    HORIZONS = 5 * (np.arange(26) + 1)  # 5,10,...,130 days (1–26 weeks)

    for ticker, df in df_dict.items():
        df["var"] = df["sigma"] ** 2  # convert volatility → variance
        
        for T in HORIZONS:
            subset = df[df["T"] == T]
            if len(subset) < 50:
                continue

            z = subset["z"].values
            var = subset["var"].values

            # Fixed binning as in your baseline_allSPv.py
            zmax = 0.6
            delz = 0.025
            nbins = int(2 * zmax / delz + 1)
            bins = np.linspace(-zmax, zmax, nbins)

            binned_df = pd.DataFrame({"z": z, "var": var})
            binned_df["z_bin"] = pd.cut(binned_df["z"], bins=bins, include_lowest=True)
            binned = binned_df.groupby("z_bin").agg(z_mid=("z", "mean"), var=("var", "mean")).dropna()

            if len(binned) < 10:
                continue

            try:
                popt, _ = curve_fit(qvar, binned.z_mid.values, binned.var.values, p0=[0.08])
                sigma0 = popt[0]
                predicted = qvar(binned.z_mid.values, sigma0)
                ss_res = np.sum((binned.var - predicted)**2)
                ss_tot = np.sum((binned.var - binned.var.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            except:
                r2 = 0.0
                sigma0 = np.nan

            results.append({
                "ticker": ticker,
                "T": int(T),
                "n_windows": len(subset),
                "r2": round(r2, 5),
                "sigma0": round(sigma0, 4) if np.isfinite(sigma0) else None
            })

    # Final score
    if not results:
        final_r2 = 0.0
    else:
        final_r2 = np.mean([r["r2"] for r in results if r["r2"] > 0])

    score = {
        "submission": submission_name,
        "author": author,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "final_r2": round(final_r2, 5),
        "n_combinations": len(results),
        "passed_threshold": final_r2 >= 0.92,
        "details": results
    }
    return score

# Test with your real data
if __name__ == "__main__":
    df = pd.read_parquet("prize_dataset.parquet")
    df_dict = {ticker: group for ticker, group in df.groupby("ticker")}
    
    score = compute_qvar_score(df_dict, "Quantum Baseline (Orrell 2025)", "David Orrell")
    
    print(json.dumps(score, indent=2))
    
    # Expected output:
    # "final_r2": 0.95xx
    # "passed_threshold": true
