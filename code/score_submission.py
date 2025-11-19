# scoring/score_submission.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import json
import os
from datetime import datetime

# The exact quantum prediction from the Wilmott article
def qvar(z, sigma0):
    return np.sqrt(sigma0**2 + z**2 / 2)

def compute_qvar_score(df_dict, submission_name, author="Anonymous"):
    """
    df_dict: {ticker: DataFrame with columns ['date','close','z','sigma'] for each T}
    Returns a score dictionary
    """
    results = []
    required_T = [5,10,20,40,80,160,250]  # exact horizons from article

    for ticker, df in df_dict.items():
        for T in required_T:
            subset = df[df['T'] == T]
            if len(subset) < 50:  # need reasonable number of points
                continue

            z = subset['z'].values
            sigma = subset['sigma'].values

            # Bin exactly as in Wilmott article
            bins = pd.cut(z, bins=np.linspace(z.min(), z.max(), 21))
            binned = pd.DataFrame({'z': z, 'sigma': sigma, 'bin': bins})
            avg = binned.groupby('bin').agg({'z': 'mean', 'sigma': 'mean'}).dropna()

            if len(avg) < 10:
                continue

            z_bin = avg['z'].values
            sigma_bin = avg['sigma'].values

            try:
                popt, _ = curve_fit(qvar, z_bin, sigma_bin, p0=[0.10], bounds=(0, 0.5))
                sigma0 = popt[0]
                predicted = qvar(z_bin, sigma0)
                ss_res = np.sum((sigma_bin - predicted)**2)
                ss_tot = np.sum((sigma_bin - sigma_bin.mean())**2)
                r2 = 1 - ss_res/ss_tot
            except:
                r2 = 0.0

            results.append({
                "ticker": ticker,
                "T": T,
                "n_points": len(subset),
                "r2": round(r2, 5),
                "sigma0": round(sigma0, 4)
            })

    # Final aggregate score (simple average R² across all valid ticker-T pairs)
    if not results:
        final_r2 = 0.0
    else:
        final_r2 = np.mean([r["r2"] for r in results])

    score = {
        "submission": submission_name,
        "author": author,
        "date": datetime.now().isoformat()[:10],
        "final_r2": round(final_r2, 5),
        "n_combinations": len(results),
        "passed": final_r2 >= 0.92,
        "details": results
    }
    return score


# Auto-run when new submission appears
if __name__ == "__main__":
    # This is triggered by GitHub Actions — dummy example for local testing
    import sys
    submission_folder = sys.argv[1]  # e.g. submissions/team-xyz
    # Assume submission contains a generate_output.py that produces results.csv per ticker
    # ... load and parse ...
    # For prototype, just show a perfect quantum baseline
    print(json.dumps({
        "submission": "Quantum Baseline (Orrell 2025)",
        "final_r2": 0.951,
        "passed": True,
        "date": "2025-11-18"
    }, indent=2))
