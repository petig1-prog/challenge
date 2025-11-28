# baseline_fit.py
# comment out lines to read files in 3 parts or 1 part
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# load the parquet files from data_loader.py
###df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])

df = pd.read_parquet("dataset.parquet")  # READ SUBMISSION DATA

data = df.copy()
data["var"] = data.sigma**2

print(f"{len(data)} windows")
print(f"z has NaNs: {data['z'].isna().sum()}")  # → 0

zmax = 0.6
delz = 0.025*2
nbins = int(2*zmax/delz + 1)
bins = np.linspace(-zmax, zmax, nbins)         # fixed bins

# create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
               .groupby('z_bin',observed=False)
               .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
               .dropna())

def qvar(z, s0, zoff):    # define q-variance function, parameter is minimal volatility s0
    return (s0**2 + (z - zoff)**2 / 2)

# curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
###popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.02, 0])  # fit this data
popt = [0.255, 0.020]  # same as optimized fit to data

fitted = qvar(binned.z_mid, popt[0], popt[1])  # cols are z_bin, which is a range like (-0.601, -0.55], and qvar
r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)

print(f"σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")

# plot of all stocks
markfac = 1  # default is 1, can increase to 3 if less data points
plt.figure(figsize=(9,7))
plt.scatter(data.z, data['var'], c='steelblue', alpha=markfac*0.1, s=markfac*1, edgecolor='none')
numeric_array = (1 - data["T"]/130)
string_array = [str(x) for x in numeric_array]
plt.plot(binned.z_mid, binned['var'], 'b-', lw=3)     # label='binned'
plt.plot(binned.z_mid, fitted, 'red', lw=3, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')

plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised variance', fontsize=12)
plt.title('Q-Variance: all data T=1 to 26 weeks', fontsize=14)

plt.xlim(-zmax, zmax) 
plt.ylim(0.0, 0.35)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.show()

# now check for time-invariant distribution

# Quantum density function — returns plain array for curve_fit
def quantum_density(z, sig0, zoff=0.0):
    ns = np.arange(0, 6)
    qdn = np.zeros_like(z, dtype=float)
    sigvec = sig0 * np.sqrt(2 * ns + 1)
    means = zoff * np.ones_like(ns)  # no drift term in pure Q-Variance

    for n in ns:
        weight = poisson.pmf(n, mu=0.5)
        qdn += weight * norm.pdf(z, loc=means[n], scale=sigvec[n])
    return qdn

# Plot setup
zlim = 2
zbins = np.linspace(-zlim, zlim, 51)
zmid = (zbins[:-1] + zbins[1:]) / 2

# Histogram
counts, _ = np.histogram(data["z"], bins=zbins, density=True)

# Fit quantum model
p0 = [0.62, 0.0]  # initial guess: sig0 ≈ 0.62 → σ₀ ≈ 0.079 after √2 scaling
popt, _ = curve_fit(quantum_density, zmid, counts, p0=p0, bounds=(0, [2.0, 0.5]))
sig0_fit, zoff_fit = popt

# Predict on fine grid
z_fine = np.linspace(-zlim, zlim, 1000)
q_pred_fine = quantum_density(z_fine, *popt)

# Predict on histogram bin centers for R²
q_pred_hist = quantum_density(zmid, *popt)
r2 = r2_score(counts, q_pred_hist)

print(f"Fit: σ₀ = {sig0_fit:.4f}, zoff = {zoff_fit:.4f}, R² = {r2:.4f}")

# obtain histogram bars
##counts, bin_edges, _ = plt.hist(data["z"], bins=zbins, density=True, visible=False)

# now plot with periods
TVEC = [5, 10, 20, 40, 80]

plt.figure(figsize=(9,7))
plt.plot(z_fine, q_pred_fine,
         color='red', lw=4,
         label=f'Q-Variance fit: σ₀ = {sig0_fit:.3f}, R² = {r2:.4f}')

for Tcur in TVEC:
    datacur = data[(data["T"] == Tcur)].copy()
    counts, bin_edges, _ = plt.hist(datacur["z"], bins=zbins, density=True, visible=False)
    r2 = r2_score(counts, q_pred_hist)    # use fit for whole data set
    colcur = str(Tcur/(max(TVEC)+20))
    #plt.plot(zmid, counts, c=colcur, lw=2,label=f'T = {Tcur/5:.0f}')  # , R² = {r2:.3f}' 
    plt.plot(zmid, counts, c=colcur, lw=2,label=f'T = {Tcur/5:.0f}, R² = {r2:.3f}' )

plt.title('Q-Variance: T dependence', fontsize=18, pad=20)
plt.xlabel('Scaled log-return z', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xlim(-1.2, 1.2)
plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()

# Save for announcement / paper
#plt.savefig("q_variance_density_with_R2.png", dpi=300, bbox_inches='tight')
#plt.savefig("q_variance_density_with_R2.pdf", bbox_inches='tight')

plt.show()
