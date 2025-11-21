# baseline_fit.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_parquet("prize_dataset.parquet")   # load the parquet file from data_loader.py

# Select S&P 500, T=5
# data = df[(df["ticker"] == "^GSPC") & (df["T"] == 5)].copy()
# data = df[(df["ticker"] == "^GSPC") ].copy()
data = df.copy()
data["var"] = data.sigma**2

#print(f"S&P 500 T=5: {len(data)} windows")
print(f"z has NaNs: {data['z'].isna().sum()}")  # → 0

zmax = 0.6
delz = 0.025
nbins = int(2*zmax/delz + 1)

#bins = np.linspace(-0.5, 0.5, 41)         # fixed bins
bins = np.linspace(-zmax, zmax, nbins)         # fixed bins
# create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
               .groupby('z_bin')
               .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
               .dropna())

# zmid = (bins[0:(nbins-1)] + bins[1:(nbins)])/2

def qvar(z, s0, zoff):    # define q-variance function, parameter is minimal volatility s0
    return (s0**2 + (z - zoff)**2 / 2)

# curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.02, 0])

fitted = qvar(binned.z_mid, popt[0], popt[1])  # cols are z_bin, which is a range like (-0.601, -0.55], and qvar
r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)

print(f"σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")

# plot of all stocks
plt.figure(figsize=(9,7))
plt.scatter(data.z, data['var'], c='steelblue', alpha=0.1, s=1, edgecolor='none')
plt.plot(binned.z_mid, binned['var'], 'b-', lw=3)     # label='binned'
plt.plot(binned.z_mid, fitted, 'red', lw=4, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')

plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised variance', fontsize=12)
plt.title('All data T=1 to 26 weeks – Q-Variance', fontsize=14)

plt.xlim(-zmax, zmax) 
plt.ylim(0.0, 0.35)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.show()

# now do a panel figure for selected assets 
# The 8 tickers in order to appear
#TICKERS = ["^GSPC", "^DJI", "^FTSE", "AAPL", "MSFT", "AMZN", "JPM","PG"]  # BTC-USD

# 352 stocks from SP500
TICKERS = ["A" , "AAPL", "ABT", "ACGL","ADBE" , "ADI" , "ADM" , "ADP" , "ADSK" , "AEE" , "AEP" ,
"AES" , "AFL" , "AIG" , "AJG" , "AKAM" , "ALB" , "ALL" , "AMAT" , "AMD" , "AME" , "AMGN",
 "AMT" , "AMZN" , "AON" , "AOS" , "APA" , "APD" , "APH" , "ARE" , "ATO" , "AVB",            # no "ANSS" 
"AVY" , "AXP" , "AZO" , "BA" , "BAC" , "BALL" , "BAX" , "BBY" , "BDX" , "BEN" , "BIIB",
"BK" , "BKNG" , "BKR" , "BLK" , "BMY" , "BRO" , "BSX" , "BXP" , "C" , "CAG" , "CAH",
"CAT" , "CB" , "CCI" , "CCL" , "CDNS" , "CHD" , "CHRW" , "CI" , "CINF" , "CL" , "CLX",  
"CMCSA" , "CMI" , "CMS" , "CNP" , "COF" , "COO" , "COP" , "COR" , "COST" , "CPB" , "CPRT",
"CPT" , "CSCO" , "CSGP" , "CSX" , "CTAS" , "CTRA" , "CTSH" , "CVS" , "CVX" , "D" , "DD",
"DE" , "DECK" , "DGX" , "DHI" , "DHR" , "DIS" , "DLTR" , "DOC" , "DOV" , "DRI" , "DTE",
"DUK" , "DVA" , "DVN" , "EA" , "EBAY" , "ECL" , "ED" , "EFX" , "EG" , "EIX" , "EL",
"EMN" , "EMR" , "EOG" , "EQR" , "EQT" , "ERIE" , "ES" , "ESS" , "ETN" , "ETR" , "EVRG",
"EW" , "EXC" , "EXPD" , "F" , "FAST" , "FCX" , "FDS" , "FDX" , "FE" , "FFIV" , "FI",
"FICO" , "FITB" , "FRT" , "GD" , "GE" , "GEN" , "GILD" , "GIS" , "GL" , "GLW" , "GPC", 
"GS" , "GWW" , "HAL" , "HAS" , "HBAN" , "HD"  , "HIG" , "HOLX" , "HON" , "HPQ",    # no "HES"
"HRL" , "HSIC" , "HST" , "HSY" , "HUBB" , "HUM" , "IBM" , "IDXX" , "IEX" , "IFF" , "INCY",
"INTC" , "INTU" , "IP" , "IPG" , "IRM" , "IT" , "ITW" , "IVZ" , "J" , "JBHT" , "JBL",
 "JCI" , "JKHY" , "JNJ" , "JPM" , "K" , "KEY" , "KIM" , "KLAC" , "KMB" , "KMX",   # no "JNPR"
 "KO" , "KR" , "L" , "LEN" , "LH" , "LHX" , "LII" , "LIN" , "LLY" , "LMT" , "LNT",
 "LOW" , "LRCX" , "LUV" , "MAA" , "MAR" , "MAS" , "MCD" , "MCHP" , "MCK" , "MCO" , "MDT",
 "MET" , "MGM" , "MHK" , "MKC" , "MLM" , "MMC" , "MMM" , "MNST" , "MO" , "MOS" , "MRK",
"MS" , "MSFT" , "MSI" , "MTB" , "MTCH" , "MTD" , "MU" , "NDSN" , "NEE" , "NEM" , "NI",
"NKE" , "NOC" , "NSC" , "NTAP" , "NTRS" , "NUE" , "NVDA" , "NVR" , "O" , "ODFL" , "OKE",
"OMC" , "ORCL" , "ORLY" , "OXY" , "PAYX" , "PCAR" , "PCG" , "PEG" , "PEP" , "PFE" , "PG", 
 "PGR" , "PH" , "PHM" , "PKG" , "PLD" , "PNC" , "PNR" , "PNW" , "POOL" , "PPG" , "PPL",
 "PSA" , "PTC" , "PWR" , "QCOM" , "RCL" , "REG" , "REGN" , "RF" , "RJF" , "RL" , "RMD",
"ROK" , "ROL" , "ROP" , "ROST" , "RSG" , "RTX" , "RVTY" , "SBAC" , "SBUX" , "SCHW" , "SHW",
 "SJM" , "SLB" , "SNA" , "SNPS" , "SO" , "SPG" , "SPGI" , "SRE" , "STE" , "STLD" , "STT",
"STZ" , "SWK" , "SWKS" , "SYK" , "SYY" , "T" , "TAP" , "TDY" , "TECH" , "TER" , "TFC",
 "TGT" , "TJX" , "TKO" , "TMO" , "TPL" , "TRMB" , "TROW" , "TRV" , "TSCO" , "TSN" , "TT", 
"TTWO" , "TXN" , "TXT" , "TYL" , "UDR" , "UHS" , "UNH" , "UNP" , "UPS" , "URI" , "USB",
"VLO" , "VMC" , "VRSN" , "VRTX" , "VTR" , "VTRS" , "VZ" , "WAB" , "WAT" , "WDC",    # no  "WBA"
"WEC" , "WELL" , "WFC" , "WM" , "WMB" , "WMT" , "WRB" , "WSM" , "WST" , "WY" , "XEL",
"XOM" , "YUM" , "ZBRA"]

# T = 20                                      # fixed horizon

TICKERS = TICKERS[:100]  # first 100
# Set up the grid
fig, axes = plt.subplots(10, 10, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()                       # makes indexing easy: axes[0] … axes[7]

for idx, ticker in enumerate(TICKERS):
    ax = axes[idx]
    data = df[(df.ticker == ticker)].copy()
    data["var"] = data.sigma**2

    # Create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
    binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
                  .groupby('z_bin')
                  .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
                  .dropna())
              
    # Curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
    print(ticker)
    popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.12, 0])     #  maxfev=2000
    fitted = qvar(binned.z_mid, popt[0], popt[1])
    
    r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)

    # Plot scatter + fit
    ax.scatter(data.z, data['var'], c='steelblue', alpha=0.1, s=0.5, edgecolor='none')
    ax.plot(binned.z_mid, binned['var'], 'b-', lw=1.5)
    ax.plot(binned.z_mid, fitted, 'red', lw=1.5, label= f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.4f}, R² = {r2:.3f}')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 0.4)
    ax.set_title(ticker, fontsize=7, pad=0)
    #ax.legend(fontsize=6, loc='upper right')
    ax.grid(alpha=0.3)

# Shared labels
fig.supxlabel('z (scaled log return)', fontsize=10, y=0.04)
fig.supylabel('Annualised realised volatility', fontsize=10, x=0.04)

# Big main title
# plt.suptitle('Non-overlapping windows across 8 assets', fontsize=12, y=0.96, weight='bold')

plt.tight_layout(rect=[0.05, 0.05, 1, 0.94])   # makes room for suptitle
plt.show()



