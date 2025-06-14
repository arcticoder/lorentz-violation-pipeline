#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaincinv
import os

# --- Configuration ---
INPUT_CSV = os.path.join("summary", "dataSummarySD1500.csv")
OUTPUT_CSV = os.path.join("data", "uhecr", "sd1500_spectrum.csv")
PLOT_FILE  = os.path.join("results", "sd1500_spectrum.png")

# Calibration constants
A_cal = 4.17e16  # eV
B_cal = 1.07

# Bin edges
logE_min, logE_max, n_bins = 17.5, 20.0, 25
bin_edges = np.linspace(logE_min, logE_max, n_bins + 1)
bin_centers_log = 0.5 * (bin_edges[:-1] + bin_edges[1:])

def gehrels_errors(N, CL=0.84):
    alpha = 1 - CL
    if N > 0:
        low  = N - gammaincinv(N, alpha/2)
        high = gammaincinv(N+1, 1-alpha/2) - N
    else:
        low, high = 0.0, gammaincinv(1, 1-alpha/2)
    return low, high

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # Data validation and cleaning
    print(f"Initial data: {len(df)} events")
    df = df.dropna(subset=['sd_s38', 'sd_exposure'])  # Remove NaN values
    df = df[df['sd_s38'] > 0]  # Signal must be positive
    df = df[df['sd_exposure'] > 0]  # Exposure must be positive
    print(f"After cleaning: {len(df)} valid events")
    
    E = A_cal * (df['sd_s38'] ** B_cal)        # in eV
    expo = df['sd_exposure'].values            # km² sr yr

    print(f"Energy range: {E.min():.2e} - {E.max():.2e} eV")
    print(f"Energy range: {E.min()/1e18:.2f} - {E.max()/1e18:.1f} EeV")
    print(f"Total exposure: {expo.sum():.2e} km² sr yr")

    inds = np.digitize(np.log10(E), bin_edges) - 1
    results = []

    for i in range(n_bins):
        mask = (inds == i)
        N = mask.sum()
        lowN, highN = gehrels_errors(N)
        expo_bin = expo[mask].sum()
        E_center = 10**bin_centers_log[i]
        dlogE = bin_edges[1] - bin_edges[0]
        dE = E_center * np.log(10) * dlogE

        flux     = N / (expo_bin * dE) if expo_bin>0 else 0.0
        err_low  = lowN / (expo_bin * dE) if expo_bin>0 else 0.0
        err_high = highN/ (expo_bin * dE) if expo_bin>0 else 0.0

        results.append({
            'logE_center':    bin_centers_log[i],
            'E_center_eV':    E_center,
            'N_events':       N,
            'expo_km2sr_yr':  expo_bin,
            'flux':           flux,
            'flux_err_low':   err_low,
            'flux_err_high':  err_high
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

    df_spec = pd.DataFrame(results)
    df_spec.to_csv(OUTPUT_CSV, index=False)

    # Plot E^3 * J(E)
    plt.errorbar(
        df_spec['E_center_eV'],
        df_spec['flux'] * df_spec['E_center_eV']**3,
        yerr=[
            df_spec['flux_err_low'] * df_spec['E_center_eV']**3,
            df_spec['flux_err_high']* df_spec['E_center_eV']**3
        ],
        fmt='o', ms=4
    )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (eV)')
    plt.ylabel(r'$E^3J(E)$ [km⁻² sr⁻¹ yr⁻¹ eV²]')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.title('Auger SD1500 UHECR Spectrum')
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Spectrum: {n_bins} bins -> {OUTPUT_CSV}")
    print(f"Plot: {PLOT_FILE}")

if __name__ == "__main__":
    main()
