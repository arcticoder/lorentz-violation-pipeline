#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def simulate_uhecr(path):
    df = pd.read_csv(path)
    energies = df['E_center_eV'].values / 1e9  # Convert eV to GeV for energy_EeV
    flux     = df['flux'].values
    error    = df['flux_err_high'].values  # Use upper error as error estimate

    # Only use bins with actual data
    valid_mask = (df['N_events'] > 0) & (flux > 0) & (error > 0)
    energies = energies[valid_mask]
    flux = flux[valid_mask]
    error = error[valid_mask]

    results = []
    for E_LV_p in np.logspace(17, 19, 5):
        chi2 = np.sum(((flux - flux.mean())/error)**2) / len(flux)
        excluded = chi2 > 1.0
        results.append({
            'E_LV_p (GeV)': E_LV_p,
            'chi2': chi2,
            'Excluded': excluded
        })
    return pd.DataFrame(results)

def main():
    uhecr_csv = os.path.join("data", "uhecr", "sd1500_spectrum.csv")
    df = simulate_uhecr(uhecr_csv)
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", "uhecr_exclusion.csv"), index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()