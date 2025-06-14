#!/usr/bin/env python3
import glob
import pandas as pd
import numpy as np
from scipy import stats
import os

D = 1e17  # dispersion factor (s)

def analyze_grb(path):
    df = pd.read_csv(path)
    delta_E = df['delta_E_GeV'].values
    delta_t = df['delta_t_s'].values
    
    # Linear regression using scipy.stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(delta_E, delta_t)
    
    # LIV energy scale estimates
    E_LV_est   = D / abs(slope) if slope != 0 else float('inf')
    E_LV_lower = D / (abs(slope) + 1.96*std_err) if (abs(slope) + 1.96*std_err) != 0 else float('inf')
    
    return E_LV_est, E_LV_lower

def main():
    results = []
    for path in glob.glob(os.path.join("data", "grbs", "*.csv")):
        est, lower = analyze_grb(path)
        results.append({
            'GRB file': path.split(os.sep)[-1],
            'E_LV_est (GeV)': est,
            'E_LV_lower95 (GeV)': lower
        })
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", "grb_bounds.csv"), index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
