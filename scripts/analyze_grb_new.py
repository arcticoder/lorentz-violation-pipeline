#!/usr/bin/env python3
import glob
import pandas as pd
import statsmodels.api as sm
import os

D = 1e17  # dispersion factor (s)

def analyze_grb(path):
    df = pd.read_csv(path)
    delta_E = df['delta_E_GeV'].values
    delta_t = df['delta_t_s'].values
    X = sm.add_constant(delta_E)
    model = sm.OLS(delta_t, X).fit()
    m, m_se = model.params[1], model.bse[1]
    E_LV_est   = D / m
    E_LV_lower = D / (m + 1.96*m_se)
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
