#!/usr/bin/env python3
import pandas as pd
import os

def main():
    grb_df   = pd.read_csv(os.path.join("results", "grb_bounds.csv"))
    uhecr_df = pd.read_csv(os.path.join("results", "uhecr_exclusion.csv"))

    min_grb = grb_df['E_LV_lower95 (GeV)'].min()
    valid_p = uhecr_df.loc[~uhecr_df['Excluded'], 'E_LV_p (GeV)']
    min_p   = valid_p.min() if not valid_p.empty else float('nan')
    FOM = min(min_grb, min_p) / 1e17

    df = pd.DataFrame([{
        'Min GRB Bound (1e17 GeV)': min_grb/1e17,
        'Min UHECR Bound (1e17 GeV)': min_p/1e17,
        'Overall FOM': FOM
    }])
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", "combined_fom.csv"), index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()