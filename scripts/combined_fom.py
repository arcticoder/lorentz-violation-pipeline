import pandas as pd

def main():
    # Load GRB bounds
    df_grb = pd.read_csv("data/grbs/grb_bounds.csv")  # or capture analyze_grb output
    # Load UHECR exclusions
    df_uhecr = pd.read_csv("data/uhecr/uhecr_exclusion.csv")

    min_grb = df_grb['E_LV_lower95 (GeV)'].min()
    valid_p = df_uhecr.loc[~df_uhecr['Excluded'], 'E_LV_p (GeV)']
    min_p   = valid_p.min() if not valid_p.empty else float('nan')

    FOM = min(min_grb, min_p) / 1e17
    df = pd.DataFrame([{
        'Min GRB Bound (1e17 GeV)': min_grb/1e17,
        'Min UHECR Bound (1e17 GeV)': min_p/1e17,
        'Overall FOM': FOM
    }])
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()