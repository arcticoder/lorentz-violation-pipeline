import pandas as pd
import numpy as np

def simulate_uhecr(path):
    df = pd.read_csv(path)
    energies = df['energy_EeV'].values
    flux     = df['flux'].values
    error    = df['error'].values

    results = []
    for E_LV_p in np.logspace(17, 19, 5):
        # placeholder χ²: compare flux to its mean
        chi2 = np.sum(((flux - flux.mean())/error)**2) / len(flux)
        excluded = chi2 > 1.0
        results.append({
            'E_LV_p (GeV)': E_LV_p,
            'chi2': chi2,
            'Excluded': excluded
        })
    return pd.DataFrame(results)

def main():
    df = simulate_uhecr("data/uhecr/auger_spectrum.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()