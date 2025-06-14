import glob
import pandas as pd
import statsmodels.api as sm

D = 1e17  # dispersion factor in seconds

def analyze_grb(path, D=D):
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
    for path in glob.glob("data/grbs/*.csv"):
        est, lower = analyze_grb(path)
        results.append({
            'GRB file': path.split('/')[-1],
            'E_LV_est (GeV)': est,
            'E_LV_lower95 (GeV)': lower
        })
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    import pandas as pd
    main()