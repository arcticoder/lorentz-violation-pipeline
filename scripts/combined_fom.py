import pandas as pd
import numpy as np

def main():
    print("Combined Figure of Merit Analysis")
    print("=" * 40)
    
    # Load GRB bounds
    print("Loading GRB bounds...")
    try:
        df_grb = pd.read_csv("data/grbs/grb_bounds.csv")
        print(f"  Loaded {len(df_grb)} GRB bounds")
    except FileNotFoundError:
        print("  GRB bounds file not found. Run analyze_grb.py first.")
        df_grb = pd.DataFrame()
    
    # Load UHECR exclusions
    print("Loading UHECR bounds...")
    try:
        df_uhecr = pd.read_csv("data/uhecr/uhecr_exclusion.csv")
        print(f"  Loaded {len(df_uhecr)} UHECR bounds")
    except FileNotFoundError:
        print("  UHECR bounds file not found. Run uhecr_liv_analysis.py first.")
        df_uhecr = pd.DataFrame()    
    # Calculate combined figure of merit
    results = []
    
    if not df_grb.empty and 'E_LV_lower95 (GeV)' in df_grb.columns:
        min_grb = df_grb['E_LV_lower95 (GeV)'].min()
        results.append(f"Min GRB Bound: {min_grb:.2e} GeV")
    else:
        min_grb = float('inf')
        results.append("Min GRB Bound: No data available")
    
    if not df_uhecr.empty and 'E_LV_p (GeV)' in df_uhecr.columns:
        valid_p = df_uhecr.loc[~df_uhecr['Excluded'], 'E_LV_p (GeV)']
        if not valid_p.empty:
            min_p = valid_p.min()
            results.append(f"Min UHECR Bound: {min_p:.2e} GeV")
            
            # Show all UHECR bounds for context
            print(f"\nAll UHECR bounds:")
            for _, row in df_uhecr[~df_uhecr['Excluded']].iterrows():
                print(f"  LIV order {row['LIV_Order']}: {row['E_LV_p (GeV)']:.2e} GeV ({row['Method']})")
        else:
            min_p = float('inf')
            results.append("Min UHECR Bound: All bounds excluded")
    else:
        min_p = float('inf')
        results.append("Min UHECR Bound: No data available")
    
    # Overall figure of merit (most stringent bound)
    if min_grb != float('inf') and min_p != float('inf'):
        FOM = min(min_grb, min_p)
        results.append(f"Overall FOM: {FOM:.2e} GeV")
        
        # Determine which method gives the most stringent bound
        if min_grb < min_p:
            results.append("Most stringent: GRB analysis")
        else:
            results.append("Most stringent: UHECR analysis")
    elif min_grb != float('inf'):
        FOM = min_grb
        results.append(f"Overall FOM: {FOM:.2e} GeV (GRB only)")
    elif min_p != float('inf'):
        FOM = min_p
        results.append(f"Overall FOM: {FOM:.2e} GeV (UHECR only)")
    else:
        results.append("Overall FOM: No valid bounds available")
        FOM = np.nan
    
    # Print results
    print("\nResults:")
    print("-" * 40)
    for result in results:
        print(f"  {result}")
    
    # Create summary DataFrame with better units
    summary_data = {
        'Min GRB Bound (GeV)': [min_grb if min_grb != float('inf') else np.nan],
        'Min UHECR Bound (GeV)': [min_p if min_p != float('inf') else np.nan],
        'Overall FOM (GeV)': [FOM if not np.isnan(FOM) and FOM != float('inf') else np.nan]
    }
    
    df_summary = pd.DataFrame(summary_data)
    print(f"\nSummary Table:")
    print(df_summary.to_string(index=False, float_format='%.2e'))
    
    # Save summary
    try:
        df_summary.to_csv("data/combined_fom_summary.csv", index=False)
        print(f"\nSummary saved to data/combined_fom_summary.csv")
    except Exception as e:
        print(f"Warning: Could not save summary: {e}")

if __name__ == "__main__":
    main()