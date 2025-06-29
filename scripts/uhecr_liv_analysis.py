"""
UHECR LIV Threshold Analysis

This script uses the generated UHECR energy spectrum to calculate Lorentz Invariance Violation (LIV) 
thresholds based on the cosmic ray flux suppression at the highest energies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_spectrum(spectrum_file):
    """Load the generated UHECR spectrum."""
    df = pd.read_csv(spectrum_file)
    
    # Filter to valid bins with events
    df = df[df['counts'] > 0].copy()
    
    print(f"Loaded spectrum with {len(df)} energy bins")
    print(f"Energy range: {df['E_EeV'].min():.1f} - {df['E_EeV'].max():.1f} EeV")
    
    return df

def find_flux_suppression(spectrum_df, suppression_factor=3, min_energy_eev=10):
    """
    Find the energy where the flux is suppressed by a given factor.
    
    This is a simplified approach - in reality you'd fit the spectrum to 
    theoretical models and look for deviations.
    
    Updated to work with the available energy range from the calibrated data.
    """
    
    # Look at high energy bins above min_energy_eev
    high_e = spectrum_df[spectrum_df['E_EeV'] >= min_energy_eev].copy()
    
    if len(high_e) == 0:
        print(f"No data above {min_energy_eev} EeV")
        return None
    
    # Alternative approach: look for the steepest decline in flux
    if len(high_e) < 3:
        print(f"Insufficient data above {min_energy_eev} EeV for suppression analysis")
        return None
    
    # Find the energy where flux drops most significantly
    high_e = high_e.sort_values('E_EeV')
    flux_ratios = high_e['flux'].iloc[:-1].values / high_e['flux'].iloc[1:].values
    
    # Look for significant flux drops
    max_ratio_idx = np.argmax(flux_ratios)
    if flux_ratios[max_ratio_idx] >= suppression_factor:
        cutoff_energy = high_e['E_EeV'].iloc[max_ratio_idx + 1]
        print(f"Flux suppression by factor {flux_ratios[max_ratio_idx]:.1f} found at {cutoff_energy:.1f} EeV")
        return cutoff_energy
    else:
        # Use the highest energy with good statistics as an upper limit
        good_stats = high_e[high_e['counts'] >= 5]  # Require at least 5 events
        if len(good_stats) > 0:
            cutoff_energy = good_stats['E_EeV'].max()
            print(f"Using highest energy with good statistics: {cutoff_energy:.1f} EeV")
            print(f"(Conservative limit - no clear suppression found)")
            return cutoff_energy
        else:
            print(f"No clear flux suppression by factor {suppression_factor} found")
            return None

def estimate_liv_threshold(cutoff_energy_eev, n=1):
    """
    Estimate LIV energy scale from GZK cutoff suppression.
    
    For LIV modifications to photon dispersion:
    E_threshold ≈ (E_cutoff^n / E_Planck^(n-1))^(1/n)
    
    For n=1 (linear): E_LIV ≈ E_cutoff 
    For n=2 (quadratic): E_LIV ≈ sqrt(E_cutoff * E_Planck)
    """
    
    if cutoff_energy_eev is None:
        return None
    
    E_cutoff = cutoff_energy_eev * 1e18  # Convert to eV
    E_Planck = 1.22e19  # eV
    
    if n == 1:
        # Linear LIV
        E_LIV = E_cutoff
    elif n == 2:
        # Quadratic LIV  
        E_LIV = np.sqrt(E_cutoff * E_Planck)
    else:
        # General case
        E_LIV = (E_cutoff**n / E_Planck**(n-1))**(1/n)
    
    # Convert to GeV for consistency with other parts of pipeline
    E_LIV_GeV = E_LIV / 1e9
    
    print(f"Estimated LIV scale (n={n}): {E_LIV_GeV:.2e} GeV")
    
    return E_LIV_GeV

def create_uhecr_exclusion(spectrum_df, output_file="data/uhecr/uhecr_exclusion.csv"):
    """Create UHECR LIV exclusion data compatible with combined_fom.py"""
    
    # Try different suppression factors and LIV orders
    # Use lower energy threshold since our calibrated data has lower energies
    results = []
    
    for suppression in [2, 3, 5]:
        for n in [1, 2]:
            cutoff = find_flux_suppression(spectrum_df, suppression_factor=suppression, min_energy_eev=10)
            if cutoff is not None:
                E_LIV_GeV = estimate_liv_threshold(cutoff, n=n)
                if E_LIV_GeV is not None:
                    results.append({
                        'Suppression_Factor': suppression,
                        'LIV_Order': n,
                        'Cutoff_Energy_EeV': cutoff,
                        'E_LV_p (GeV)': E_LIV_GeV,
                        'Excluded': False,  # Assume these are valid bounds
                        'Method': f'Flux_Analysis_{suppression}x'
                    })
    
    # If no suppression found, create conservative limits based on highest energy
    if not results:
        max_energy_row = spectrum_df.loc[spectrum_df['E_EeV'].idxmax()]
        max_energy_eev = max_energy_row['E_EeV']
        
        print(f"No flux suppression found. Using conservative limit from max energy: {max_energy_eev:.1f} EeV")
        
        for n in [1, 2]:
            E_LIV_GeV = estimate_liv_threshold(max_energy_eev, n=n)
            if E_LIV_GeV is not None:
                results.append({
                    'Suppression_Factor': 1,  # Conservative - no suppression
                    'LIV_Order': n,
                    'Cutoff_Energy_EeV': max_energy_eev,
                    'E_LV_p (GeV)': E_LIV_GeV,
                    'Excluded': False,
                    'Method': 'Conservative_Upper_Limit'
                })
    
    if results:
        exclusion_df = pd.DataFrame(results)
        exclusion_df.to_csv(output_file, index=False)
        print(f"UHECR exclusion data saved to {output_file}")
        print(exclusion_df.to_string(index=False))
        return exclusion_df
    else:
        print("No valid LIV bounds found from UHECR spectrum")
        return None

def plot_spectrum_with_liv(spectrum_df, liv_bounds=None, save_path=None):
    """Plot spectrum with potential LIV threshold markers."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot spectrum
    valid = spectrum_df['flux'] > 0
    df_plot = spectrum_df[valid]
    
    E_EeV = df_plot['E_EeV']
    flux_E3 = df_plot['flux'] * (df_plot['E_eV']**3)
    flux_E3_error = df_plot['flux_error'] * (df_plot['E_eV']**3)
    
    ax.errorbar(E_EeV, flux_E3, yerr=flux_E3_error,
               fmt='o', capsize=3, label='UHECR Data', color='blue')
    
    # Add LIV threshold markers if available
    if liv_bounds is not None and len(liv_bounds) > 0:
        for _, bound in liv_bounds.iterrows():
            E_LIV_EeV = bound['E_LV_p (GeV)'] * 1e9 / 1e18  # Convert GeV to EeV
            if E_LIV_EeV < ax.get_xlim()[1]:  # Only plot if in range
                ax.axvline(E_LIV_EeV, color='red', linestyle='--', alpha=0.7,
                          label=f"LIV n={bound['LIV_Order']:.0f} ({bound['Method']})")
    
    # Add GZK cutoff reference
    ax.axvline(50, color='green', linestyle=':', alpha=0.7, label='Expected GZK Cutoff')
    
    ax.set_xlabel('Energy [EeV]')
    ax.set_ylabel('E³ × J(E) [eV² km⁻² sr⁻¹ year⁻¹]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('UHECR Spectrum with LIV Thresholds')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main analysis function."""
    
    print("UHECR LIV Threshold Analysis")
    print("="*50)
    
    # Load the SD1500 spectrum (most events)
    spectrum_file = "data/uhecr/sd1500_spectrum.csv"
    
    try:
        spectrum_df = load_spectrum(spectrum_file)
    except FileNotFoundError:
        print(f"Spectrum file {spectrum_file} not found.")
        print("Please run uhecr_spectrum.py first to generate the spectrum.")
        return
    
    # Create LIV exclusion bounds
    print("\nCalculating LIV exclusion bounds...")
    exclusion_df = create_uhecr_exclusion(spectrum_df)
    
    # Plot spectrum with LIV markers
    plot_path = "data/uhecr/uhecr_spectrum_with_liv.png"
    plot_spectrum_with_liv(spectrum_df, exclusion_df, save_path=plot_path)
    
    print("\nAnalysis complete!")
    print("UHECR exclusion data is now ready for combined_fom.py")

if __name__ == "__main__":
    main()
