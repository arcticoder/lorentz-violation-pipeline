"""
UHECR Spectrum Analysis Script

This script builds the Ultra-High Energy Cosmic Ray (UHECR) spectrum from Pierre Auger Observatory data
using the sd_s38 signal size (S(1000) corrected to 38° zenith angle) as the primary energy estimator.

Energy calibration: E_primary = A * (sd_s38)^B
where A ≈ 1.49e17 eV and B ≈ 1.07 (Auger SD calibration)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Auger SD calibration parameters
A_CALIB = 1.49e17  # eV
B_CALIB = 1.07

def load_uhecr_data(filename):
    """Load UHECR data and calculate primary energies."""
    print(f"Loading {filename}...")
    df = pd.read_csv(f"data/{filename}")
    
    # Check for required columns
    if 'sd_s38' not in df.columns or 'sd_exposure' not in df.columns:
        raise ValueError(f"Missing required columns in {filename}")
    
    # Remove events with invalid sd_s38 or exposure
    df = df.dropna(subset=['sd_s38', 'sd_exposure'])
    df = df[df['sd_s38'] > 0]  # Signal must be positive
    df = df[df['sd_exposure'] > 0]  # Exposure must be positive
    
    # Calculate primary energy using Auger calibration
    df['E_primary_eV'] = A_CALIB * (df['sd_s38'] ** B_CALIB)
    df['log10_E_eV'] = np.log10(df['E_primary_eV'])
    
    print(f"  Loaded {len(df)} valid events")
    print(f"  Energy range: {df['E_primary_eV'].min():.2e} - {df['E_primary_eV'].max():.2e} eV")
    print(f"  Log10(E) range: {df['log10_E_eV'].min():.2f} - {df['log10_E_eV'].max():.2f}")
    
    return df

def build_spectrum(df, log_e_min=18.5, log_e_max=20.5, n_bins=20):
    """Build energy spectrum from event data."""
    
    # Define energy bins in log10(E/eV)
    log_e_bins = np.linspace(log_e_min, log_e_max, n_bins + 1)
    log_e_centers = (log_e_bins[:-1] + log_e_bins[1:]) / 2
    delta_log_e = log_e_bins[1] - log_e_bins[0]
    
    # Histogram events in energy bins
    counts, _ = np.histogram(df['log10_E_eV'], bins=log_e_bins)
    
    # Sum exposure for each bin
    exposure_per_bin = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (df['log10_E_eV'] >= log_e_bins[i]) & (df['log10_E_eV'] < log_e_bins[i+1])
        if mask.any():
            exposure_per_bin[i] = df.loc[mask, 'sd_exposure'].sum()
    
    # Calculate flux: J(E) = N / (exposure * ΔE)
    # Convert to standard units: events/(km²·sr·year·eV)
    E_centers_eV = 10**log_e_centers
    delta_E_eV = E_centers_eV * np.log(10) * delta_log_e  # dE = E * ln(10) * d(log E)
    
    # Flux calculation (assuming exposure is in km²·sr·year)
    flux = np.zeros(n_bins)
    flux_error = np.zeros(n_bins)
    
    valid_bins = (counts > 0) & (exposure_per_bin > 0)
    flux[valid_bins] = counts[valid_bins] / (exposure_per_bin[valid_bins] * delta_E_eV[valid_bins])
    
    # Poisson errors: sqrt(N) / (exposure * ΔE)
    flux_error[valid_bins] = np.sqrt(counts[valid_bins]) / (exposure_per_bin[valid_bins] * delta_E_eV[valid_bins])
    
    # Create spectrum dataframe
    spectrum_df = pd.DataFrame({
        'log10_E_eV': log_e_centers,
        'E_eV': E_centers_eV,
        'E_EeV': E_centers_eV / 1e18,  # Convert to EeV (10^18 eV)
        'counts': counts,
        'exposure': exposure_per_bin,
        'flux': flux,
        'flux_error': flux_error,
        'delta_E_eV': delta_E_eV
    })
    
    return spectrum_df

def plot_spectrum(spectrum_df, title="UHECR Energy Spectrum", save_path=None):
    """Plot the energy spectrum."""
    
    # Filter out empty bins
    valid = (spectrum_df['counts'] > 0) & (spectrum_df['flux'] > 0)
    df_plot = spectrum_df[valid].copy()
    
    if len(df_plot) == 0:
        print("No valid bins to plot!")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Main spectrum plot: E^3 * J(E) vs E
    E_EeV = df_plot['E_EeV']
    flux = df_plot['flux']
    flux_error = df_plot['flux_error']
    
    # Multiply by E^3 for presentation
    flux_E3 = flux * (df_plot['E_eV']**3)
    flux_E3_error = flux_error * (df_plot['E_eV']**3)
    
    ax1.errorbar(E_EeV, flux_E3, yerr=flux_E3_error, 
                 fmt='o', capsize=3, capthick=1, label='Data')
    ax1.set_xlabel('Energy [EeV]')
    ax1.set_ylabel('E³ × J(E) [eV² km⁻² sr⁻¹ year⁻¹]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title} - Multiplicity Weighted')
    ax1.legend()
    
    # Raw flux plot
    ax2.errorbar(E_EeV, flux, yerr=flux_error, 
                 fmt='s', capsize=3, capthick=1, label='Data', color='red')
    ax2.set_xlabel('Energy [EeV]')
    ax2.set_ylabel('J(E) [km⁻² sr⁻¹ year⁻¹ eV⁻¹]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'{title} - Differential Flux')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def analyze_dataset(filename, output_prefix):
    """Complete analysis for a single dataset."""
    
    print(f"\n{'='*50}")
    print(f"Analyzing {filename}")
    print(f"{'='*50}")
    
    # Load and process data
    df = load_uhecr_data(filename)
    
    # Build spectrum
    spectrum_df = build_spectrum(df)
    
    # Print summary statistics
    total_events = len(df)
    total_exposure = df['sd_exposure'].sum()
    energy_range = f"{df['E_primary_eV'].min():.2e} - {df['E_primary_eV'].max():.2e} eV"
    
    print(f"\nSummary Statistics:")
    print(f"  Total events: {total_events:,}")
    print(f"  Total exposure: {total_exposure:.2f}")
    print(f"  Energy range: {energy_range}")
    print(f"  Valid spectrum bins: {(spectrum_df['counts'] > 0).sum()}")
    
    # Save spectrum data
    output_file = f"data/uhecr/{output_prefix}_spectrum.csv"
    Path("data/uhecr").mkdir(exist_ok=True)
    spectrum_df.to_csv(output_file, index=False)
    print(f"  Spectrum saved to {output_file}")
    
    # Plot spectrum
    dataset_name = filename.replace('.csv', '').replace('dataSummary', '')
    plot_title = f"UHECR Spectrum - {dataset_name}"
    plot_path = f"data/uhecr/{output_prefix}_spectrum.png"
    plot_spectrum(spectrum_df, title=plot_title, save_path=plot_path)
    
    return df, spectrum_df

def main():
    """Main analysis function."""
    
    print("UHECR Energy Spectrum Analysis")
    print(f"Using Auger calibration: E = {A_CALIB:.2e} × (sd_s38)^{B_CALIB}")
    
    # Analyze both datasets
    datasets = [
        ('dataSummarySD1500.csv', 'sd1500'),
        ('dataSummarySD750.csv', 'sd750')
    ]
    
    all_spectra = {}
    
    for filename, prefix in datasets:
        try:
            df, spectrum_df = analyze_dataset(filename, prefix)
            all_spectra[prefix] = spectrum_df
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            continue
    
    # Combined analysis
    if len(all_spectra) > 1:
        print(f"\n{'='*50}")
        print("Combined Analysis")
        print(f"{'='*50}")
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green']
        for i, (name, spectrum_df) in enumerate(all_spectra.items()):
            valid = (spectrum_df['counts'] > 0) & (spectrum_df['flux'] > 0)
            df_plot = spectrum_df[valid]
            
            if len(df_plot) > 0:
                E_EeV = df_plot['E_EeV']
                flux_E3 = df_plot['flux'] * (df_plot['E_eV']**3)
                flux_E3_error = df_plot['flux_error'] * (df_plot['E_eV']**3)
                
                ax.errorbar(E_EeV, flux_E3, yerr=flux_E3_error,
                           fmt='o', capsize=3, label=f'{name.upper()} data', 
                           color=colors[i % len(colors)])
        
        ax.set_xlabel('Energy [EeV]')
        ax.set_ylabel('E³ × J(E) [eV² km⁻² sr⁻¹ year⁻¹]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title('UHECR Energy Spectrum Comparison')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('data/uhecr/spectrum_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"\n{'='*50}")
    print("Analysis complete!")
    print("Check the data/uhecr/ directory for output files.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
