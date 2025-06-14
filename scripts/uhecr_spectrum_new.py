#!/usr/bin/env python3
"""
UHECR Spectrum Analysis Script

This script builds the Ultra-High Energy Cosmic Ray (UHECR) spectrum from Pierre Auger Observatory data
using the sd_s38 signal size (S(1000) corrected to 38° zenith angle) as the primary energy estimator.

Key features:
- Logarithmic energy binning with proper exposure summation
- Flux calculation: J(E) = N / (exposure × ΔE) 
- Gehrels (1986) asymmetric Poisson errors and upper limits
- Publication-quality E³-weighted spectrum output

Energy calibration: E [eV] = 4.17×10¹⁶ × (sd_s38)^1.07
Reference: Latest Auger SD calibration (2020-2024)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaincinv
from pathlib import Path

# --- Configuration ---
INPUT_CSV = "data/dataSummarySD1500.csv"
OUTPUT_CSV = "data/uhecr/sd1500_spectrum.csv"
PLOT_FILE = "data/uhecr/sd1500_spectrum.png"

# Energy calibration constants
A_cal = 4.17e16  # eV
B_cal = 1.07

# Bin edges: log10(E/eV) from 17.5 to 20.0 in 25 bins
logE_min, logE_max, n_bins = 17.5, 20.0, 25
bin_edges = np.linspace(logE_min, logE_max, n_bins + 1)
bin_centers_log = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Gehrels (1986) upper & lower confidence limits for Poisson counts
def gehrels_errors(N, CL=0.84):
    """Calculate Gehrels (1986) asymmetric Poisson confidence intervals."""
    alpha = 1 - CL
    low = N - gammaincinv(N, alpha/2) if N>0 else 0.0
    high = gammaincinv(N+1, 1-alpha/2) - N
    return low, high

def analyze_dataset(input_file, output_prefix):
    """Analyze a single dataset and produce spectrum."""
    
    print(f"\nAnalyzing {input_file}")
    print("=" * 50)
    
    # --- Load & Calibrate ---
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} events from {input_file}")
    
    # Use sd_s38 as energy estimator
    sd38 = df['sd_s38'].values
    E = A_cal * (sd38 ** B_cal)  # in eV
    
    # Exposure per event (km² sr yr)
    expo = df['sd_exposure'].values  # assume this column holds per-event exposure
    
    print(f"Energy range: {E.min():.2e} - {E.max():.2e} eV")
    print(f"Energy range: {E.min()/1e18:.2f} - {E.max()/1e18:.1f} EeV")
    print(f"Total exposure: {expo.sum():.2e} km² sr yr")
    
    # --- Bin & Sum ---
    inds = np.digitize(np.log10(E), bin_edges) - 1
    results = []
    
    print(f"Binning into {n_bins} logarithmic energy bins...")
    
    for i in range(n_bins):
        in_bin = (inds == i)
        N = in_bin.sum()
        if N==0:
            # zero-count bin → upper limit only
            low_count, high_count = 0.0, gammaincinv(1, 1-0.84)
        else:
            low_count, high_count = gehrels_errors(N)
        
        # total exposure in bin
        expo_bin = expo[in_bin].sum()
        
        # bin width in eV: ΔE = E * ln(10) * Δ(logE)
        E_center = 10**bin_centers_log[i]
        dlogE = bin_edges[1] - bin_edges[0]
        dE = E_center * np.log(10) * dlogE
    
        # flux and errors [km⁻² sr⁻¹ yr⁻¹ eV⁻¹]
        flux      = N / (expo_bin * dE) if expo_bin>0 else 0.0
        err_low   = low_count / (expo_bin * dE) if expo_bin>0 else 0.0
        err_high  = high_count/ (expo_bin * dE) if expo_bin>0 else 0.0
    
        results.append({
            'logE_center': bin_centers_log[i],
            'E_center_eV': E_center,
            'E_center_EeV': E_center/1e18,
            'N_events': N,
            'expo_km2sr_yr': expo_bin,
            'delta_E_eV': dE,
            'flux': flux,
            'flux_err_low': err_low,
            'flux_err_high': err_high,
            'flux_E3': flux * (E_center**3),
            'flux_E3_err_low': err_low * (E_center**3),
            'flux_E3_err_high': err_high * (E_center**3),
            'valid_bin': N > 0,
            'high_statistics': N >= 10
        })
    
    df_spec = pd.DataFrame(results)
    
    # Save spectrum
    output_csv = f"data/uhecr/{output_prefix}_spectrum.csv"
    Path("data/uhecr").mkdir(exist_ok=True)
    df_spec.to_csv(output_csv, index=False)
    print(f"Spectrum saved to {output_csv}")
    
    # Print summary
    valid_bins = df_spec['valid_bin'].sum()
    high_stats_bins = df_spec['high_statistics'].sum()
    total_events = df_spec['N_events'].sum()
    
    print(f"\nSpectrum Summary:")
    print(f"  Valid bins with events: {valid_bins}/{n_bins}")
    print(f"  High statistics bins (≥10): {high_stats_bins}/{n_bins}")
    print(f"  Total events in spectrum: {total_events}")
    
    if valid_bins > 0:
        valid_df = df_spec[df_spec['valid_bin']]
        flux_range = f"{valid_df['flux'].min():.2e} - {valid_df['flux'].max():.2e}"
        print(f"  Flux range: {flux_range} km⁻² sr⁻¹ yr⁻¹ eV⁻¹")
    
    # --- Plotting ---
    plot_file = f"data/uhecr/{output_prefix}_spectrum.png"
    
    valid_data = df_spec[df_spec['valid_bin']]
    if len(valid_data) > 0:
        plt.figure(figsize=(12, 8))
        
        plt.errorbar(
            valid_data['E_center_eV'], valid_data['flux_E3'],
            yerr=[valid_data['flux_E3_err_low'], valid_data['flux_E3_err_high']],
            fmt='o', ms=6, capsize=4, capthick=1.5, 
            label=f'{output_prefix.upper()} data', color='blue'
        )
        
        # Mark high statistics bins
        high_stats = valid_data[valid_data['high_statistics']]
        if len(high_stats) > 0:
            plt.scatter(high_stats['E_center_eV'], high_stats['flux_E3'],
                       s=80, facecolors='none', edgecolors='red', linewidth=2,
                       label='High statistics (≥10 events)')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Energy (eV)', fontsize=12)
        plt.ylabel(r'$E^3\,J(E)\ \mathrm{[eV^2\,km^{-2}\,sr^{-1}\,yr^{-1}]}$', fontsize=12)
        plt.title(f'UHECR Spectrum - {output_prefix.upper()} (E³-weighted)', fontsize=14)
        plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
        plt.legend()
        
        # Add statistics text
        stats_text = f"Events: {total_events:,}\\n"
        stats_text += f"Valid bins: {valid_bins}/{n_bins}\\n"
        stats_text += f"Exposure: {expo.sum():.1e} km²sr yr"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {plot_file}")
    else:
        print("No valid data to plot!")
    
    return df_spec

def main():
    """Main analysis function."""
    
    print("UHECR Spectrum Analysis")
    print(f"Energy calibration: E [eV] = {A_cal:.2e} × (sd_s38)^{B_cal}")
    print(f"Logarithmic binning: {n_bins} bins from 10^{logE_min} to 10^{logE_max} eV")
    
    # Analyze datasets
    datasets = [
        ('data/dataSummarySD1500.csv', 'sd1500'),
        ('data/dataSummarySD750.csv', 'sd750')
    ]
    
    all_spectra = {}
    
    for filename, prefix in datasets:
        try:
            if Path(filename).exists():
                spectrum_df = analyze_dataset(filename, prefix)
                all_spectra[prefix] = spectrum_df
            else:
                print(f"File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            continue
    
    # Combined comparison plot
    if len(all_spectra) > 1:
        print(f"\nCreating comparison plot...")
        
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (name, spectrum_df) in enumerate(all_spectra.items()):
            valid_data = spectrum_df[spectrum_df['valid_bin']]
            
            if len(valid_data) > 0:
                plt.errorbar(
                    valid_data['E_center_eV'], valid_data['flux_E3'],
                    yerr=[valid_data['flux_E3_err_low'], valid_data['flux_E3_err_high']],
                    fmt='o', ms=5, capsize=3, label=f'{name.upper()} data',
                    color=colors[i % len(colors)]
                )
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Energy (eV)', fontsize=12)
        plt.ylabel(r'$E^3\,J(E)\ \mathrm{[eV^2\,km^{-2}\,sr^{-1}\,yr^{-1}]}$', fontsize=12)
        plt.title('UHECR Energy Spectrum Comparison', fontsize=14)
        plt.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        comparison_file = 'data/uhecr/spectrum_comparison.png'
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Comparison plot saved to {comparison_file}")
    
    print(f"\nAnalysis complete!")
    print("Check the data/uhecr/ directory for output files.")

if __name__ == "__main__":
    main()
