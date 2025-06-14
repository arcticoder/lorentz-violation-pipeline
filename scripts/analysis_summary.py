"""
UHECR Spectrum Analysis Summary

This script provides a comprehensive summary of the UHECR energy spectrum analysis
and Lorentz Invariance Violation (LIV) threshold calculations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def print_data_summary():
    """Print summary of available data files."""
    print("DATA SUMMARY")
    print("=" * 50)
    
    data_files = [
        ("dataSummarySD1500.csv", "SD1500"),
        ("dataSummarySD750.csv", "SD750"), 
        ("dataSummaryInclined.csv", "Inclined")
    ]
    
    for filename, description in data_files:
        try:
            df = pd.read_csv(f"data/{filename}")
            print(f"{description:12} : {len(df):6,} events")
            
            # Check for key columns
            has_s38 = 'sd_s38' in df.columns
            has_exposure = 'sd_exposure' in df.columns
            print(f"{'':14} sd_s38: {'✓' if has_s38 else '✗'}, sd_exposure: {'✓' if has_exposure else '✗'}")
            
            if has_s38:
                valid_s38 = df[df['sd_s38'] > 0]['sd_s38']
                if len(valid_s38) > 0:
                    print(f"{'':14} sd_s38 range: {valid_s38.min():.1f} - {valid_s38.max():.1f}")
                
        except FileNotFoundError:
            print(f"{description:12} : File not found")
        except Exception as e:
            print(f"{description:12} : Error - {e}")
    
    print()

def print_spectrum_summary():
    """Print summary of generated spectrum."""
    print("SPECTRUM ANALYSIS RESULTS")
    print("=" * 50)
    
    spectrum_files = [
        ("data/uhecr/sd1500_spectrum.csv", "SD1500 Spectrum"),
        ("data/uhecr/sd750_spectrum.csv", "SD750 Spectrum")
    ]
    
    for filename, description in spectrum_files:
        try:
            df = pd.read_csv(filename)
            valid_bins = df[df['counts'] > 0]
            
            print(f"{description}")
            print(f"  Total energy bins: {len(df)}")
            print(f"  Bins with events: {len(valid_bins)}")
            
            if len(valid_bins) > 0:
                print(f"  Energy range: {valid_bins['E_EeV'].min():.1f} - {valid_bins['E_EeV'].max():.1f} EeV")
                print(f"  Total events: {valid_bins['counts'].sum():,}")
                print(f"  Total exposure: {valid_bins['exposure'].sum():.1f}")
                
                # Find highest energy bin with good statistics (>10 events)
                good_stats = valid_bins[valid_bins['counts'] >= 10]
                if len(good_stats) > 0:
                    max_reliable_energy = good_stats['E_EeV'].max()
                    print(f"  Max reliable energy (≥10 events): {max_reliable_energy:.1f} EeV")
            
            print()
            
        except FileNotFoundError:
            print(f"{description}: File not found")
        except Exception as e:
            print(f"{description}: Error - {e}")

def print_liv_summary():
    """Print summary of LIV analysis."""
    print("LIV THRESHOLD ANALYSIS")
    print("=" * 50)
    
    try:
        df = pd.read_csv("data/uhecr/uhecr_exclusion.csv")
        
        print("Energy Calibration Used:")
        print("  E_primary = 1.49×10¹⁷ × (sd_s38)¹·⁰⁷ eV")
        print("  (Pierre Auger Observatory SD calibration)")
        print()
        
        print("LIV Bounds Derived:")
        for _, row in df.iterrows():
            liv_order = int(row['LIV_Order'])
            energy_gev = row['E_LV_p (GeV)']
            cutoff_eev = row['Cutoff_Energy_EeV']
            method = row['Method']
            
            print(f"  LIV Order n={liv_order}:")
            print(f"    E_LIV ≥ {energy_gev:.2e} GeV")
            print(f"    Based on flux suppression at {cutoff_eev:.1f} EeV")
            print(f"    Method: {method}")
            print()
            
        # Compare to theoretical expectations
        print("Theoretical Context:")
        print("  Planck scale: M_Pl ≈ 1.22×10¹⁹ eV ≈ 1.22×10¹⁰ GeV")
        print("  String scale: M_s ~ 10¹⁶-10¹⁸ GeV (model dependent)")
        print("  GZK cutoff: Expected around 50-60 EeV")
        print()
        
        # Show where our bounds fit
        min_bound = df['E_LV_p (GeV)'].min()
        planck_gev = 1.22e10
        
        if min_bound > planck_gev:
            print(f"  Our strongest bound ({min_bound:.1e} GeV) is above Planck scale")
            print("  → Suggests sub-Planckian LIV effects not detectable with current data")
        else:
            print(f"  Our strongest bound ({min_bound:.1e} GeV) is below Planck scale")
            print("  → Could constrain some quantum gravity models")
            
    except FileNotFoundError:
        print("LIV exclusion file not found. Run uhecr_liv_analysis.py first.")
    except Exception as e:
        print(f"Error loading LIV data: {e}")

def print_methodology():
    """Print explanation of the methodology."""
    print("METHODOLOGY")
    print("=" * 50)
    
    methodology = """
1. DATA PREPARATION:
   • Used Pierre Auger Observatory cosmic ray data
   • Selected events with valid sd_s38 (S(1000) @ 38°) and exposure
   • Applied energy calibration: E = 1.49×10¹⁷ × (sd_s38)¹·⁰⁷ eV

2. SPECTRUM CONSTRUCTION:
   • Binned events in log₁₀(E/eV) from 18.5 to 20.5 (20 bins)
   • Calculated flux: J(E) = N/(exposure × ΔE)
   • Estimated Poisson errors: σ = √N/(exposure × ΔE)

3. LIV THRESHOLD ESTIMATION:
   • Identified flux suppression at highest energies
   • Used suppression energy as proxy for LIV threshold
   • Applied theoretical relations:
     - Linear LIV (n=1): E_LIV ≈ E_cutoff
     - Quadratic LIV (n=2): E_LIV ≈ √(E_cutoff × M_Planck)

4. FIGURE OF MERIT:
   • Determined most stringent LIV bound from all methods
   • Compared with GRB bounds (when available)
   • Reported overall limit on Lorentz violation scale
"""
    print(methodology)

def create_summary_plots():
    """Create summary visualization."""
    try:
        # Load main spectrum
        df_spectrum = pd.read_csv("data/uhecr/sd1500_spectrum.csv")
        df_liv = pd.read_csv("data/uhecr/uhecr_exclusion.csv")
        
        valid = df_spectrum['counts'] > 0
        df_plot = df_spectrum[valid]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Energy spectrum
        E_EeV = df_plot['E_EeV']
        flux_E3 = df_plot['flux'] * (df_plot['E_eV']**3)
        flux_E3_error = df_plot['flux_error'] * (df_plot['E_eV']**3)
        
        ax1.errorbar(E_EeV, flux_E3, yerr=flux_E3_error,
                    fmt='o', capsize=3, label='UHECR Data', color='blue')
        
        # Add GZK cutoff and LIV threshold markers
        ax1.axvline(50, color='green', linestyle=':', alpha=0.7, label='Expected GZK Cutoff')
        
        for _, bound in df_liv.iterrows():
            cutoff_eev = bound['Cutoff_Energy_EeV']
            liv_order = int(bound['LIV_Order'])
            ax1.axvline(cutoff_eev, color='red', linestyle='--', alpha=0.7,
                       label=f'Flux Suppression (n={liv_order})')
        
        ax1.set_xlabel('Energy [EeV]')
        ax1.set_ylabel('E³ × J(E) [eV² km⁻² sr⁻¹ year⁻¹]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('UHECR Energy Spectrum')
        ax1.legend()
        
        # Plot 2: LIV bounds comparison
        energy_scales = ['Planck Scale\n(1.22×10¹⁰ GeV)', 'String Scale\n(~10¹⁶-10¹⁸ GeV)']
        energy_values = [1.22e10, 1e17]  # Representative string scale
        
        liv_bounds = df_liv['E_LV_p (GeV)'].values
        liv_labels = [f'LIV n={int(row["LIV_Order"])}' for _, row in df_liv.iterrows()]
        
        all_energies = list(energy_values) + list(liv_bounds)
        all_labels = energy_scales + liv_labels
        colors = ['green', 'orange'] + ['red'] * len(liv_bounds)
        
        y_pos = range(len(all_energies))
        
        ax2.barh(y_pos, all_energies, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(all_labels)
        ax2.set_xlabel('Energy Scale [GeV]')
        ax2.set_xscale('log')
        ax2.set_title('LIV Bounds vs Theoretical Scales')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/uhecr/analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Summary plot saved to data/uhecr/analysis_summary.png")
        
    except Exception as e:
        print(f"Could not create summary plots: {e}")

def main():
    """Main summary function."""
    print("UHECR LORENTZ VIOLATION ANALYSIS - COMPLETE SUMMARY")
    print("=" * 60)
    print()
    
    print_data_summary()
    print_spectrum_summary()
    print_liv_summary()
    print_methodology()
    
    print("VISUALIZATION")
    print("=" * 50)
    create_summary_plots()
    
    print("\nFILES GENERATED:")
    print("=" * 50)
    output_files = [
        "data/uhecr/sd1500_spectrum.csv",
        "data/uhecr/sd1500_spectrum.png", 
        "data/uhecr/sd750_spectrum.csv",
        "data/uhecr/sd750_spectrum.png",
        "data/uhecr/uhecr_exclusion.csv",
        "data/uhecr/uhecr_spectrum_with_liv.png",
        "data/uhecr/spectrum_comparison.png",
        "data/uhecr/analysis_summary.png",
        "data/combined_fom_summary.csv"
    ]
    
    for filename in output_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename}")
    
    print(f"\nAnalysis complete! Check the data/uhecr/ directory for all outputs.")

if __name__ == "__main__":
    main()
