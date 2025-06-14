#!/usr/bin/env python3
"""
Hidden Sector Energy Leakage in Polymer-QED

This module computes energy leakage into hidden sectors (dark photons, axions)
with polymer-QED modifications and tests observability against:
1. GRB time-delay measurements
2. Terrestrial precision tests (torsion balance, etc.)

Key Physics:
- Visible/total branching ratio: Γ_visible/Γ_total = f(E,μ,g_hidden)
- Modified dispersion: ω² = k²[1 + polymer corrections + hidden sector losses]
- Observable signatures: Additional time delays, energy-dependent attenuation

Hidden sector candidates:
- Dark photons (kinetic mixing): γ → γ' with coupling ε
- Axions (Primakoff effect): γ → a with coupling g_aγγ
- Extra dimensions: Energy leakage to bulk gravitons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Physical constants
ELECTRON_MASS_GEV = 0.511e-3    # GeV
ALPHA_EM = 1/137.036             # Fine structure constant
E_PLANCK_GEV = 1.22e19          # GeV
HBAR_C = 0.197                  # GeV·fm

class HiddenSectorAnalysis:
    """
    Calculate energy leakage into hidden sectors with polymer-QED corrections.
    
    Features:
    1. Dark photon kinetic mixing
    2. Axion-photon coupling (Primakoff effect)
    3. Extra-dimensional leakage
    4. Polymer-QED enhancement of hidden couplings
    """
    
    def __init__(self, polymer_scale_GeV=1e16):
        """
        Initialize hidden sector analysis.
        
        Parameters:
        -----------
        polymer_scale_GeV : float
            Polymer energy scale μ in GeV
        """
        self.polymer_scale = polymer_scale_GeV
        
    def dark_photon_mixing_rate(self, photon_energy_GeV, epsilon_mixing=1e-3):
        """
        Calculate γ → γ' conversion rate via kinetic mixing.
        
        Parameters:
        -----------
        photon_energy_GeV : float
            Photon energy in GeV
        epsilon_mixing : float
            Kinetic mixing parameter ε
            
        Returns:
        --------
        gamma_dark : float
            Conversion rate Γ(γ → γ') in GeV
        """
        # Standard kinetic mixing rate
        # Γ ~ ε² α E (approximate)
        standard_rate = epsilon_mixing**2 * ALPHA_EM * photon_energy_GeV
        
        return standard_rate
    
    def axion_conversion_rate(self, photon_energy_GeV, g_agamma=1e-10, B_field_T=1e-4):
        """
        Calculate γ → a conversion rate (Primakoff effect).
        
        Parameters:
        -----------
        photon_energy_GeV : float
            Photon energy in GeV
        g_agamma : float
            Axion-photon coupling in GeV⁻¹
        B_field_T : float
            Background magnetic field in Tesla
            
        Returns:
        --------
        gamma_axion : float
            Conversion rate Γ(γ → a) in GeV
        """
        # Primakoff conversion: Γ ~ g²_aγγ B² E / (16π)
        # Convert B field to natural units
        B_natural = B_field_T * 1.95e-20  # GeV² (rough conversion)
        
        axion_rate = (g_agamma**2 * B_natural * photon_energy_GeV) / (16 * np.pi)
        
        return axion_rate
    
    def extra_dimensional_leakage(self, photon_energy_GeV, n_extra=2, M_string_GeV=1e16):
        """
        Calculate energy leakage to extra dimensions.
        
        Parameters:
        -----------
        photon_energy_GeV : float
            Photon energy in GeV
        n_extra : int
            Number of extra dimensions
        M_string_GeV : float
            String scale in GeV
            
        Returns:
        --------
        gamma_extra : float
            Leakage rate to extra dimensions in GeV
        """
        # Extra-dimensional leakage: Γ ~ (E/M_s)^(n+2)
        energy_ratio = photon_energy_GeV / M_string_GeV
        extra_rate = ALPHA_EM * photon_energy_GeV * energy_ratio**(n_extra + 2)
        
        return extra_rate
    
    def polymer_enhancement_factor(self, photon_energy_GeV, model='linear'):
        """
        Polymer-QED enhancement of hidden sector couplings.
        
        The idea: Polymer modifications can enhance or suppress
        hidden sector interactions beyond standard field theory.
        """
        x = photon_energy_GeV / self.polymer_scale
        
        if model == 'linear':
            # Linear enhancement: g_eff = g_0 (1 + αx)
            alpha = 0.1
            enhancement = 1 + alpha * x
            
        elif model == 'quadratic':
            # Quadratic: g_eff = g_0 (1 + αx + βx²)
            alpha, beta = 0.1, 0.01
            enhancement = 1 + alpha * x + beta * x**2
            
        elif model == 'threshold':
            # Threshold activation above polymer scale
            if photon_energy_GeV > self.polymer_scale:
                enhancement = (photon_energy_GeV / self.polymer_scale)**0.5
            else:
                enhancement = 1.0
                
        elif model == 'suppression':
            # Exponential suppression
            enhancement = np.exp(-x)
            
        else:
            enhancement = 1.0
            
        return enhancement
    
    def total_hidden_sector_rate(self, photon_energy_GeV, hidden_params=None):
        """
        Calculate total energy leakage rate to all hidden sectors.
        
        Parameters:
        -----------
        photon_energy_GeV : float
            Photon energy in GeV
        hidden_params : dict
            Hidden sector parameters
            
        Returns:
        --------
        rates : dict
            Breakdown of leakage rates by mechanism
        """
        if hidden_params is None:
            hidden_params = {
                'epsilon_dark': 1e-3,      # Dark photon mixing
                'g_axion': 1e-10,          # Axion coupling GeV⁻¹
                'B_field_T': 1e-4,         # Background B field
                'n_extra': 2,              # Extra dimensions
                'M_string_GeV': 1e16,      # String scale
                'polymer_model': 'linear'   # Polymer enhancement
            }
        
        # Get polymer enhancement
        enhancement = self.polymer_enhancement_factor(
            photon_energy_GeV, 
            hidden_params['polymer_model']
        )
        
        # Calculate individual rates
        gamma_dark = self.dark_photon_mixing_rate(
            photon_energy_GeV, 
            hidden_params['epsilon_dark']
        ) * enhancement
        
        gamma_axion = self.axion_conversion_rate(
            photon_energy_GeV,
            hidden_params['g_axion'],
            hidden_params['B_field_T']
        ) * enhancement
        
        gamma_extra = self.extra_dimensional_leakage(
            photon_energy_GeV,
            hidden_params['n_extra'],
            hidden_params['M_string_GeV']
        ) * enhancement
        
        # Total hidden sector rate
        gamma_total_hidden = gamma_dark + gamma_axion + gamma_extra
        
        return {
            'dark_photon': gamma_dark,
            'axion': gamma_axion,
            'extra_dim': gamma_extra,
            'total_hidden': gamma_total_hidden,
            'enhancement_factor': enhancement
        }
    
    def visible_fraction(self, photon_energy_GeV, hidden_params=None):
        """
        Calculate branching ratio Γ_visible/Γ_total.
        
        This determines what fraction of photon energy remains
        in the visible sector vs leaks to hidden sectors.
        """
        hidden_rates = self.total_hidden_sector_rate(photon_energy_GeV, hidden_params)
        
        # Standard electromagnetic rate (approximate)
        gamma_visible = ALPHA_EM * photon_energy_GeV  # Thomson scattering, etc.
        
        # Total rate
        gamma_total = gamma_visible + hidden_rates['total_hidden']
        
        # Visible fraction
        if gamma_total > 0:
            visible_fraction = gamma_visible / gamma_total
        else:
            visible_fraction = 1.0
            
        return {
            'visible_fraction': visible_fraction,
            'hidden_fraction': 1 - visible_fraction,
            'gamma_visible': gamma_visible,
            'gamma_total': gamma_total,
            'hidden_breakdown': hidden_rates
        }
    
    def modified_dispersion_relation(self, photon_energy_GeV, hidden_params=None):
        """
        Calculate modified dispersion relation including hidden sector losses.
        
        ω² = k²[1 + polymer corrections - i×hidden_losses]
        
        The imaginary part leads to energy-dependent attenuation.
        """
        # Get visible fraction
        result = self.visible_fraction(photon_energy_GeV, hidden_params)
        
        # Energy-dependent attenuation coefficient
        attenuation = -np.log(result['visible_fraction'])  # Natural log
        
        # Modified group velocity (real part)
        polymer_velocity_correction = (photon_energy_GeV / self.polymer_scale) * 0.1
        
        # Effective dispersion parameters
        return {
            'energy_GeV': photon_energy_GeV,
            'attenuation_coefficient': attenuation,
            'velocity_correction': polymer_velocity_correction,
            'visible_fraction': result['visible_fraction'],
            'effective_coupling': result['gamma_total'] / (ALPHA_EM * photon_energy_GeV)
        }

def scan_hidden_sector_parameter_space():
    """Comprehensive scan of hidden sector parameter space."""
    print("HIDDEN SECTOR ENERGY LEAKAGE PARAMETER SCAN")
    print("=" * 60)
    
    # Parameter ranges to scan
    polymer_scales = np.logspace(12, 19, 20)  # 10¹² to 10¹⁹ GeV
    photon_energies = np.logspace(-3, 3, 30)  # 10⁻³ to 10³ GeV (meV to TeV)
    
    # Hidden sector coupling ranges
    epsilon_values = np.logspace(-6, -1, 10)  # Dark photon mixing
    g_axion_values = np.logspace(-12, -8, 10)  # Axion coupling
    
    results = []
    
    print("Scanning parameter combinations...")
    print(f"Polymer scales: {len(polymer_scales)}")
    print(f"Photon energies: {len(photon_energies)}")
    print(f"Dark photon couplings: {len(epsilon_values)}")
    print(f"Axion couplings: {len(g_axion_values)}")
    
    for i, mu in enumerate(polymer_scales):
        if i % 5 == 0:
            print(f"  Processing μ = {mu:.1e} GeV...")
            
        analyzer = HiddenSectorAnalysis(mu)
        
        for E_gamma in photon_energies:
            for epsilon in epsilon_values:
                for g_axion in g_axion_values:
                    
                    hidden_params = {
                        'epsilon_dark': epsilon,
                        'g_axion': g_axion,
                        'B_field_T': 1e-4,
                        'n_extra': 2,
                        'M_string_GeV': 1e16,
                        'polymer_model': 'linear'
                    }
                    
                    # Calculate visible fraction
                    result = analyzer.visible_fraction(E_gamma, hidden_params)
                    
                    # Store results
                    results.append({
                        'polymer_scale_GeV': mu,
                        'photon_energy_GeV': E_gamma,
                        'epsilon_dark': epsilon,
                        'g_axion_GeV_inv': g_axion,
                        'visible_fraction': result['visible_fraction'],
                        'hidden_fraction': result['hidden_fraction'],
                        'dark_rate': result['hidden_breakdown']['dark_photon'],
                        'axion_rate': result['hidden_breakdown']['axion'],
                        'extra_dim_rate': result['hidden_breakdown']['extra_dim'],
                        'enhancement_factor': result['hidden_breakdown']['enhancement_factor']
                    })
    
    print(f"Scan complete! Generated {len(results)} parameter combinations.")
    return pd.DataFrame(results)

def analyze_grb_constraints(scan_results):
    """Analyze constraints from GRB time-delay measurements."""
    print(f"\n{'=' * 60}")
    print("GRB TIME-DELAY CONSTRAINTS ON HIDDEN SECTORS")
    print(f"{'=' * 60}")
    
    # Typical GRB parameters
    grb_energies = [0.1, 1.0, 10.0, 100.0]  # GeV
    grb_distance = 1e17  # seconds (typical time-delay factor)
    
    print("Energy (GeV) | Visible Fraction | Hidden Loss | Time Delay (s)")
    print("-" * 65)
    
    # Find most extreme hidden sector cases
    extreme_cases = scan_results[scan_results['hidden_fraction'] > 0.01]  # >1% loss
    
    for E_grb in grb_energies:
        # Find cases near this energy
        energy_mask = np.abs(np.log10(extreme_cases['photon_energy_GeV']) - np.log10(E_grb)) < 0.2
        energy_cases = extreme_cases[energy_mask]
        
        if len(energy_cases) > 0:
            # Find case with maximum hidden loss
            worst_case = energy_cases.loc[energy_cases['hidden_fraction'].idxmax()]
            
            visible_frac = worst_case['visible_fraction']
            hidden_loss = worst_case['hidden_fraction']
            
            # Additional time delay from hidden sector interactions
            # Δt ≈ (1/v_eff - 1/c) × distance ≈ hidden_loss × distance/c
            additional_delay = hidden_loss * grb_distance
            
            print(f"{E_grb:8.1f}     | {visible_frac:13.6f} | {hidden_loss:10.6f} | {additional_delay:12.3e}")
        else:
            print(f"{E_grb:8.1f}     | {1.0:13.6f} | {0.0:10.6f} | {0.0:12.3e}")
    
    # Summary of GRB observability
    observable_cases = extreme_cases[extreme_cases['hidden_fraction'] > 0.1]  # >10% loss
    
    print(f"\nGRB Observability Summary:")
    print(f"Cases with >1% hidden loss: {len(extreme_cases)}")
    print(f"Cases with >10% hidden loss: {len(observable_cases)}")
    
    if len(observable_cases) > 0:
        print("🎯 Potentially observable with GRB time-delay measurements!")
        best_case = observable_cases.loc[observable_cases['hidden_fraction'].idxmax()]
        print(f"Best case: μ = {best_case['polymer_scale_GeV']:.1e} GeV")
        print(f"          ε = {best_case['epsilon_dark']:.1e}")
        print(f"          g = {best_case['g_axion_GeV_inv']:.1e} GeV⁻¹")
    else:
        print("❌ Hidden sector effects below GRB sensitivity")

def analyze_terrestrial_constraints(scan_results):
    """Analyze constraints from terrestrial precision measurements."""
    print(f"\n{'=' * 60}")
    print("TERRESTRIAL PRECISION TEST CONSTRAINTS")
    print(f"{'=' * 60}")
    
    # Terrestrial test energy scales
    terrestrial_tests = {
        'Torsion balance': 1e-6,      # GeV (~ meV)
        'Casimir force': 1e-3,        # GeV (~ eV)
        'Atomic spectroscopy': 1e-2,  # GeV (~ 10 eV)
        'Laboratory laser': 1e0      # GeV
    }
    
    print("Test Type            | Energy Scale | Max Hidden Loss | Observable?")
    print("-" * 70)
    
    for test_name, E_test in terrestrial_tests.items():
        # Find cases near this energy scale
        energy_mask = np.abs(np.log10(scan_results['photon_energy_GeV']) - np.log10(E_test)) < 0.3
        energy_cases = scan_results[energy_mask]
        
        if len(energy_cases) > 0:
            max_hidden = energy_cases['hidden_fraction'].max()
            observable = "Yes" if max_hidden > 1e-6 else "No"  # 1 ppm sensitivity
            
            print(f"{test_name:20} | {E_test:10.1e} | {max_hidden:14.3e} | {observable}")
        else:
            print(f"{test_name:20} | {E_test:10.1e} | {'N/A':>14} | No")
    
    # Torsion balance sensitivity analysis
    torsion_cases = scan_results[
        (scan_results['photon_energy_GeV'] > 1e-7) & 
        (scan_results['photon_energy_GeV'] < 1e-5)
    ]
    
    if len(torsion_cases) > 0:
        significant_cases = torsion_cases[torsion_cases['hidden_fraction'] > 1e-9]
        
        if len(significant_cases) > 0:
            print(f"\n🔬 Torsion balance observable cases: {len(significant_cases)}")
            best_torsion = significant_cases.loc[significant_cases['hidden_fraction'].idxmax()]
            print(f"Best sensitivity: {best_torsion['hidden_fraction']:.2e} at E = {best_torsion['photon_energy_GeV']:.1e} GeV")
        else:
            print(f"\n❌ Hidden sector effects below torsion balance sensitivity")

def create_hidden_sector_plots(scan_results):
    """Create comprehensive hidden sector analysis plots."""
    print(f"\n{'=' * 60}")
    print("GENERATING HIDDEN SECTOR ANALYSIS PLOTS")
    print(f"{'=' * 60}")
    
    os.makedirs('results', exist_ok=True)
    
    # Plot 1: Visible fraction vs energy for different polymer scales
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select representative polymer scales
    mu_values = [1e14, 1e16, 1e18]
    colors = ['blue', 'red', 'green']
    
    # Plot visible fraction vs photon energy
    ax = axes[0, 0]
    for i, mu in enumerate(mu_values):
        mu_data = scan_results[scan_results['polymer_scale_GeV'] == mu]
        if len(mu_data) > 0:
            ax.semilogx(mu_data['photon_energy_GeV'], mu_data['visible_fraction'], 
                       'o-', color=colors[i], alpha=0.6, label=f'μ = {mu:.0e} GeV')
    
    ax.set_xlabel('Photon Energy (GeV)')
    ax.set_ylabel('Visible Fraction')
    ax.set_title('Energy-Dependent Hidden Sector Leakage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Hidden fraction vs dark photon coupling
    ax = axes[0, 1]
    epsilon_unique = np.sort(scan_results['epsilon_dark'].unique())
    max_hidden_by_epsilon = []
    
    for eps in epsilon_unique:
        eps_data = scan_results[scan_results['epsilon_dark'] == eps]
        max_hidden_by_epsilon.append(eps_data['hidden_fraction'].max())
    
    ax.loglog(epsilon_unique, max_hidden_by_epsilon, 'bo-')
    ax.set_xlabel('Dark Photon Mixing ε')
    ax.set_ylabel('Maximum Hidden Fraction')
    ax.set_title('Dark Photon Coupling Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Hidden fraction vs axion coupling
    ax = axes[1, 0]
    g_unique = np.sort(scan_results['g_axion_GeV_inv'].unique())
    max_hidden_by_axion = []
    
    for g in g_unique:
        g_data = scan_results[scan_results['g_axion_GeV_inv'] == g]
        max_hidden_by_axion.append(g_data['hidden_fraction'].max())
    
    ax.loglog(g_unique, max_hidden_by_axion, 'ro-')
    ax.set_xlabel('Axion Coupling g_aγγ (GeV⁻¹)')
    ax.set_ylabel('Maximum Hidden Fraction')
    ax.set_title('Axion Coupling Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Parameter space overview
    ax = axes[1, 1]
    
    # Create 2D histogram of observable cases
    observable_data = scan_results[scan_results['hidden_fraction'] > 0.01]
    
    if len(observable_data) > 0:
        ax.scatter(np.log10(observable_data['polymer_scale_GeV']), 
                  np.log10(observable_data['photon_energy_GeV']),
                  c=observable_data['hidden_fraction'], 
                  cmap='viridis', alpha=0.6)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Hidden Fraction')
    
    ax.set_xlabel('log₁₀(Polymer Scale μ [GeV])')
    ax.set_ylabel('log₁₀(Photon Energy [GeV])')
    ax.set_title('Observable Hidden Sector Parameter Space')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/hidden_sector_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Hidden sector analysis plots saved: results/hidden_sector_analysis.png")

def main():
    """Run comprehensive hidden sector energy leakage analysis."""
    print("HIDDEN SECTOR ENERGY LEAKAGE ANALYSIS")
    print("=" * 60)
    print("Testing polymer-QED coupling to dark photons, axions, and extra dimensions")
    print("Constraints from GRB time delays and terrestrial precision measurements")
    print("=" * 60)
    
    # Perform parameter space scan
    scan_results = scan_hidden_sector_parameter_space()
    
    # Save raw results
    os.makedirs('results', exist_ok=True)
    scan_results.to_csv('results/hidden_sector_scan.csv', index=False)
    
    # Analyze GRB constraints
    analyze_grb_constraints(scan_results)
    
    # Analyze terrestrial constraints
    analyze_terrestrial_constraints(scan_results)
    
    # Create visualizations
    create_hidden_sector_plots(scan_results)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("HIDDEN SECTOR ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    
    total_cases = len(scan_results)
    observable_grb = len(scan_results[scan_results['hidden_fraction'] > 0.01])
    observable_terrestrial = len(scan_results[scan_results['hidden_fraction'] > 1e-6])
    
    print(f"Total parameter combinations: {total_cases}")
    print(f"GRB-observable cases (>1% loss): {observable_grb}")
    print(f"Terrestrial-observable cases (>1 ppm): {observable_terrestrial}")
    
    print(f"\nResults saved:")
    print(f"  - results/hidden_sector_scan.csv")
    print(f"  - results/hidden_sector_analysis.png")
    
    return scan_results

if __name__ == "__main__":
    results = main()
