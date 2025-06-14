#!/usr/bin/env python3
"""
Vacuum Instability Rates in Polymer-QED

This module computes Schwinger-like pair production rates with polymer-QED
corrections, testing whether LIV models predict vacuum breakdown at 
laboratory-accessible field strengths.

Key Physics:
- Standard Schwinger rate: Γ = exp[-π m²/(eE)]
- Polymer-QED correction: Γ_poly = exp[-π m²/(eE) × f(μ,E)]
- Critical field: E_crit = m²/(e) ≈ 1.3 × 10¹⁶ V/m

The polymer correction factor f(μ,E) modifies the vacuum structure,
potentially making pair production observable at lower field strengths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.special import factorial
from scipy.optimize import minimize_scalar

# Physical constants
ELECTRON_MASS = 0.511e-3  # GeV
ELECTRON_CHARGE = 4.8e-10  # esu (Gaussian units)
ALPHA_EM = 1/137.036  # Fine structure constant
HBAR_C = 0.197  # GeV·fm
E_PLANCK = 1.22e19  # GeV
E_CRITICAL_SCHWINGER = ELECTRON_MASS**2 / (ELECTRON_CHARGE * HBAR_C)  # Critical field

# Unit conversions
GEV_TO_V_PER_M = 5.1e15  # Conversion factor
V_PER_M_TO_GEV = 1/GEV_TO_V_PER_M

class VacuumInstabilityCalculator:
    """
    Calculate vacuum instability rates with polymer-QED corrections.
    
    Features:
    1. Standard Schwinger rate computation
    2. Polymer-QED modified rates with scale μ
    3. Laboratory field strength scanning
    4. Critical parameter identification
    """
    
    def __init__(self):
        self.results = []
        
    def schwinger_rate_standard(self, E_field):
        """
        Standard Schwinger pair production rate.
        
        Parameters:
        -----------
        E_field : float or array
            Electric field strength in V/m
            
        Returns:
        --------
        gamma : float or array
            Pair production rate (dimensionless exponent)
        """
        # Convert to natural units (GeV scale)
        E_natural = E_field * V_PER_M_TO_GEV
        
        # Schwinger rate: Γ = exp[-π m²/(eE)]
        exponent = -np.pi * ELECTRON_MASS**2 / (ALPHA_EM * E_natural)
        
        return exponent
    
    def polymer_correction_factor(self, mu_scale, E_field, model='linear'):
        """
        Polymer-QED correction factor f(μ,E).
        
        Parameters:
        -----------
        mu_scale : float
            Polymer energy scale in GeV
        E_field : float or array  
            Electric field strength in V/m
        model : str
            Type of polymer correction ('linear', 'quadratic', 'exponential')
            
        Returns:
        --------
        f_factor : float or array
            Polymer correction factor
        """
        # Convert field to energy scale
        E_natural = E_field * V_PER_M_TO_GEV
        
        # Dimensionless parameter
        x = E_natural / mu_scale
        
        if model == 'linear':
            # Linear polymer correction: f(x) = 1 + αx
            alpha = 0.1  # Polymer coupling
            f_factor = 1 + alpha * x
            
        elif model == 'quadratic':
            # Quadratic correction: f(x) = 1 + αx + βx²
            alpha, beta = 0.1, 0.01
            f_factor = 1 + alpha * x + beta * x**2
            
        elif model == 'exponential':
            # Exponential suppression: f(x) = exp(-x/x₀)
            x0 = 1.0  # Characteristic scale
            f_factor = np.exp(-x / x0)
            
        elif model == 'rainbow':
            # Gravity-rainbow type: f(x) = (1 + ηx)^(-n)
            eta, n = 1.0, 1.0
            f_factor = (1 + eta * x)**(-n)
            
        else:
            f_factor = np.ones_like(x)
            
        return f_factor
    
    def schwinger_rate_polymer(self, E_field, mu_scale, model='linear'):
        """
        Polymer-QED modified Schwinger rate.
        
        Parameters:
        -----------
        E_field : float or array
            Electric field strength in V/m
        mu_scale : float
            Polymer energy scale in GeV
        model : str
            Type of polymer correction
            
        Returns:
        --------
        gamma_poly : float or array
            Modified pair production rate exponent
        """
        # Standard Schwinger exponent
        gamma_standard = self.schwinger_rate_standard(E_field)
        
        # Polymer correction factor
        f_factor = self.polymer_correction_factor(mu_scale, E_field, model)
        
        # Modified rate: Γ_poly = exp[γ_standard × f(μ,E)]
        gamma_poly = gamma_standard * f_factor
        
        return gamma_poly
    
    def find_critical_fields(self, mu_scale, model='linear', threshold=-50):
        """
        Find field strengths where vacuum instability becomes significant.
        
        Parameters:
        -----------
        mu_scale : float
            Polymer energy scale in GeV
        model : str
            Type of polymer correction  
        threshold : float
            Log threshold for "observable" rates (e.g., -50 means exp(-50))
            
        Returns:
        --------
        critical_fields : dict
            Dictionary with critical field information
        """
        # Scan field strengths from laboratory to cosmic scales
        E_fields = np.logspace(6, 20, 1000)  # 10⁶ to 10²⁰ V/m
        
        # Calculate rates
        gamma_standard = self.schwinger_rate_standard(E_fields)
        gamma_polymer = self.schwinger_rate_polymer(E_fields, mu_scale, model)
        
        # Find where rates exceed threshold
        observable_standard = E_fields[gamma_standard > threshold]
        observable_polymer = E_fields[gamma_polymer > threshold]
        
        results = {
            'mu_scale_GeV': mu_scale,
            'model': model,
            'threshold': threshold,
            'E_critical_standard_V_per_m': observable_standard[0] if len(observable_standard) > 0 else np.inf,
            'E_critical_polymer_V_per_m': observable_polymer[0] if len(observable_polymer) > 0 else np.inf,
            'enhancement_factor': (observable_standard[0] / observable_polymer[0]) if len(observable_polymer) > 0 and len(observable_standard) > 0 else 1.0
        }
        
        return results
    
    def scan_parameter_space(self, mu_scales=None, models=None, thresholds=None):
        """
        Comprehensive scan of polymer parameter space.
        
        Parameters:
        -----------
        mu_scales : array, optional
            Polymer energy scales to test (GeV)
        models : list, optional
            Polymer correction models to test
        thresholds : array, optional
            Rate thresholds to consider
            
        Returns:
        --------
        scan_results : DataFrame
            Complete parameter scan results
        """
        if mu_scales is None:
            mu_scales = np.logspace(10, 19, 20)  # 10¹⁰ to 10¹⁹ GeV
            
        if models is None:
            models = ['linear', 'quadratic', 'exponential', 'rainbow']
            
        if thresholds is None:
            thresholds = [-100, -50, -20, -10]  # Various observability thresholds
        
        results = []
        
        print("Scanning Polymer-QED Parameter Space for Vacuum Instabilities")
        print("=" * 70)
        
        for model in models:
            print(f"\nModel: {model.upper()}")
            print("-" * 30)
            
            for threshold in thresholds:
                print(f"  Threshold: exp({threshold})")
                
                observable_count = 0
                min_critical_field = np.inf
                best_mu = None
                
                for mu in mu_scales:
                    result = self.find_critical_fields(mu, model, threshold)
                    
                    if result['E_critical_polymer_V_per_m'] < np.inf:
                        observable_count += 1
                        
                        if result['E_critical_polymer_V_per_m'] < min_critical_field:
                            min_critical_field = result['E_critical_polymer_V_per_m']
                            best_mu = mu
                    
                    result['threshold'] = threshold
                    results.append(result)
                
                print(f"    Observable cases: {observable_count}/{len(mu_scales)}")
                if best_mu is not None:
                    print(f"    Best μ = {best_mu:.2e} GeV → E_crit = {min_critical_field:.2e} V/m")
        
        return pd.DataFrame(results)
    
    def laboratory_accessibility_analysis(self):
        """
        Analyze whether polymer-QED vacuum instabilities are accessible
        in laboratory or astrophysical environments.
        """
        print("\nLABORATORY ACCESSIBILITY ANALYSIS")
        print("=" * 50)
        
        # Define field strength scales
        field_scales = {
            'Laboratory (strong laser)': 1e13,      # V/m
            'Laboratory (extreme)': 1e15,          # V/m  
            'Neutron star surface': 1e12,          # V/m
            'Neutron star magnetosphere': 1e15,    # V/m
            'Black hole ergosphere': 1e18,         # V/m
            'Schwinger critical field': 1.3e16,    # V/m
        }
        
        # Test different polymer scales
        mu_scales = [1e12, 1e15, 1e16, 1e17, 1e18, 1e19]  # GeV
        
        accessibility_results = []
        
        for name, E_field in field_scales.items():
            print(f"\n{name}: E = {E_field:.1e} V/m")
            print("-" * 40)
            
            for mu in mu_scales:
                gamma_std = self.schwinger_rate_standard(E_field)
                gamma_poly = self.schwinger_rate_polymer(E_field, mu, 'linear')
                
                enhancement = gamma_poly / gamma_std if gamma_std != 0 else 1.0
                
                print(f"  μ = {mu:.1e} GeV: log(Γ_std) = {gamma_std:.1f}, log(Γ_poly) = {gamma_poly:.1f}, enhancement = {enhancement:.2f}")
                
                accessibility_results.append({
                    'Environment': name,
                    'E_field_V_per_m': E_field,
                    'mu_scale_GeV': mu,
                    'log_gamma_standard': gamma_std,
                    'log_gamma_polymer': gamma_poly,
                    'enhancement_factor': enhancement,
                    'observable_standard': gamma_std > -50,
                    'observable_polymer': gamma_poly > -50
                })
        
        return pd.DataFrame(accessibility_results)

def plot_vacuum_instability_results(calculator, scan_results, output_dir='results'):
    """Create comprehensive plots of vacuum instability analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Critical fields vs polymer scale
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = scan_results['model'].unique()
    thresholds = scan_results['threshold'].unique()
    
    for i, model in enumerate(models[:4]):  # Plot first 4 models
        ax = axes.flatten()[i]
        
        model_data = scan_results[scan_results['model'] == model]
        
        for threshold in thresholds:
            threshold_data = model_data[model_data['threshold'] == threshold]
            
            # Only plot finite critical fields
            finite_mask = np.isfinite(threshold_data['E_critical_polymer_V_per_m'])
            
            if np.any(finite_mask):
                ax.loglog(
                    threshold_data[finite_mask]['mu_scale_GeV'],
                    threshold_data[finite_mask]['E_critical_polymer_V_per_m'],
                    'o-', label=f'threshold = {threshold}', alpha=0.7
                )
        
        # Add reference lines
        ax.axhline(1e13, color='red', linestyle='--', alpha=0.5, label='Lab laser')
        ax.axhline(1.3e16, color='black', linestyle='--', alpha=0.5, label='Schwinger')
        
        ax.set_xlabel('Polymer Scale μ (GeV)')
        ax.set_ylabel('Critical Field (V/m)')
        ax.set_title(f'{model.title()} Model')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vacuum_instability_critical_fields.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Critical field analysis plot saved: {output_dir}/vacuum_instability_critical_fields.png")

def main():
    """Run comprehensive vacuum instability analysis."""
    
    print("POLYMER-QED VACUUM INSTABILITY ANALYSIS")
    print("=" * 60)
    print("Testing whether LIV models predict observable vacuum breakdown")
    print("at laboratory-accessible field strengths.")
    print("=" * 60)
    
    # Initialize calculator
    calculator = VacuumInstabilityCalculator()
    
    # Run parameter space scan
    print("\n1. PARAMETER SPACE SCAN")
    scan_results = calculator.scan_parameter_space()
    
    # Laboratory accessibility analysis
    print("\n2. LABORATORY ACCESSIBILITY ANALYSIS")
    lab_results = calculator.laboratory_accessibility_analysis()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    scan_results.to_csv('results/vacuum_instability_scan.csv', index=False)
    lab_results.to_csv('results/vacuum_instability_laboratory.csv', index=False)
    
    # Generate plots
    plot_vacuum_instability_results(calculator, scan_results)
    
    # Summary analysis
    print(f"\n{'=' * 60}")
    print("VACUUM INSTABILITY SUMMARY")
    print(f"{'=' * 60}")
    
    # Find most promising parameter regions
    observable_cases = scan_results[
        np.isfinite(scan_results['E_critical_polymer_V_per_m']) &
        (scan_results['E_critical_polymer_V_per_m'] < 1e16)  # Below Schwinger critical
    ]
    
    if len(observable_cases) > 0:
        print(f"✅ Found {len(observable_cases)} potentially observable cases!")
        print("\nMost promising parameters:")
        best_cases = observable_cases.nsmallest(5, 'E_critical_polymer_V_per_m')
        print(best_cases[['model', 'mu_scale_GeV', 'E_critical_polymer_V_per_m', 'enhancement_factor']].to_string(index=False))
    else:
        print("❌ No cases found with critical fields below Schwinger limit")
        print("   Polymer-QED corrections insufficient for laboratory observation")
    
    # Laboratory feasibility
    lab_observable = lab_results[
        (lab_results['observable_polymer'] == True) &
        (lab_results['E_field_V_per_m'] <= 1e15)  # Laboratory achievable
    ]
    
    if len(lab_observable) > 0:
        print(f"\n✅ Laboratory observable cases: {len(lab_observable)}")
    else:
        print(f"\n❌ No laboratory-observable vacuum instabilities predicted")
    
    print(f"\nResults saved:")
    print(f"  - results/vacuum_instability_scan.csv")  
    print(f"  - results/vacuum_instability_laboratory.csv")
    print(f"  - results/vacuum_instability_critical_fields.png")
    
    return scan_results, lab_results

if __name__ == "__main__":
    scan_results, lab_results = main()
