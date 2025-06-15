#!/usr/bin/env python3
"""
Vacuum Instability Module: (μ, E) → Γ(E)

This module provides a clean interface for computing vacuum instability rates
as a function of Lorentz-violating scale μ and electric field E, with 
comprehensive scanning capabilities to identify exponential enhancements
at accessible field strengths.

Key Features:
- Physically accurate Schwinger pair production rates
- Multiple LIV enhancement models
- Parameter space scanning with optimization
- Laboratory accessibility analysis
- Exponential enhancement detection

Physics:
Standard: Γ_std = (eE)²/(2π)³ × exp[-π m²/(eE)]
LIV-enhanced: Γ_LIV = Γ_std × F(μ,E)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar, curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Physical constants
class PhysicalConstants:
    """Container for physical constants in various unit systems."""
    
    # SI units
    ELECTRON_MASS_KG = 9.109e-31      # kg
    ELECTRON_CHARGE = 1.602e-19       # C
    HBAR = 1.055e-34                  # J·s
    C_LIGHT = 2.998e8                 # m/s
    
    # Natural units (GeV)
    ELECTRON_MASS_GEV = 0.511e-3      # GeV
    ALPHA_EM = 1/137.036              # Fine structure constant
    E_PLANCK_GEV = 1.22e19            # GeV
    
    # Derived quantities
    E_SCHWINGER = ELECTRON_MASS_KG**2 * C_LIGHT**3 / (ELECTRON_CHARGE * HBAR)  # V/m ≈ 1.3×10¹⁶
    
    # Laboratory field scales
    LAB_LASER_STRONG = 1e13           # V/m (current strong laser)
    LAB_LASER_EXTREME = 1e15          # V/m (next-generation)
    NEUTRON_STAR_SURFACE = 1e12       # V/m
    NEUTRON_STAR_MAGNETOSPHERE = 1e15 # V/m

# Initialize constants
CONST = PhysicalConstants()

class VacuumInstabilityCore:
    """
    Core vacuum instability calculations: (μ, E) → Γ(E)
    
    This class implements the fundamental mapping from LIV scale μ and 
    electric field E to pair production rate Γ(E).
    """
    
    def __init__(self, model='polynomial', coupling_strength=1.0):
        """
        Initialize vacuum instability calculator.
        
        Parameters:
        -----------
        model : str
            LIV enhancement model ('polynomial', 'exponential', 'threshold', 'rainbow')
        coupling_strength : float
            Overall strength of LIV coupling
        """
        self.model = model
        self.coupling_strength = coupling_strength
        self.cache = {}  # For computational efficiency
        
    def schwinger_rate_standard(self, E_field_V_per_m):
        """
        Standard Schwinger pair production rate.
        
        Γ_std = (eE)²/(2π)³ × exp[-π m²/(eE)]
        
        Parameters:
        -----------
        E_field_V_per_m : float or array
            Electric field strength in V/m
            
        Returns:
        --------
        rate : float or array
            Pair production rate in units of m⁻³s⁻¹
        """
        E_field = np.asarray(E_field_V_per_m)
        
        # Prefactor: (eE)²/(2π)³
        eE = CONST.ELECTRON_CHARGE * E_field  # J/m
        eE_natural = eE / (CONST.HBAR * CONST.C_LIGHT)  # 1/m
        prefactor = (eE_natural)**2 / (2 * np.pi)**3
        
        # Exponential factor: exp[-π m²/(eE)]
        field_ratio = E_field / CONST.E_SCHWINGER
        exponent = -np.pi / field_ratio
        exponential = np.exp(exponent)
        
        rate = prefactor * exponential
        return rate
        
    def schwinger_log_rate_standard(self, E_field_V_per_m):
        """
        Log of standard Schwinger rate for numerical stability.
        
        Returns log₁₀(Γ_std) to handle extremely small rates.
        """
        E_field = np.asarray(E_field_V_per_m)
        
        # Log prefactor
        eE = CONST.ELECTRON_CHARGE * E_field
        eE_natural = eE / (CONST.HBAR * CONST.C_LIGHT)
        log_prefactor = 2 * np.log10(eE_natural) - 3 * np.log10(2 * np.pi)
        
        # Log exponential (convert from natural log)
        field_ratio = E_field / CONST.E_SCHWINGER
        log_exponent = -np.pi / (field_ratio * np.log(10))
        
        return log_prefactor + log_exponent
    
    def liv_enhancement_factor(self, E_field_V_per_m, mu_GeV):
        """
        LIV enhancement factor F(μ,E).
        
        This is where the physics of Lorentz violation enters.
        Different models give different functional forms.
        
        Parameters:
        -----------
        E_field_V_per_m : float or array
            Electric field strength in V/m
        mu_GeV : float
            LIV energy scale in GeV
            
        Returns:
        --------
        enhancement : float or array
            Enhancement factor F(μ,E) ≥ 1
        """
        E_field = np.asarray(E_field_V_per_m)
        
        # Convert field to energy scale (rough conversion)
        # E [V/m] → E [GeV] via dimensional analysis
        E_GeV = E_field * CONST.ELECTRON_CHARGE * 1e-9 / 1.602e-10  # Rough conversion
        
        # Dimensionless parameter
        x = E_GeV / mu_GeV
        
        if self.model == 'polynomial':
            # Polynomial enhancement: F(x) = 1 + αx + βx²
            alpha = self.coupling_strength
            beta = self.coupling_strength * 0.1
            enhancement = 1 + alpha * x + beta * x**2
            
        elif self.model == 'exponential':
            # Exponential enhancement: F(x) = exp(αx)
            alpha = self.coupling_strength * 0.1  # Keep moderate
            enhancement = np.exp(alpha * x)
            
        elif self.model == 'threshold':
            # Threshold behavior: F(x) = 1 + α×max(0, x-1)
            alpha = self.coupling_strength * 10
            enhancement = 1 + alpha * np.maximum(0, x - 1)
            
        elif self.model == 'rainbow':
            # Rainbow gravity: F(x) = (1 + ηx)^β
            eta = self.coupling_strength
            beta = 2.0
            enhancement = (1 + eta * x)**beta
            
        elif self.model == 'suppression':
            # Suppression model: F(x) = exp(-x)
            enhancement = np.exp(-x)
            
        else:
            # No enhancement
            enhancement = np.ones_like(x)
        
        return np.maximum(enhancement, 1e-10)  # Avoid numerical issues
    
    def vacuum_instability_rate(self, E_field_V_per_m, mu_GeV):
        """
        Total LIV-enhanced vacuum instability rate.
        
        Γ_LIV(E,μ) = Γ_std(E) × F(μ,E)
        
        Parameters:
        -----------
        E_field_V_per_m : float or array
            Electric field strength in V/m
        mu_GeV : float
            LIV energy scale in GeV
            
        Returns:
        --------
        rate : float or array
            Enhanced pair production rate
        """
        # Standard rate
        gamma_std = self.schwinger_rate_standard(E_field_V_per_m)
        
        # LIV enhancement
        enhancement = self.liv_enhancement_factor(E_field_V_per_m, mu_GeV)
        
        # Total rate
        gamma_liv = gamma_std * enhancement
        
        return gamma_liv
    
    def vacuum_instability_log_rate(self, E_field_V_per_m, mu_GeV):
        """
        Log of LIV-enhanced rate for numerical stability.
        
        Returns log₁₀(Γ_LIV) = log₁₀(Γ_std) + log₁₀(F)
        """
        # Log standard rate
        log_gamma_std = self.schwinger_log_rate_standard(E_field_V_per_m)
        
        # Log enhancement factor
        enhancement = self.liv_enhancement_factor(E_field_V_per_m, mu_GeV)
        log_enhancement = np.log10(enhancement)
        
        return log_gamma_std + log_enhancement


class VacuumInstabilityScanDriver:
    """
    Scan driver for systematic exploration of (μ,E) parameter space.
    
    Features:
    - Multi-dimensional parameter scanning
    - Exponential enhancement detection
    - Laboratory accessibility analysis
    - Optimization for maximum enhancement
    """
    
    def __init__(self, calculator=None):
        """
        Initialize scan driver.
        
        Parameters:
        -----------
        calculator : VacuumInstabilityCore, optional
            Core calculator instance
        """
        if calculator is None:
            calculator = VacuumInstabilityCore()
        self.calculator = calculator
        self.results = []
        
    def field_range_scan(self, mu_GeV, E_min=1e10, E_max=1e18, n_points=100):
        """
        Scan enhancement factor across field range for fixed μ.
        
        Parameters:
        -----------
        mu_GeV : float
            LIV scale in GeV
        E_min, E_max : float
            Field range in V/m
        n_points : int
            Number of field points
            
        Returns:
        --------
        results : dict
            Scan results with field-dependent enhancement
        """
        E_fields = np.logspace(np.log10(E_min), np.log10(E_max), n_points)
        
        enhancements = []
        log_rates_std = []
        log_rates_liv = []
        
        for E in E_fields:
            enhancement = self.calculator.liv_enhancement_factor(E, mu_GeV)
            log_rate_std = self.calculator.schwinger_log_rate_standard(E)
            log_rate_liv = self.calculator.vacuum_instability_log_rate(E, mu_GeV)
            
            enhancements.append(enhancement)
            log_rates_std.append(log_rate_std)
            log_rates_liv.append(log_rate_liv)
        
        return {
            'mu_GeV': mu_GeV,
            'E_fields_V_per_m': E_fields,
            'enhancements': np.array(enhancements),
            'log_rates_standard': np.array(log_rates_std),
            'log_rates_liv': np.array(log_rates_liv),
            'max_enhancement': np.max(enhancements),
            'optimal_field': E_fields[np.argmax(enhancements)]
        }
    
    def mu_parameter_scan(self, mu_min=1e10, mu_max=1e19, n_mu=50, 
                         target_fields=None):
        """
        Scan LIV parameter μ for exponential enhancements.
        
        Parameters:
        -----------
        mu_min, mu_max : float
            LIV scale range in GeV
        n_mu : int
            Number of μ points
        target_fields : array, optional
            Specific field strengths to test
            
        Returns:
        --------
        scan_results : DataFrame
            Complete μ-scan results
        """
        if target_fields is None:
            target_fields = [CONST.LAB_LASER_STRONG, CONST.LAB_LASER_EXTREME, 
                           CONST.NEUTRON_STAR_SURFACE, CONST.NEUTRON_STAR_MAGNETOSPHERE,
                           CONST.E_SCHWINGER]
        
        mu_scales = np.logspace(np.log10(mu_min), np.log10(mu_max), n_mu)
        
        results = []
        
        print(f"Scanning {n_mu} LIV scales from {mu_min:.1e} to {mu_max:.1e} GeV")
        print("=" * 60)
        
        for i, mu in enumerate(mu_scales):
            if i % 10 == 0:
                print(f"Progress: {i+1}/{n_mu} (μ = {mu:.2e} GeV)")
            
            for j, E_field in enumerate(target_fields):
                field_name = ['Lab Strong', 'Lab Extreme', 'NS Surface', 
                             'NS Magnetosphere', 'Schwinger'][j]
                
                # Calculate rates and enhancement
                log_rate_std = self.calculator.schwinger_log_rate_standard(E_field)
                log_rate_liv = self.calculator.vacuum_instability_log_rate(E_field, mu)
                enhancement = self.calculator.liv_enhancement_factor(E_field, mu)
                
                # Detectability criterion
                observable_std = log_rate_std > -50
                observable_liv = log_rate_liv > -50
                exponential_enhancement = enhancement > 10
                
                results.append({
                    'mu_GeV': mu,
                    'field_name': field_name,
                    'E_field_V_per_m': E_field,
                    'log_rate_standard': log_rate_std,
                    'log_rate_liv': log_rate_liv,
                    'enhancement_factor': enhancement,
                    'log_enhancement': np.log10(enhancement),
                    'observable_standard': observable_std,
                    'observable_liv': observable_liv,
                    'exponential_enhancement': exponential_enhancement,
                    'net_enhancement_db': 10 * np.log10(enhancement)  # In decibels
                })
        
        return pd.DataFrame(results)
    
    def find_optimal_parameters(self, target_field=None, enhancement_threshold=10):
        """
        Find LIV parameters that maximize enhancement at target field.
        
        Parameters:
        -----------
        target_field : float, optional
            Target field strength in V/m
        enhancement_threshold : float
            Minimum enhancement factor to consider "significant"
            
        Returns:
        --------
        optimal_params : dict
            Optimal μ and corresponding enhancement
        """
        if target_field is None:
            target_field = CONST.LAB_LASER_EXTREME
        
        def negative_enhancement(log_mu):
            """Objective function: -enhancement (for minimization)"""
            mu = 10**log_mu
            enhancement = self.calculator.liv_enhancement_factor(target_field, mu)
            return -enhancement
        
        # Optimize over reasonable μ range
        result = minimize_scalar(negative_enhancement, 
                               bounds=(10, 19),  # 10¹⁰ to 10¹⁹ GeV
                               method='bounded')
        
        optimal_mu = 10**result.x
        max_enhancement = -result.fun
        
        return {
            'target_field_V_per_m': target_field,
            'optimal_mu_GeV': optimal_mu,
            'max_enhancement': max_enhancement,
            'log_enhancement': np.log10(max_enhancement),
            'significant': max_enhancement > enhancement_threshold,
            'optimization_success': result.success
        }
    
    def laboratory_accessibility_assessment(self):
        """
        Comprehensive assessment of laboratory accessibility.
        
        Tests whether LIV-enhanced vacuum instabilities could be
        observed with current or near-future laboratory technology.
        """
        print("\nLABORATORY ACCESSIBILITY ASSESSMENT")
        print("=" * 50)
        
        # Define laboratory scenarios
        lab_scenarios = {
            'Current Strong Laser (10¹³ V/m)': CONST.LAB_LASER_STRONG,
            'Next-Gen Extreme Laser (10¹⁵ V/m)': CONST.LAB_LASER_EXTREME,
            'Future Ultrastrong (10¹⁶ V/m)': CONST.E_SCHWINGER * 0.1
        }
        
        # Test range of μ values
        mu_test_values = np.logspace(12, 18, 20)  # 10¹² to 10¹⁸ GeV
        
        accessibility_results = []
        
        for scenario_name, E_field in lab_scenarios.items():
            print(f"\n{scenario_name}")
            print("-" * 40)
            
            best_enhancement = 0
            best_mu = None
            observable_count = 0
            
            for mu in mu_test_values:
                enhancement = self.calculator.liv_enhancement_factor(E_field, mu)
                log_rate_liv = self.calculator.vacuum_instability_log_rate(E_field, mu)
                
                observable = log_rate_liv > -30  # More lenient threshold
                if observable:
                    observable_count += 1
                
                if enhancement > best_enhancement:
                    best_enhancement = enhancement
                    best_mu = mu
                
                accessibility_results.append({
                    'scenario': scenario_name,
                    'E_field_V_per_m': E_field,
                    'mu_GeV': mu,
                    'enhancement_factor': enhancement,
                    'log_rate_liv': log_rate_liv,
                    'observable': observable
                })
            
            print(f"  Best enhancement: {best_enhancement:.2e} at μ = {best_mu:.1e} GeV")
            print(f"  Observable cases: {observable_count}/{len(mu_test_values)}")
            
            if best_enhancement > 100:
                print(f"  🚀 SIGNIFICANT ENHANCEMENT FOUND!")
            elif best_enhancement > 10:
                print(f"  ✅ Moderate enhancement possible")
            else:
                print(f"  ❌ Limited enhancement at this field strength")
        
        return pd.DataFrame(accessibility_results)


class VacuumInstabilityVisualizer:
    """
    Visualization tools for vacuum instability analysis.
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_enhancement_vs_field(self, scan_results, mu_values=None):
        """Plot enhancement factor vs field strength for different μ."""
        
        if mu_values is None:
            # Select representative μ values
            all_mu = scan_results['mu_GeV'].unique()
            mu_values = np.logspace(np.log10(all_mu.min()), np.log10(all_mu.max()), 5)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Enhancement factor vs field
        for mu in mu_values:
            mu_data = scan_results[np.isclose(scan_results['mu_GeV'], mu, rtol=0.1)]
            if len(mu_data) > 0:
                ax1.loglog(mu_data['E_field_V_per_m'], mu_data['enhancement_factor'],
                          'o-', label=f'μ = {mu:.1e} GeV', alpha=0.7)
        
        # Add reference lines
        ax1.axvline(CONST.LAB_LASER_STRONG, color='red', linestyle='--', 
                   alpha=0.5, label='Lab Strong')
        ax1.axvline(CONST.LAB_LASER_EXTREME, color='orange', linestyle='--', 
                   alpha=0.5, label='Lab Extreme')
        ax1.axvline(CONST.E_SCHWINGER, color='black', linestyle='--', 
                   alpha=0.5, label='Schwinger')
        ax1.axhline(10, color='green', linestyle=':', alpha=0.5, label='10× Enhancement')
        
        ax1.set_xlabel('Electric Field (V/m)')
        ax1.set_ylabel('Enhancement Factor')
        ax1.set_title('LIV Enhancement vs Field Strength')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log rate enhancement
        for mu in mu_values:
            mu_data = scan_results[np.isclose(scan_results['mu_GeV'], mu, rtol=0.1)]
            if len(mu_data) > 0:
                rate_enhancement = mu_data['log_rate_liv'] - mu_data['log_rate_standard']
                ax2.semilogx(mu_data['E_field_V_per_m'], rate_enhancement,
                           'o-', label=f'μ = {mu:.1e} GeV', alpha=0.7)
        
        ax2.axvline(CONST.LAB_LASER_STRONG, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(CONST.LAB_LASER_EXTREME, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_xlabel('Electric Field (V/m)')
        ax2.set_ylabel('Log Rate Enhancement')
        ax2.set_title('Rate Enhancement vs Field Strength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/vacuum_enhancement_vs_field.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhancement plot saved: {self.output_dir}/vacuum_enhancement_vs_field.png")
    
    def plot_parameter_space_scan(self, scan_results):
        """Create comprehensive parameter space visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data for different field scenarios
        field_scenarios = scan_results['field_name'].unique()
        
        # Plot 1: Enhancement factor heatmap
        ax = axes[0, 0]
        
        # Focus on lab-accessible fields
        lab_data = scan_results[scan_results['field_name'].isin(['Lab Strong', 'Lab Extreme'])]
        
        if len(lab_data) > 0:
            pivot_data = lab_data.pivot(index='mu_GeV', columns='field_name', 
                                      values='enhancement_factor')
            
            im = ax.imshow(np.log10(pivot_data.values), aspect='auto', 
                          extent=[0, len(pivot_data.columns), 
                                 np.log10(pivot_data.index.min()), 
                                 np.log10(pivot_data.index.max())],
                          origin='lower', cmap='viridis')
            
            ax.set_ylabel('log₁₀(μ [GeV])')
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns, rotation=45)
            ax.set_title('log₁₀(Enhancement Factor)')
            plt.colorbar(im, ax=ax)
        
        # Plot 2: Observable parameter regions
        ax = axes[0, 1]
        
        for field_name in field_scenarios[:4]:  # Limit to 4 scenarios
            field_data = scan_results[scan_results['field_name'] == field_name]
            observable_mask = field_data['observable_liv']
            
            if np.any(observable_mask):
                ax.scatter(field_data[observable_mask]['mu_GeV'], 
                          field_data[observable_mask]['enhancement_factor'],
                          label=f'{field_name} (Observable)', alpha=0.7, s=30)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('Enhancement Factor')
        ax.set_title('Observable Parameter Regions')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Enhancement vs μ for fixed field
        ax = axes[1, 0]
        
        target_field = 'Lab Extreme'
        target_data = scan_results[scan_results['field_name'] == target_field]
        
        if len(target_data) > 0:
            ax.loglog(target_data['mu_GeV'], target_data['enhancement_factor'], 
                     'o-', color='red', alpha=0.7)
            ax.axhline(10, color='green', linestyle='--', alpha=0.5, 
                      label='10× Enhancement')
            ax.axhline(100, color='orange', linestyle='--', alpha=0.5, 
                      label='100× Enhancement')
        
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('Enhancement Factor')
        ax.set_title(f'Enhancement vs μ ({target_field})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Rate comparison
        ax = axes[1, 1]
        
        if len(target_data) > 0:
            ax.semilogx(target_data['mu_GeV'], target_data['log_rate_standard'], 
                       'b--', label='Standard Schwinger', alpha=0.7)
            ax.semilogx(target_data['mu_GeV'], target_data['log_rate_liv'], 
                       'r-', label='LIV Enhanced', alpha=0.7)
            ax.axhline(-30, color='green', linestyle=':', alpha=0.5, 
                      label='Observable Threshold')
        
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('log₁₀(Rate)')
        ax.set_title('Rate Comparison vs μ')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/vacuum_parameter_space_scan.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter space plot saved: {self.output_dir}/vacuum_parameter_space_scan.png")


def main():
    """
    Main execution function for comprehensive vacuum instability analysis.
    """
    print("VACUUM INSTABILITY ANALYSIS: (μ, E) → Γ(E)")
    print("=" * 60)
    print("Systematic exploration of LIV-enhanced vacuum breakdown")
    print("Looking for exponential enhancements at accessible fields")
    print("=" * 60)
    
    # Initialize system
    models_to_test = ['polynomial', 'exponential', 'threshold', 'rainbow']
    coupling_strengths = [0.1, 1.0, 10.0]
    
    all_results = []
    all_optimizations = []
    
    for model in models_to_test:
        print(f"\n🔍 TESTING MODEL: {model.upper()}")
        print("-" * 40)
        
        for coupling in coupling_strengths:
            print(f"  Coupling strength: {coupling}")
            
            # Initialize calculator and scanner
            calculator = VacuumInstabilityCore(model=model, 
                                             coupling_strength=coupling)
            scanner = VacuumInstabilityScanDriver(calculator)
            
            # Run parameter space scan
            scan_results = scanner.mu_parameter_scan(mu_min=1e12, mu_max=1e18, n_mu=30)
            scan_results['model'] = model
            scan_results['coupling'] = coupling
            all_results.append(scan_results)
            
            # Find optimal parameters for laboratory fields
            for field_name, field_value in [('Lab Strong', CONST.LAB_LASER_STRONG), 
                                          ('Lab Extreme', CONST.LAB_LASER_EXTREME)]:
                optimal = scanner.find_optimal_parameters(target_field=field_value)
                optimal.update({'model': model, 'coupling': coupling, 
                               'field_name': field_name})
                all_optimizations.append(optimal)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    optimization_results = pd.DataFrame(all_optimizations)
    
    # Laboratory accessibility assessment (using best model)
    best_calculator = VacuumInstabilityCore(model='polynomial', coupling_strength=1.0)
    best_scanner = VacuumInstabilityScanDriver(best_calculator)
    accessibility_results = best_scanner.laboratory_accessibility_assessment()
    
    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    combined_results.to_csv(f'{output_dir}/vacuum_instability_complete_scan.csv', 
                           index=False)
    optimization_results.to_csv(f'{output_dir}/vacuum_instability_optimizations.csv', 
                               index=False)
    accessibility_results.to_csv(f'{output_dir}/vacuum_instability_accessibility.csv', 
                                index=False)
    
    # Generate visualizations
    visualizer = VacuumInstabilityVisualizer(output_dir)
    
    # Select best results for plotting
    best_results = combined_results[
        (combined_results['model'] == 'polynomial') & 
        (combined_results['coupling'] == 1.0)
    ]
    
    visualizer.plot_enhancement_vs_field(best_results)
    visualizer.plot_parameter_space_scan(best_results)
    
    # Analysis summary
    print(f"\n{'=' * 60}")
    print("VACUUM INSTABILITY ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    
    # Find most promising cases
    significant_enhancements = combined_results[
        combined_results['exponential_enhancement'] == True
    ]
    
    lab_accessible = combined_results[
        (combined_results['field_name'].isin(['Lab Strong', 'Lab Extreme'])) &
        (combined_results['observable_liv'] == True)
    ]
    
    print(f"\n📊 RESULTS OVERVIEW:")
    print(f"  Total parameter combinations tested: {len(combined_results):,}")
    print(f"  Cases with exponential enhancement (>10×): {len(significant_enhancements):,}")
    print(f"  Laboratory-accessible cases: {len(lab_accessible):,}")
    
    if len(significant_enhancements) > 0:
        print(f"\n🚀 TOP EXPONENTIAL ENHANCEMENTS:")
        top_cases = significant_enhancements.nlargest(5, 'enhancement_factor')
        for _, case in top_cases.iterrows():
            print(f"  {case['model']}: μ={case['mu_GeV']:.1e} GeV, "
                  f"Enhancement={case['enhancement_factor']:.1e}, "
                  f"Field={case['field_name']}")
    
    # Optimization summary
    best_optimizations = optimization_results[
        optimization_results['significant'] == True
    ]
    
    if len(best_optimizations) > 0:
        print(f"\n🎯 OPTIMAL PARAMETERS FOR LABORATORY ACCESS:")
        for _, opt in best_optimizations.iterrows():
            print(f"  {opt['field_name']} ({opt['model']}): "
                  f"μ_opt = {opt['optimal_mu_GeV']:.1e} GeV, "
                  f"Max Enhancement = {opt['max_enhancement']:.1e}")
    
    print(f"\n💾 Results saved to:")
    print(f"  - {output_dir}/vacuum_instability_complete_scan.csv")
    print(f"  - {output_dir}/vacuum_instability_optimizations.csv")
    print(f"  - {output_dir}/vacuum_instability_accessibility.csv")
    print(f"  - {output_dir}/vacuum_enhancement_vs_field.png")
    print(f"  - {output_dir}/vacuum_parameter_space_scan.png")
    
    return combined_results, optimization_results, accessibility_results


if __name__ == "__main__":
    scan_results, opt_results, access_results = main()
