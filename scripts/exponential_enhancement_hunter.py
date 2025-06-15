#!/usr/bin/env python3
"""
Enhanced Vacuum Instability Analysis: Exponential Enhancement Hunter

This module is specifically designed to find parameter regimes where 
LIV models produce dramatic exponential enhancements in vacuum 
instability rates at accessible field strengths.

Key improvements:
- More aggressive LIV enhancement models
- Better field-to-energy scale conversions
- Focused scanning on enhancement-prone regimes
- Systematic exponential enhancement detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Physical constants with better unit handling
class Constants:
    # Fundamental constants
    ELECTRON_MASS_KG = 9.109e-31      # kg
    ELECTRON_CHARGE = 1.602e-19       # C
    HBAR = 1.055e-34                  # J·s
    C_LIGHT = 2.998e8                 # m/s
    ELECTRON_MASS_GEV = 0.511e-3      # GeV
    
    # Field scales
    E_SCHWINGER = 1.3228e16           # V/m (Schwinger critical field)
    LAB_STRONG = 1e13                 # V/m
    LAB_EXTREME = 1e15                # V/m
    
    # Unit conversion: V/m → GeV
    # Using E [V/m] × e [C] × ℏc [GeV·m] / [J/GeV conversion]
    V_PER_M_TO_GEV = ELECTRON_CHARGE * HBAR * C_LIGHT / 1.602e-10  # ≈ 3.34e-16

CONST = Constants()

class ExponentialEnhancementHunter:
    """
    Aggressive search for exponential vacuum instability enhancements.
    
    This class implements more dramatic LIV enhancement models and 
    systematic scanning to identify regimes where pair production 
    could be exponentially enhanced at laboratory field strengths.
    """
    
    def __init__(self, model='aggressive_polynomial', coupling=1.0):
        self.model = model
        self.coupling = coupling
        
    def field_to_energy_scale(self, E_field_V_per_m):
        """
        Convert electric field to energy scale with better physics.
        
        Multiple conversion approaches to explore parameter space.
        """
        E_field = np.asarray(E_field_V_per_m)
        
        # Method 1: Dimensional analysis E [V/m] → E [GeV]
        E_GeV_method1 = E_field * CONST.V_PER_M_TO_GEV
        
        # Method 2: Normalized to Schwinger field
        E_GeV_method2 = CONST.ELECTRON_MASS_GEV * (E_field / CONST.E_SCHWINGER)
        
        # Method 3: Photon energy scale at Compton wavelength
        E_GeV_method3 = CONST.ELECTRON_MASS_GEV * np.sqrt(E_field / CONST.E_SCHWINGER)
        
        # Use method that gives most reasonable energy scales
        return E_GeV_method2  # This gives GeV-scale energies for strong fields
        
    def exponential_enhancement_factor(self, E_field_V_per_m, mu_GeV):
        """
        Aggressive LIV enhancement models designed to produce 
        exponential enhancements at accessible field strengths.
        """
        E_GeV = self.field_to_energy_scale(E_field_V_per_m)
        
        # Dimensionless field parameter
        x = E_GeV / mu_GeV
        
        if self.model == 'aggressive_polynomial':
            # High-order polynomial with large coefficients
            # F(x) = 1 + α₁x + α₂x² + α₃x³ + α₄x⁴
            a1 = self.coupling * 10
            a2 = self.coupling * 50
            a3 = self.coupling * 100
            a4 = self.coupling * 200
            enhancement = 1 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
            
        elif self.model == 'resonant_enhancement':
            # Resonant enhancement near specific μ values
            # F(x) = 1 + A / (1 + (x - x₀)²/Γ²)
            x0 = 1.0  # Resonance position
            Gamma = 0.1  # Resonance width
            A = self.coupling * 1000  # Resonance amplitude
            enhancement = 1 + A / (1 + ((x - x0)/Gamma)**2)
            
        elif self.model == 'exponential_runaway':
            # Exponential runaway for x > threshold
            x_threshold = 0.1
            if np.isscalar(x):
                if x > x_threshold:
                    enhancement = np.exp(self.coupling * (x - x_threshold) * 10)
                else:
                    enhancement = 1.0
            else:
                enhancement = np.where(x > x_threshold, 
                                     np.exp(self.coupling * (x - x_threshold) * 10), 
                                     1.0)
                                     
        elif self.model == 'power_law_burst':
            # Power law with high exponent
            # F(x) = (1 + αx)^β with large β
            alpha = self.coupling
            beta = 5.0 + self.coupling * 5  # High power
            enhancement = (1 + alpha * x)**beta
            
        elif self.model == 'threshold_explosion':
            # Sharp threshold with exponential enhancement
            x_crit = 0.5
            if np.isscalar(x):
                if x > x_crit:
                    enhancement = np.exp(self.coupling * (x/x_crit)**3 * 20)
                else:
                    enhancement = 1 + self.coupling * x * 0.1
            else:
                enhancement = np.where(x > x_crit,
                                     np.exp(self.coupling * (x/x_crit)**3 * 20),
                                     1 + self.coupling * x * 0.1)
                                     
        elif self.model == 'sine_wave_amplification':
            # Oscillatory enhancement with exponential envelope
            # F(x) = 1 + A × exp(αx) × sin²(βx)
            A = self.coupling * 100
            alpha = self.coupling * 2
            beta = 10
            enhancement = 1 + A * np.exp(alpha * x) * np.sin(beta * x)**2
            
        else:
            enhancement = np.ones_like(x)
        
        # Ensure numerical stability
        enhancement = np.maximum(enhancement, 1e-10)
        enhancement = np.minimum(enhancement, 1e50)  # Cap at huge but finite values
        
        return enhancement
    
    def schwinger_log_rate(self, E_field_V_per_m):
        """Schwinger rate in log form for numerical stability."""
        E_field = np.asarray(E_field_V_per_m)
        
        # Standard Schwinger exponent: -π m²/(eE) = -π (E_crit/E)
        field_ratio = E_field / CONST.E_SCHWINGER
        exponent = -np.pi / field_ratio
        
        # Add prefactor contribution (approximate)
        # log₁₀[(eE)²/(2π)³] ≈ 2×log₁₀(E) + constant
        log_prefactor = 2 * np.log10(E_field) - 10  # Rough estimate
        
        return (exponent / np.log(10)) + log_prefactor
    
    def enhanced_log_rate(self, E_field_V_per_m, mu_GeV):
        """LIV-enhanced rate in log form."""
        log_rate_std = self.schwinger_log_rate(E_field_V_per_m)
        enhancement = self.exponential_enhancement_factor(E_field_V_per_m, mu_GeV)
        log_enhancement = np.log10(enhancement)
        
        return log_rate_std + log_enhancement
    
    def scan_for_exponential_enhancements(self, target_fields=None, 
                                        mu_range=(1e10, 1e20), n_mu=100):
        """
        Systematic scan specifically looking for exponential enhancements.
        """
        if target_fields is None:
            target_fields = {
                'Lab Strong (10¹³ V/m)': CONST.LAB_STRONG,
                'Lab Extreme (10¹⁵ V/m)': CONST.LAB_EXTREME,
                'Pre-Schwinger (10¹⁶ V/m)': CONST.E_SCHWINGER * 0.1,
                'Schwinger (1.3×10¹⁶ V/m)': CONST.E_SCHWINGER
            }
        
        mu_values = np.logspace(np.log10(mu_range[0]), np.log10(mu_range[1]), n_mu)
        
        results = []
        exponential_cases = []
        
        print(f"🔍 EXPONENTIAL ENHANCEMENT SCAN ({self.model.upper()})")
        print(f"Coupling: {self.coupling}, μ range: {mu_range[0]:.1e} - {mu_range[1]:.1e} GeV")
        print("=" * 70)
        
        for field_name, E_field in target_fields.items():
            print(f"\n📍 {field_name}")
            print("-" * 50)
            
            max_enhancement = 0
            optimal_mu = None
            exponential_count = 0
            
            for i, mu in enumerate(mu_values):
                if i % 20 == 0:
                    print(f"  Progress: {i+1}/{n_mu}")
                
                enhancement = self.exponential_enhancement_factor(E_field, mu)
                log_rate_std = self.schwinger_log_rate(E_field)
                log_rate_enhanced = self.enhanced_log_rate(E_field, mu)
                
                # Classification
                exponential_enhancement = enhancement > 10
                mega_enhancement = enhancement > 1000
                ultra_enhancement = enhancement > 1e6
                observable_enhanced = log_rate_enhanced > -30
                
                if exponential_enhancement:
                    exponential_count += 1
                
                if enhancement > max_enhancement:
                    max_enhancement = enhancement
                    optimal_mu = mu
                
                # Store detailed results for exponential cases
                if exponential_enhancement:
                    exponential_cases.append({
                        'field_name': field_name,
                        'E_field_V_per_m': E_field,
                        'mu_GeV': mu,
                        'enhancement_factor': enhancement,
                        'log_enhancement': np.log10(enhancement),
                        'log_rate_standard': log_rate_std,
                        'log_rate_enhanced': log_rate_enhanced,
                        'rate_improvement': log_rate_enhanced - log_rate_std,
                        'mega_enhancement': mega_enhancement,
                        'ultra_enhancement': ultra_enhancement,
                        'observable_enhanced': observable_enhanced,
                        'model': self.model,
                        'coupling': self.coupling
                    })
                
                results.append({
                    'field_name': field_name,
                    'E_field_V_per_m': E_field,
                    'mu_GeV': mu,
                    'enhancement_factor': enhancement,
                    'log_enhancement': np.log10(enhancement),
                    'exponential_enhancement': exponential_enhancement,
                    'observable_enhanced': observable_enhanced,
                    'model': self.model,
                    'coupling': self.coupling
                })
            
            print(f"  📊 Results for {field_name}:")
            print(f"    Max enhancement: {max_enhancement:.2e} at μ = {optimal_mu:.1e} GeV")
            print(f"    Exponential cases (>10×): {exponential_count}/{n_mu}")
            
            if max_enhancement > 1000:
                print(f"    🚀 MEGA ENHANCEMENT DETECTED!")
            elif max_enhancement > 10:
                print(f"    ✅ Significant enhancement found")
            else:
                print(f"    ❌ Limited enhancement")
        
        return pd.DataFrame(results), pd.DataFrame(exponential_cases)


class ExponentialEnhancementVisualizer:
    """Specialized visualization for exponential enhancement analysis."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_enhancement_landscapes(self, results_df, exponential_df):
        """Create comprehensive enhancement landscape plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Get unique field scenarios
        field_scenarios = results_df['field_name'].unique()
        
        # Plot 1: Enhancement vs μ for all fields
        ax = axes[0, 0]
        for field_name in field_scenarios[:4]:  # Limit to 4 fields
            field_data = results_df[results_df['field_name'] == field_name]
            ax.loglog(field_data['mu_GeV'], field_data['enhancement_factor'], 
                     'o-', label=field_name, alpha=0.7, markersize=3)
        
        ax.axhline(10, color='green', linestyle='--', alpha=0.5, label='10× Enhancement')
        ax.axhline(1000, color='orange', linestyle='--', alpha=0.5, label='1000× Enhancement')
        ax.axhline(1e6, color='red', linestyle='--', alpha=0.5, label='10⁶× Enhancement')
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('Enhancement Factor')
        ax.set_title('Enhancement Factor vs LIV Scale')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Exponential enhancement parameter space
        ax = axes[0, 1]
        if len(exponential_df) > 0:
            scatter = ax.scatter(exponential_df['mu_GeV'], exponential_df['E_field_V_per_m'],
                               c=np.log10(exponential_df['enhancement_factor']),
                               s=50, alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='log₁₀(Enhancement)')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('Electric Field (V/m)')
        ax.set_title('Exponential Enhancement Regions')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rate improvement
        ax = axes[0, 2]
        lab_extreme_data = results_df[results_df['field_name'].str.contains('Lab Extreme')]
        if len(lab_extreme_data) > 0:
            ax.semilogx(lab_extreme_data['mu_GeV'], lab_extreme_data['log_enhancement'],
                       'o-', color='red', alpha=0.7)
            ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='10× Enhancement')
            ax.axhline(3, color='orange', linestyle='--', alpha=0.5, label='1000× Enhancement')
        
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('log₁₀(Enhancement)')
        ax.set_title('Enhancement vs μ (Lab Extreme Field)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Observable enhancement regions
        ax = axes[1, 0]
        observable_data = results_df[results_df['observable_enhanced'] == True]
        if len(observable_data) > 0:
            for field_name in field_scenarios[:3]:
                field_obs = observable_data[observable_data['field_name'] == field_name]
                if len(field_obs) > 0:
                    ax.scatter(field_obs['mu_GeV'], field_obs['enhancement_factor'],
                             label=f'{field_name} (Observable)', alpha=0.7, s=30)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('μ (GeV)')
        ax.set_ylabel('Enhancement Factor')
        ax.set_title('Observable Enhancement Regions')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Enhancement distribution
        ax = axes[1, 1]
        if len(exponential_df) > 0:
            ax.hist(np.log10(exponential_df['enhancement_factor']), 
                   bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(1, color='green', linestyle='--', label='10× Enhancement')
            ax.axvline(3, color='orange', linestyle='--', label='1000× Enhancement')
        
        ax.set_xlabel('log₁₀(Enhancement Factor)')
        ax.set_ylabel('Number of Cases')
        ax.set_title('Distribution of Exponential Enhancements')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Field strength dependence
        ax = axes[1, 2]
        if len(exponential_df) > 0:
            # Group by field strength
            field_groups = exponential_df.groupby('E_field_V_per_m')['enhancement_factor'].max()
            ax.loglog(field_groups.index, field_groups.values, 'ro-', alpha=0.7)
        
        ax.set_xlabel('Electric Field (V/m)')
        ax.set_ylabel('Max Enhancement Factor')
        ax.set_title('Peak Enhancement vs Field Strength')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/exponential_enhancement_landscape.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhancement landscape plot saved: {self.output_dir}/exponential_enhancement_landscape.png")


def main():
    """
    Main execution: Hunt for exponential vacuum instability enhancements.
    """
    print("🚀 EXPONENTIAL VACUUM INSTABILITY ENHANCEMENT HUNTER")
    print("=" * 70)
    print("Systematic search for dramatic LIV enhancements at accessible fields")
    print("=" * 70)
    
    # Define aggressive models to test
    models_to_test = [
        ('aggressive_polynomial', [0.1, 1.0, 10.0]),
        ('resonant_enhancement', [0.1, 1.0, 10.0]),
        ('exponential_runaway', [0.1, 1.0, 5.0]),
        ('power_law_burst', [0.1, 1.0, 5.0]),
        ('threshold_explosion', [0.1, 1.0, 5.0]),
        ('sine_wave_amplification', [0.1, 1.0, 3.0])
    ]
    
    all_results = []
    all_exponential_cases = []
    
    for model_name, coupling_values in models_to_test:
        print(f"\n{'='*50}")
        print(f"🔬 TESTING MODEL: {model_name.upper()}")
        print(f"{'='*50}")
        
        for coupling in coupling_values:
            print(f"\n🎛️  Coupling strength: {coupling}")
            
            # Initialize hunter
            hunter = ExponentialEnhancementHunter(model=model_name, coupling=coupling)
            
            # Run enhancement scan
            results, exponential_cases = hunter.scan_for_exponential_enhancements(
                mu_range=(1e10, 1e20), n_mu=50
            )
            
            # Add model info
            results['model'] = model_name
            results['coupling'] = coupling
            if len(exponential_cases) > 0:
                exponential_cases['model'] = model_name
                exponential_cases['coupling'] = coupling
            
            all_results.append(results)
            if len(exponential_cases) > 0:
                all_exponential_cases.append(exponential_cases)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    if all_exponential_cases:
        combined_exponential = pd.concat(all_exponential_cases, ignore_index=True)
    else:
        combined_exponential = pd.DataFrame()
    
    # Save results
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    combined_results.to_csv(f'{output_dir}/exponential_enhancement_scan.csv', index=False)
    if len(combined_exponential) > 0:
        combined_exponential.to_csv(f'{output_dir}/exponential_cases_detailed.csv', index=False)
    
    # Create visualizations
    visualizer = ExponentialEnhancementVisualizer(output_dir)
    visualizer.plot_enhancement_landscapes(combined_results, combined_exponential)
    
    # Analysis summary
    print(f"\n{'='*70}")
    print("🎯 EXPONENTIAL ENHANCEMENT HUNTER SUMMARY")
    print(f"{'='*70}")
    
    # Overall statistics
    total_cases = len(combined_results)
    exponential_cases = len(combined_exponential)
    exponential_fraction = exponential_cases / total_cases * 100 if total_cases > 0 else 0
    
    print(f"\n📊 SCAN STATISTICS:")
    print(f"  Total parameter combinations: {total_cases:,}")
    print(f"  Exponential enhancements (>10×): {exponential_cases:,} ({exponential_fraction:.1f}%)")
    
    if len(combined_exponential) > 0:
        # Best cases analysis
        mega_cases = combined_exponential[combined_exponential['mega_enhancement'] == True]
        ultra_cases = combined_exponential[combined_exponential['ultra_enhancement'] == True]
        observable_cases = combined_exponential[combined_exponential['observable_enhanced'] == True]
        
        print(f"  Mega enhancements (>1000×): {len(mega_cases):,}")
        print(f"  Ultra enhancements (>10⁶×): {len(ultra_cases):,}")
        print(f"  Observable enhanced rates: {len(observable_cases):,}")
        
        # Top enhancement cases
        top_cases = combined_exponential.nlargest(10, 'enhancement_factor')
        print(f"\n🚀 TOP 10 EXPONENTIAL ENHANCEMENTS:")
        print("-" * 50)
        for i, (_, case) in enumerate(top_cases.iterrows(), 1):
            print(f"{i:2d}. {case['model']:20s} | μ={case['mu_GeV']:8.1e} GeV | "
                  f"Enhancement={case['enhancement_factor']:8.1e} | {case['field_name']}")
        
        # Laboratory accessibility
        lab_accessible = combined_exponential[
            combined_exponential['field_name'].str.contains('Lab') &
            (combined_exponential['enhancement_factor'] > 100)
        ]
        
        if len(lab_accessible) > 0:
            print(f"\n🏆 LABORATORY-ACCESSIBLE MEGA ENHANCEMENTS:")
            print("-" * 50)
            for _, case in lab_accessible.nlargest(5, 'enhancement_factor').iterrows():
                print(f"  {case['model']:20s} | μ={case['mu_GeV']:8.1e} GeV | "
                      f"Enhancement={case['enhancement_factor']:8.1e}")
                print(f"    Field: {case['field_name']} | Rate improvement: {case['rate_improvement']:+.1f} decades")
        else:
            print(f"\n❌ No laboratory-accessible mega enhancements found")
    
    else:
        print(f"\n❌ No exponential enhancements found with current models")
        print("    Try more aggressive coupling strengths or different models")
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"  - {output_dir}/exponential_enhancement_scan.csv")
    if len(combined_exponential) > 0:
        print(f"  - {output_dir}/exponential_cases_detailed.csv")
    print(f"  - {output_dir}/exponential_enhancement_landscape.png")
    
    return combined_results, combined_exponential


if __name__ == "__main__":
    results, exponential_cases = main()
