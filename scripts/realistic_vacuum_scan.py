#!/usr/bin/env python3
"""
Realistic Vacuum Instability Analysis with Proper Schwinger Physics

This implements the correct Schwinger pair production rate:
Γ = (eE)²/(2π)³ × exp[-π m²/(eE)]

And tests polymer-QED modifications:
Γ_poly = Γ_standard × f(μ,E)

The key question: Do polymer corrections make pair production 
observable at field strengths E << E_Schwinger = m²/e ≈ 1.3×10¹⁶ V/m?
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Physical constants (SI and natural units)
ELECTRON_MASS = 9.109e-31       # kg
ELECTRON_CHARGE = 1.602e-19     # C
HBAR = 1.055e-34               # J·s
C_LIGHT = 2.998e8              # m/s
ALPHA_EM = 1/137.036            # Fine structure constant

# Natural units
ELECTRON_MASS_GEV = 0.511e-3    # GeV
E_PLANCK_GEV = 1.22e19          # GeV

# Schwinger critical field
E_SCHWINGER = ELECTRON_MASS**2 * C_LIGHT**3 / (ELECTRON_CHARGE * HBAR)  # V/m
print(f"Schwinger critical field: {E_SCHWINGER:.2e} V/m")

class RealisticVacuumAnalysis:
    """Physically accurate vacuum instability analysis."""
    
    def __init__(self):
        self.critical_threshold = 1e-50  # Rate threshold for "observable"
    
    def schwinger_rate_prefactor(self, E_field_V_per_m):
        """
        Calculate the Schwinger rate prefactor: (eE)²/(2π)³
        
        This gives the actual rate, not just the exponential part.
        """
        # Convert to natural units
        eE = ELECTRON_CHARGE * E_field_V_per_m  # J/m
        eE_natural = eE / (HBAR * C_LIGHT)      # 1/m
        
        prefactor = (eE_natural)**2 / (2 * np.pi)**3
        return prefactor
    
    def schwinger_exponent_standard(self, E_field_V_per_m):
        """Standard Schwinger exponential: exp[-π m²/(eE)]"""
        # Critical field ratio
        field_ratio = E_field_V_per_m / E_SCHWINGER
        
        # Exponent: -π m²/(eE) = -π (E_crit/E)
        exponent = -np.pi / field_ratio
        return exponent
    
    def polymer_enhancement_factor(self, E_field_V_per_m, mu_GeV, model='linear'):
        """
        Polymer-QED enhancement factor for pair production.
        
        Different models for how polymer scale μ modifies vacuum structure.
        """
        # Field strength in GeV units (rough conversion)
        E_GeV = E_field_V_per_m * ELECTRON_CHARGE * HBAR * C_LIGHT / (1.602e-10)  # Very rough
        
        # Dimensionless parameter
        x = E_GeV / mu_GeV
        
        if model == 'linear':
            # Linear enhancement: exp[α × (E/μ)]
            alpha = 1.0  # Coupling strength
            enhancement = np.exp(alpha * x)
            
        elif model == 'power_law':
            # Power law enhancement: (E/μ)^β
            beta = 2.0
            enhancement = x**beta if x > 0 else 1.0
            
        elif model == 'threshold':
            # Threshold behavior: exp[(E-μ)/μ] for E > μ
            if E_GeV > mu_GeV:
                enhancement = np.exp((E_GeV - mu_GeV) / mu_GeV)
            else:
                enhancement = 1.0
                
        elif model == 'suppression':
            # Exponential suppression: exp[-μ/E]
            enhancement = np.exp(-mu_GeV / E_GeV) if E_GeV > 0 else 0.0
            
        else:
            enhancement = 1.0
        
        return enhancement
    
    def total_pair_production_rate(self, E_field_V_per_m, mu_GeV=None, model='none'):
        """
        Calculate total pair production rate.
        
        Returns:
        --------
        rate : float
            Actual pair production rate (pairs/m³/s or similar units)
        """
        # Standard Schwinger components
        prefactor = self.schwinger_rate_prefactor(E_field_V_per_m)
        exponent = self.schwinger_exponent_standard(E_field_V_per_m)
        
        # Standard rate
        standard_rate = prefactor * np.exp(exponent)
        
        if model == 'none' or mu_GeV is None:
            return standard_rate
        
        # Polymer enhancement
        enhancement = self.polymer_enhancement_factor(E_field_V_per_m, mu_GeV, model)
        
        return standard_rate * enhancement
    
    def scan_realistic_parameter_space(self):
        """Scan parameter space with realistic physics."""
        print("REALISTIC VACUUM INSTABILITY PARAMETER SCAN")
        print("=" * 60)
        print(f"Schwinger critical field: E_crit = {E_SCHWINGER:.2e} V/m")
        print("Testing polymer-QED enhancements to pair production")
        print("=" * 60)
        
        # Parameter ranges
        field_strengths = np.logspace(10, 18, 40)    # 10¹⁰ to 10¹⁸ V/m
        polymer_scales = np.logspace(12, 20, 40)     # 10¹² to 10²⁰ GeV
        
        models = ['none', 'linear', 'power_law', 'threshold', 'suppression']
        
        results = {}
        
        for model in models:
            print(f"\nModel: {model.upper()}")
            print("-" * 30)
            
            # Track observable cases
            observable_cases = []
            rate_grid = np.zeros((len(polymer_scales), len(field_strengths)))
            
            for i, mu in enumerate(polymer_scales):
                for j, E_field in enumerate(field_strengths):
                    
                    if model == 'none':
                        rate = self.total_pair_production_rate(E_field)
                    else:
                        rate = self.total_pair_production_rate(E_field, mu, model)
                    
                    rate_grid[i, j] = rate
                    
                    # Check if observable
                    if rate > self.critical_threshold:
                        observable_cases.append({
                            'mu_GeV': mu,
                            'E_field_V_per_m': E_field,
                            'rate': rate,
                            'enhancement': rate / self.total_pair_production_rate(E_field) if model != 'none' else 1.0
                        })
            
            results[model] = {
                'observable_cases': observable_cases,
                'rate_grid': rate_grid,
                'polymer_scales': polymer_scales,
                'field_strengths': field_strengths
            }
            
            print(f"Observable cases: {len(observable_cases)}")
            
            if observable_cases:
                # Find laboratory-accessible cases (E < 10¹⁵ V/m)
                lab_cases = [case for case in observable_cases if case['E_field_V_per_m'] < 1e15]
                
                if lab_cases:
                    best_lab = min(lab_cases, key=lambda x: x['E_field_V_per_m'])
                    print(f"🎯 Best laboratory case:")
                    print(f"   μ = {best_lab['mu_GeV']:.1e} GeV")
                    print(f"   E = {best_lab['E_field_V_per_m']:.1e} V/m")
                    print(f"   Enhancement = {best_lab['enhancement']:.1e}")
                else:
                    print("❌ No laboratory-accessible cases")
            else:
                print("❌ No observable cases found")
        
        return results
    
    def laboratory_feasibility_analysis(self, results):
        """Analyze feasibility of observing vacuum instability in laboratory."""
        print(f"\n{'=' * 60}")
        print("LABORATORY FEASIBILITY ANALYSIS")
        print(f"{'=' * 60}")
        
        # Laboratory field benchmarks
        lab_benchmarks = {
            'Current high-intensity laser': 1e13,      # V/m
            'Next-generation laser': 1e14,             # V/m
            'Theoretical limit': 1e15,                 # V/m
        }
        
        print("Field Strength | Standard Rate | Best Polymer Enhancement")
        print("-" * 60)
        
        for name, E_field in lab_benchmarks.items():
            standard_rate = self.total_pair_production_rate(E_field)
            
            # Find best polymer enhancement at this field
            best_enhancement = 1.0
            best_model = 'none'
            
            for model in ['linear', 'power_law', 'threshold', 'suppression']:
                if model in results:
                    cases = results[model]['observable_cases']
                    field_cases = [c for c in cases if abs(c['E_field_V_per_m'] - E_field) / E_field < 0.1]
                    
                    if field_cases:
                        best_case = max(field_cases, key=lambda x: x['enhancement'])
                        if best_case['enhancement'] > best_enhancement:
                            best_enhancement = best_case['enhancement']
                            best_model = model
            
            print(f"{name:20} | {standard_rate:.2e} | {best_enhancement:.2e} ({best_model})")
            
            # Check observability
            enhanced_rate = standard_rate * best_enhancement
            if enhanced_rate > self.critical_threshold:
                print(f"                     → ✅ OBSERVABLE with {best_model} polymer corrections!")
            else:
                print(f"                     → ❌ Still below threshold")

def create_parameter_space_plot(results):
    """Create realistic parameter space visualization."""
    print(f"\nGenerating realistic parameter space plots...")
    
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    models = ['none', 'linear', 'power_law', 'threshold']
    
    for i, model in enumerate(models):
        if model not in results:
            continue
            
        ax = axes[i]
        data = results[model]
        
        # Create rate contour plot
        X, Y = np.meshgrid(
            np.log10(data['field_strengths']),
            np.log10(data['polymer_scales'])
        )
        Z = np.log10(data['rate_grid'] + 1e-100)  # Add small value to avoid log(0)
        
        # Plot rate contours
        levels = np.arange(-100, 0, 10)
        cs = ax.contour(X, Y, Z, levels=levels, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=8)
        
        # Highlight observable threshold
        observable_level = np.log10(1e-50)
        ax.contour(X, Y, Z, levels=[observable_level], colors='red', linewidths=3)
        
        # Laboratory field reference lines
        ax.axvline(np.log10(1e13), color='green', linestyle='--', alpha=0.7, label='High laser')
        ax.axvline(np.log10(1e15), color='orange', linestyle='--', alpha=0.7, label='Extreme laser') 
        ax.axvline(np.log10(E_SCHWINGER), color='black', linestyle='--', alpha=0.7, label='Schwinger')
        
        ax.set_xlabel('log₁₀(Electric Field [V/m])')
        ax.set_ylabel('log₁₀(Polymer Scale μ [GeV])')
        ax.set_title(f'{model.title()} Model')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/realistic_vacuum_instability_scan.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Realistic parameter scan plot saved: results/realistic_vacuum_instability_scan.png")

def main():
    """Run realistic vacuum instability analysis."""
    
    analyzer = RealisticVacuumAnalysis()
    
    # Perform parameter scan with realistic physics
    results = analyzer.scan_realistic_parameter_space()
    
    # Analyze laboratory feasibility  
    analyzer.laboratory_feasibility_analysis(results)
    
    # Create visualization
    create_parameter_space_plot(results)
    
    print(f"\n{'=' * 60}")
    print("REALISTIC VACUUM INSTABILITY ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print("Summary:")
    print("• Used proper Schwinger pair production physics")
    print("• Tested multiple polymer-QED enhancement models")
    print("• Evaluated laboratory accessibility")
    print("• Generated realistic parameter space maps")
    
    return results

if __name__ == "__main__":
    results = main()
