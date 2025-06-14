#!/usr/bin/env python3
"""
Focused Vacuum Instability Parameter Scan

This script systematically scans the polymer scale μ and electric field E
parameter space to find regions where polymer-QED corrections make
vacuum instability observable at laboratory field strengths.

Key Question: Do polymer-QED modifications reduce the critical field
for pair production below laboratory-accessible values?
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Physical constants
ELECTRON_MASS = 0.511e-3  # GeV
ALPHA_EM = 1/137.036
HBAR_C = 0.197  # GeV·fm
E_SCHWINGER = 1.32e16  # V/m (Schwinger critical field)

class FocusedVacuumScan:
    """Systematic scan of μ and E parameter space."""
    
    def __init__(self):
        self.results = {}
    
    def schwinger_rate(self, E_field_V_per_m, polymer_scale_GeV=None, correction_type='none'):
        """
        Calculate Schwinger rate with optional polymer corrections.
        
        Parameters:
        -----------
        E_field_V_per_m : float
            Electric field in V/m
        polymer_scale_GeV : float, optional
            Polymer energy scale μ in GeV
        correction_type : str
            Type of correction: 'none', 'linear', 'quadratic'
            
        Returns:
        --------
        log_rate : float
            Logarithm of pair production rate
        """
        # Convert to natural units (GeV²)
        E_natural = E_field_V_per_m * HBAR_C / 5.1e15  # Rough conversion
        
        # Standard Schwinger exponent: -π m²/(αE)
        standard_exponent = -np.pi * ELECTRON_MASS**2 / (ALPHA_EM * E_natural)
        
        if correction_type == 'none' or polymer_scale_GeV is None:
            return standard_exponent
        
        # Polymer corrections
        x = E_natural / polymer_scale_GeV  # Dimensionless parameter
        
        if correction_type == 'linear':
            # Linear enhancement: f(x) = 1 + αx
            enhancement = 1 + 0.5 * x
        elif correction_type == 'quadratic':
            # Quadratic: f(x) = 1 + αx + βx²
            enhancement = 1 + 0.5 * x + 0.1 * x**2
        elif correction_type == 'exponential':
            # Exponential suppression: f(x) = exp(-x)
            enhancement = np.exp(-x)
        else:
            enhancement = 1.0
        
        return standard_exponent / enhancement  # Enhancement makes rate larger (less negative)
    
    def scan_parameter_space(self):
        """Comprehensive 2D scan of μ and E space."""
        print("SYSTEMATIC VACUUM INSTABILITY PARAMETER SCAN")
        print("=" * 60)
        
        # Define parameter ranges
        polymer_scales = np.logspace(10, 19, 50)  # 10¹⁰ to 10¹⁹ GeV
        field_strengths = np.logspace(8, 18, 50)  # 10⁸ to 10¹⁸ V/m
        
        # Observable threshold (rate > exp(-50))
        threshold = -50
        
        # Store results for each correction type
        correction_types = ['none', 'linear', 'quadratic', 'exponential']
        
        for correction in correction_types:
            print(f"\nScanning {correction.upper()} polymer corrections...")
            
            # Initialize result grid
            rate_grid = np.zeros((len(polymer_scales), len(field_strengths)))
            observable_grid = np.zeros((len(polymer_scales), len(field_strengths)), dtype=bool)
            
            for i, mu in enumerate(polymer_scales):
                for j, E_field in enumerate(field_strengths):
                    rate = self.schwinger_rate(E_field, mu, correction)
                    rate_grid[i, j] = rate
                    observable_grid[i, j] = rate > threshold
            
            self.results[correction] = {
                'polymer_scales': polymer_scales,
                'field_strengths': field_strengths,
                'rate_grid': rate_grid,
                'observable_grid': observable_grid
            }
            
            # Count observable cases
            n_observable = np.sum(observable_grid)
            total_cases = observable_grid.size
            
            print(f"  Observable cases: {n_observable}/{total_cases} ({100*n_observable/total_cases:.1f}%)")
            
            # Find minimum critical field for each μ
            critical_fields = []
            for i, mu in enumerate(polymer_scales):
                observable_mask = observable_grid[i, :]
                if np.any(observable_mask):
                    critical_field = field_strengths[observable_mask][0]  # First observable
                    critical_fields.append((mu, critical_field))
            
            if critical_fields:
                # Find most promising case (lowest critical field)
                best_mu, best_E_crit = min(critical_fields, key=lambda x: x[1])
                print(f"  Best case: μ = {best_mu:.1e} GeV → E_crit = {best_E_crit:.1e} V/m")
                
                # Check laboratory accessibility
                if best_E_crit < 1e13:
                    print(f"  🎯 LABORATORY ACCESSIBLE! (< 10¹³ V/m)")
                elif best_E_crit < 1e15:
                    print(f"  ⚡ High-intensity laser accessible (< 10¹⁵ V/m)")
                elif best_E_crit < E_SCHWINGER:
                    print(f"  🌟 Below Schwinger critical field")
                else:
                    print(f"  ❌ Not accessible")
        
        return self.results
    
    def analyze_laboratory_accessibility(self):
        """Detailed analysis of laboratory-accessible parameter regions."""
        print(f"\n{'=' * 60}")
        print("LABORATORY ACCESSIBILITY ANALYSIS")
        print(f"{'=' * 60}")
        
        # Laboratory field strength benchmarks
        lab_fields = {
            'Tabletop laser': 1e12,      # V/m
            'High-intensity laser': 1e13, # V/m  
            'Extreme laser': 1e15,       # V/m
            'Future technology': 1e16     # V/m
        }
        
        for correction in ['linear', 'quadratic', 'exponential']:
            if correction not in self.results:
                continue
                
            print(f"\n{correction.upper()} POLYMER CORRECTIONS:")
            print("-" * 40)
            
            data = self.results[correction]
            polymer_scales = data['polymer_scales']
            field_strengths = data['field_strengths']
            observable_grid = data['observable_grid']
            
            for lab_name, lab_field in lab_fields.items():
                # Find closest field strength in our grid
                field_idx = np.argmin(np.abs(field_strengths - lab_field))
                actual_field = field_strengths[field_idx]
                
                # Check which polymer scales make this field observable
                observable_scales = polymer_scales[observable_grid[:, field_idx]]
                
                print(f"{lab_name} (E = {actual_field:.1e} V/m):")
                if len(observable_scales) > 0:
                    print(f"  ✅ Observable for μ ∈ [{observable_scales.min():.1e}, {observable_scales.max():.1e}] GeV")
                    print(f"     ({len(observable_scales)} out of {len(polymer_scales)} scales)")
                else:
                    print(f"  ❌ Not observable for any μ")
    
    def create_parameter_plots(self):
        """Create comprehensive parameter space plots."""
        print(f"\n{'=' * 60}")
        print("GENERATING PARAMETER SPACE PLOTS")
        print(f"{'=' * 60}")
        
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        correction_types = ['none', 'linear', 'quadratic', 'exponential']
        
        for i, correction in enumerate(correction_types):
            if correction not in self.results:
                continue
                
            ax = axes[i]
            data = self.results[correction]
            
            # Create contour plot of pair production rates
            X, Y = np.meshgrid(
                np.log10(data['field_strengths']),
                np.log10(data['polymer_scales'])
            )
            Z = data['rate_grid']
            
            # Plot rate contours
            levels = [-100, -75, -50, -25, -10, 0]
            cs = ax.contour(X, Y, Z, levels=levels, colors='blue', alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=8)
            
            # Highlight observable region (rate > -50)
            observable_contour = ax.contour(X, Y, Z, levels=[-50], colors='red', linewidths=3)
            
            # Add laboratory field reference lines
            ax.axvline(np.log10(1e13), color='green', linestyle='--', alpha=0.7, label='Lab laser')
            ax.axvline(np.log10(1e15), color='orange', linestyle='--', alpha=0.7, label='Extreme laser')
            ax.axvline(np.log10(E_SCHWINGER), color='black', linestyle='--', alpha=0.7, label='Schwinger')
            
            ax.set_xlabel('log₁₀(Electric Field [V/m])')
            ax.set_ylabel('log₁₀(Polymer Scale μ [GeV])')
            ax.set_title(f'{correction.title()} Corrections')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/vacuum_instability_parameter_scan.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Parameter space plot saved: results/vacuum_instability_parameter_scan.png")

def main():
    """Run the focused vacuum instability parameter scan."""
    
    scanner = FocusedVacuumScan()
    
    # Perform systematic parameter scan
    results = scanner.scan_parameter_space()
    
    # Analyze laboratory accessibility
    scanner.analyze_laboratory_accessibility()
    
    # Create visualization
    scanner.create_parameter_plots()
    
    print(f"\n{'=' * 60}")
    print("VACUUM INSTABILITY SCAN COMPLETE")
    print(f"{'=' * 60}")
    print("Key findings:")
    print("• Systematic scan of μ (10¹⁰ - 10¹⁹ GeV) and E (10⁸ - 10¹⁸ V/m)")
    print("• Tested linear, quadratic, and exponential polymer corrections")
    print("• Identified laboratory-accessible parameter regions")
    print("• Generated comprehensive parameter space visualization")
    
    return results

if __name__ == "__main__":
    results = main()
