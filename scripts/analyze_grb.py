#!/usr/bin/env python3
"""
Enhanced GRB Analysis with Polynomial Dispersion Relations

This module replaces the simple linear time delay model with polynomial 
dispersion relations that can test gravity-rainbow and polymer-QED models.

Key features:
- Polynomial fits: Δt = D(z)[α₁(E/E_Pl) + α₂(E²/E_Pl²) + ...]
- Model comparison (linear, quadratic, cubic, quartic)  
- Theoretical model constraints
- Enhanced uncertainty quantification
"""

import glob
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

# Import enhanced polynomial fitting capabilities
try:
    from enhanced_grb_analysis import PolynomialDispersionFitter, TheoreticalModelTester
    ENHANCED_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced analysis modules not available, using basic linear fit")
    ENHANCED_AVAILABLE = False

# Physical constants
E_PLANCK = 1.22e19  # GeV
D = 1e17  # dispersion factor (s)

def analyze_grb_basic(path):
    """Basic linear dispersion analysis (fallback)"""
    df = pd.read_csv(path)
    delta_E = df['delta_E_GeV'].values
    delta_t = df['delta_t_s'].values
    
    # Linear regression using scipy.stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(delta_E, delta_t)
    
    # LIV energy scale estimates
    E_LV_est   = D / abs(slope) if slope != 0 else float('inf')
    E_LV_lower = D / (abs(slope) + 1.96*std_err) if (abs(slope) + 1.96*std_err) != 0 else float('inf')
    
    return {'E_LV_est': E_LV_est, 'E_LV_lower95': E_LV_lower, 'model': 'linear'}

def analyze_grb_enhanced(path, max_order=3):
    """Enhanced polynomial dispersion analysis"""
    if not ENHANCED_AVAILABLE:
        return analyze_grb_basic(path)
    
    # Load data
    df = pd.read_csv(path)
    energies = df['delta_E_GeV'].values
    times = df['delta_t_s'].values
    
    # Initialize polynomial fitter
    fitter = PolynomialDispersionFitter(distance_factor=D)
    
    # Fit multiple polynomial orders
    results = {}
    best_order = 1
    best_aic = float('inf')
    
    for order in range(1, max_order + 1):
        try:
            fit_result = fitter.fit_polynomial(energies, times, order)
            results[f'order_{order}'] = fit_result
            
            # Model selection using AIC
            if fit_result['aic'] < best_aic:
                best_aic = fit_result['aic']
                best_order = order
                
        except Exception as e:
            print(f"Warning: Order {order} fit failed for {path}: {e}")
            continue
    
    # Return best fit results
    if f'order_{best_order}' in results:
        best_result = results[f'order_{best_order}']
        return {
            'E_LV_est': best_result.get('E_LV_scale', float('inf')),
            'E_LV_lower95': best_result.get('E_LV_lower', float('inf')),
            'model': f'polynomial_order_{best_order}',
            'coefficients': best_result.get('coefficients', []),
            'aic': best_result.get('aic', float('inf')),
            'chi2_dof': best_result.get('chi2_dof', float('inf')),
            'all_orders': results
        }
    else:
        return analyze_grb_basic(path)

def analyze_grb(path, use_enhanced=True, max_order=3):
    """Main GRB analysis function with model selection"""
    if use_enhanced and ENHANCED_AVAILABLE:
        return analyze_grb_enhanced(path, max_order)
    else:
        return analyze_grb_basic(path)

def main():
    """Main analysis pipeline with enhanced polynomial fitting"""
    results = []
    
    print("Running Enhanced GRB Analysis with Polynomial Dispersion Relations")
    print("=" * 70)
    
    # Process all GRB files
    grb_files = glob.glob(os.path.join("data", "grbs", "*.csv"))
    if not grb_files:
        print("Warning: No GRB files found in data/grbs/")
        return
    
    for path in grb_files:
        print(f"\nAnalyzing: {path}")
        
        try:
            # Run enhanced analysis
            result = analyze_grb(path, use_enhanced=True, max_order=3)
            
            # Format results for output
            grb_result = {
                'GRB_file': path.split(os.sep)[-1],
                'E_LV_est_GeV': result['E_LV_est'],
                'E_LV_lower95_GeV': result['E_LV_lower95'],
                'best_model': result.get('model', 'linear'),
                'aic': result.get('aic', 'N/A'),
                'chi2_dof': result.get('chi2_dof', 'N/A')
            }
            
            # Add polynomial coefficients if available
            coeffs = result.get('coefficients', [])
            for i, coeff in enumerate(coeffs):
                grb_result[f'alpha_{i+1}'] = coeff
            
            results.append(grb_result)
            
            # Print summary
            print(f"  Best model: {result.get('model', 'linear')}")
            print(f"  E_LV estimate: {result['E_LV_est']:.2e} GeV")
            print(f"  E_LV 95% lower: {result['E_LV_lower95']:.2e} GeV")
            
        except Exception as e:
            print(f"  Error analyzing {path}: {e}")
            # Fallback to basic analysis
            basic_result = analyze_grb_basic(path)
            results.append({
                'GRB_file': path.split(os.sep)[-1],
                'E_LV_est_GeV': basic_result['E_LV_est'],
                'E_LV_lower95_GeV': basic_result['E_LV_lower95'],
                'best_model': 'linear_fallback',
                'aic': 'N/A',
                'chi2_dof': 'N/A'
            })
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        
        # Save enhanced results
        df.to_csv(os.path.join("results", "grb_enhanced_bounds.csv"), index=False)
        
        # Also save simplified version for compatibility
        simple_df = df[['GRB_file', 'E_LV_est_GeV', 'E_LV_lower95_GeV']].copy()
        simple_df.columns = ['GRB file', 'E_LV_est (GeV)', 'E_LV_lower95 (GeV)']
        simple_df.to_csv(os.path.join("results", "grb_bounds.csv"), index=False)
        
        print(f"\n{'='*70}")
        print("ENHANCED GRB ANALYSIS RESULTS:")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        print(f"\nResults saved to:")
        print(f"  - results/grb_enhanced_bounds.csv (detailed)")
        print(f"  - results/grb_bounds.csv (simplified)")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()
