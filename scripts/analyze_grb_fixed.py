#!/usr/bin/env python3
"""
Enhanced GRB Analysis with 3-Term Polynomial Dispersion Relations

This module implements polynomial dispersion relations: Δt = b + α₁(E/E_Pl) + α₂(E/E_Pl)²
using matrix fitting as specified.

Key features:
- 3-term polynomial fits with model selection based on AIC
- Enhanced uncertainty quantification
- Theoretical model constraints
"""

import glob
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

# Physical constants
E_PLANCK = 1.22e19  # GeV
D = 1e17  # dispersion factor (s)

def analyze_grb_polynomial(path, max_terms=3):
    """
    Polynomial GRB analysis with up to 3 terms.
    
    Model: Δt = b + α₁(E/E_Pl) + α₂(E/E_Pl)²
    
    This directly implements the requested polynomial expansion
    using matrix fitting as specified.
    """
    # Load GRB data
    df = pd.read_csv(path)
    energies = df['delta_E_GeV'].values  # E in GeV
    times = df['delta_t_s'].values       # Δt in seconds
    
    # Normalize energies by Planck scale
    E_over_Epl = energies / E_PLANCK
    
    print(f"Analyzing {len(energies)} data points")
    print(f"Energy range: {energies.min():.2f} - {energies.max():.2f} GeV")
    print(f"E/E_Pl range: {E_over_Epl.min():.2e} - {E_over_Epl.max():.2e}")
    
    results = {}
    
    # Test different polynomial orders (1, 2, 3 terms)
    for n_terms in range(1, max_terms + 1):
        print(f"\nFitting {n_terms}-term model:")
        
        if n_terms == 1:
            # Linear model: Δt = α₁(E/E_Pl)
            X = E_over_Epl.reshape(-1, 1)
            model_name = "linear"
            param_names = ["α₁"]
            
        elif n_terms == 2:
            # Linear + constant: Δt = b + α₁(E/E_Pl)
            X = np.vstack([np.ones_like(E_over_Epl), E_over_Epl]).T
            model_name = "linear_with_constant"
            param_names = ["b", "α₁"]
            
        elif n_terms == 3:
            # Full 3-term model: Δt = b + α₁(E/E_Pl) + α₂(E/E_Pl)²
            X = np.vstack([np.ones_like(E_over_Epl), E_over_Epl, (E_over_Epl)**2]).T
            model_name = "quadratic_with_constant"
            param_names = ["b", "α₁", "α₂"]
        
        try:
            # Perform least squares fit
            coeffs, residuals, rank, s = np.linalg.lstsq(X, times, rcond=None)
            fitted_times = X @ coeffs
            
            # Calculate goodness of fit metrics
            n_data = len(times)
            n_params = len(coeffs)
            
            # Chi-squared
            chi2 = np.sum((times - fitted_times)**2)
            chi2_reduced = chi2 / (n_data - n_params) if n_data > n_params else float('inf')
            
            # R-squared
            ss_res = np.sum((times - fitted_times)**2)
            ss_tot = np.sum((times - np.mean(times))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # AIC (Akaike Information Criterion)
            if ss_res > 0:
                aic = n_data * np.log(ss_res / n_data) + 2 * n_params
            else:
                aic = -float('inf')  # Perfect fit
            
            # Parameter uncertainties (diagonal of covariance matrix)
            try:
                # Covariance matrix: σ² (X^T X)^(-1)
                sigma2 = ss_res / (n_data - n_params) if n_data > n_params else 1.0
                cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
                param_errors = np.sqrt(np.diag(cov_matrix))
            except:
                param_errors = np.full_like(coeffs, np.nan)
            
            # Store results
            results[model_name] = {
                'n_terms': n_terms,
                'coefficients': coeffs,
                'param_names': param_names,
                'param_errors': param_errors,
                'chi2': chi2,
                'chi2_reduced': chi2_reduced,
                'r_squared': r_squared,
                'aic': aic,
                'fitted_times': fitted_times,
                'residuals': times - fitted_times
            }
            
            # Print fit results
            print(f"  Model: {model_name}")
            print(f"  Coefficients:")
            for i, (name, coeff, error) in enumerate(zip(param_names, coeffs, param_errors)):
                print(f"    {name} = {coeff:.6e} ± {error:.6e}")
            print(f"  χ²/dof = {chi2_reduced:.6f}")
            print(f"  R² = {r_squared:.6f}")
            print(f"  AIC = {aic:.2f}")
            
            # Extract LIV energy scales from linear coefficient α₁
            if n_terms == 1 and len(coeffs) >= 1:
                # For 1-term linear model: Δt = α₁(E/E_Pl)
                alpha1 = coeffs[0]  # α₁ coefficient
                alpha1_err = param_errors[0]
                
                if abs(alpha1) > 0:
                    E_LV_est = D / abs(alpha1)
                    E_LV_error = D * alpha1_err / (alpha1**2) if alpha1_err > 0 else float('inf')
                    
                    results[model_name]['E_LV_scale'] = E_LV_est
                    results[model_name]['E_LV_error'] = E_LV_error
                    
                    print(f"  E_LV = {E_LV_est:.2e} ± {E_LV_error:.2e} GeV")
                    
            elif n_terms >= 2:  # Has α₁ term
                alpha1_idx = 1  # Position of α₁ coefficient
                alpha1 = coeffs[alpha1_idx]
                alpha1_err = param_errors[alpha1_idx]
                
                if abs(alpha1) > 0:
                    # E_LV scale from α₁: α₁ = D(z)/E_LV
                    E_LV_est = D / abs(alpha1)
                    E_LV_error = D * alpha1_err / (alpha1**2) if alpha1_err > 0 else float('inf')
                    
                    results[model_name]['E_LV_scale'] = E_LV_est
                    results[model_name]['E_LV_error'] = E_LV_error
                    
                    print(f"  E_LV = {E_LV_est:.2e} ± {E_LV_error:.2e} GeV")
            
        except Exception as e:
            print(f"  Error fitting {n_terms}-term model: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

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

def main():
    """Main analysis pipeline with 3-term polynomial fitting"""
    results = []
    
    print("Running Enhanced GRB Analysis with 3-Term Polynomial Dispersion Relations")
    print("Model: Δt = b + α₁(E/E_Pl) + α₂(E/E_Pl)²")
    print("=" * 70)
    
    # Process all GRB files
    grb_files = glob.glob(os.path.join("data", "grbs", "*.csv"))
    if not grb_files:
        print("Warning: No GRB files found in data/grbs/")
        return
    
    for path in grb_files:
        print(f"\nAnalyzing: {path}")
        
        try:
            # Run polynomial analysis
            poly_results = analyze_grb_polynomial(path, max_terms=3)
            
            # Determine best model based on AIC
            best_model = None
            best_aic = float('inf')
            
            for model_name, model_data in poly_results.items():
                if isinstance(model_data, dict) and 'aic' in model_data:
                    if model_data['aic'] < best_aic:
                        best_aic = model_data['aic']
                        best_model = model_name
            
            if best_model and best_model in poly_results:
                best_result = poly_results[best_model]
                
                # Format results for output
                grb_result = {
                    'GRB_file': path.split(os.sep)[-1],
                    'best_model': best_model,
                    'n_terms': best_result.get('n_terms', 'N/A'),
                    'aic': best_result.get('aic', 'N/A'),
                    'chi2_dof': best_result.get('chi2_dof', 'N/A'),
                    'r_squared': best_result.get('r_squared', 'N/A')
                }
                
                # Add coefficients
                coeffs = best_result.get('coefficients', [])
                param_names = best_result.get('param_names', [])
                param_errors = best_result.get('param_errors', [])
                
                for i, (name, coeff, error) in enumerate(zip(param_names, coeffs, param_errors)):
                    grb_result[f'{name}'] = coeff
                    grb_result[f'{name}_error'] = error
                
                # Extract LIV energy scale if available
                if 'E_LV_scale' in best_result:
                    grb_result['E_LV_est_GeV'] = best_result['E_LV_scale']
                    grb_result['E_LV_error_GeV'] = best_result.get('E_LV_error', 'N/A')
                else:
                    grb_result['E_LV_est_GeV'] = 'N/A'
                    grb_result['E_LV_error_GeV'] = 'N/A'
                
                results.append(grb_result)
                
                # Print summary
                print(f"  Best model: {best_model}")
                print(f"  Number of terms: {best_result.get('n_terms', 'N/A')}")
                
                chi2_dof = best_result.get('chi2_dof', 'N/A')
                r_squared = best_result.get('r_squared', 'N/A')
                aic = best_result.get('aic', 'N/A')
                
                if isinstance(chi2_dof, (int, float)):
                    print(f"  χ²/dof: {chi2_dof:.6f}")
                else:
                    print(f"  χ²/dof: {chi2_dof}")
                    
                if isinstance(r_squared, (int, float)):
                    print(f"  R²: {r_squared:.6f}")
                else:
                    print(f"  R²: {r_squared}")
                    
                if isinstance(aic, (int, float)):
                    print(f"  AIC: {aic:.2f}")
                else:
                    print(f"  AIC: {aic}")
                
                if 'E_LV_scale' in best_result:
                    E_LV_scale = best_result['E_LV_scale']
                    E_LV_error = best_result.get('E_LV_error', 0)
                    if isinstance(E_LV_scale, (int, float)) and isinstance(E_LV_error, (int, float)):
                        print(f"  E_LV scale: {E_LV_scale:.2e} ± {E_LV_error:.2e} GeV")
                        
            else:
                print("  Warning: No valid polynomial models found, using fallback")
                # Fallback to basic analysis
                basic_result = analyze_grb_basic(path)
                results.append({
                    'GRB_file': path.split(os.sep)[-1],
                    'best_model': 'linear_fallback',
                    'E_LV_est_GeV': basic_result['E_LV_est'],
                    'E_LV_lower95_GeV': basic_result.get('E_LV_lower95', 'N/A'),
                    'aic': 'N/A',
                    'chi2_dof': 'N/A'
                })
                
        except Exception as e:
            print(f"  Error analyzing {path}: {e}")
            # Fallback to basic analysis
            basic_result = analyze_grb_basic(path)
            results.append({
                'GRB_file': path.split(os.sep)[-1],
                'best_model': 'linear_fallback',
                'E_LV_est_GeV': basic_result['E_LV_est'],
                'E_LV_lower95_GeV': basic_result.get('E_LV_lower95', 'N/A'),
                'aic': 'N/A',
                'chi2_dof': 'N/A'
            })
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        
        # Save detailed polynomial results
        df.to_csv(os.path.join("results", "grb_polynomial_bounds.csv"), index=False)
        
        # Also save simplified version for compatibility
        simple_cols = ['GRB_file', 'E_LV_est_GeV', 'best_model']
        available_cols = [col for col in simple_cols if col in df.columns]
        if available_cols:
            simple_df = df[available_cols].copy()
            simple_df.to_csv(os.path.join("results", "grb_bounds.csv"), index=False)
        
        print(f"\n{'='*70}")
        print("3-TERM POLYNOMIAL GRB ANALYSIS RESULTS:")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        print(f"\nResults saved to:")
        print(f"  - results/grb_polynomial_bounds.csv (detailed)")
        print(f"  - results/grb_bounds.csv (simplified)")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()
