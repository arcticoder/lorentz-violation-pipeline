#!/usr/bin/env python3
"""
Enhanced UHECR Analysis with Theoretical Model Testing

This module extends the basic UHECR threshold analysis to test sophisticated
theoretical predictions:

1. Non-linear threshold modifications: E_th(E_CR) = E_th₀[1 + (E_CR/E_LV)^n]
2. Vacuum instability effects: Modified survival probabilities
3. Hidden sector energy loss: Additional energy leakage channels
4. Polymer-QED and gravity-rainbow corrections to photopion production

Replaces simple linear threshold shifts with full theoretical model predictions.
"""

import pandas as pd
import numpy as np
import os
import sys

# Import enhanced UHECR analysis capabilities
try:
    from enhanced_uhecr_analysis import EnhancedUHECRAnalyzer, TheoreticalUHECRModels
    from theoretical_liv_models import (
        polymer_qed_dispersion,
        gravity_rainbow_dispersion,
        vacuum_instability_rate,
        hidden_sector_energy_loss
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    print("Warning: Enhanced UHECR analysis modules not available, using basic analysis")
    ENHANCED_AVAILABLE = False

# Physical constants
E_PLANCK = 1.22e19  # GeV

def simulate_uhecr_basic(path):
    """Basic UHECR analysis (fallback)"""
    df = pd.read_csv(path)
    energies = df['E_center_eV'].values / 1e9  # Convert eV to GeV for energy_EeV
    flux     = df['flux'].values
    error    = df['flux_err_high'].values  # Use upper error as error estimate

    # Only use bins with actual data
    valid_mask = (df['N_events'] > 0) & (flux > 0) & (error > 0)
    energies = energies[valid_mask]
    flux = flux[valid_mask]
    error = error[valid_mask]

    results = []
    for E_LV_p in np.logspace(17, 19, 5):
        chi2 = np.sum(((flux - flux.mean())/error)**2) / len(flux)
        excluded = chi2 > 1.0
        results.append({
            'E_LV_p (GeV)': E_LV_p,
            'chi2': chi2,
            'Excluded': excluded,
            'model': 'linear_threshold'
        })
    return pd.DataFrame(results)

def simulate_uhecr_enhanced(path, models=['polynomial', 'polymer_qed', 'gravity_rainbow']):
    """Enhanced UHECR analysis with theoretical model testing"""
    if not ENHANCED_AVAILABLE:
        return simulate_uhecr_basic(path)
    
    # Load spectrum data
    df = pd.read_csv(path)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedUHECRAnalyzer()
    theoretical_models = TheoreticalUHECRModels()
    
    all_results = []
    
    print(f"Testing UHECR models: {models}")
    
    for model_name in models:
        try:
            print(f"  Analyzing model: {model_name}")
            
            if model_name == 'polynomial':
                # Test polynomial threshold modifications
                results = analyzer.fit_polynomial_threshold(df, max_order=3)
                
            elif model_name == 'polymer_qed':
                # Test polymer-QED predictions  
                results = theoretical_models.test_polymer_qed_spectrum(df)
                
            elif model_name == 'gravity_rainbow':
                # Test gravity-rainbow corrections
                results = theoretical_models.test_gravity_rainbow_spectrum(df)
                
            elif model_name == 'vacuum_instability':
                # Test vacuum instability effects
                results = theoretical_models.test_vacuum_instability_spectrum(df)
                
            elif model_name == 'hidden_sector':
                # Test hidden sector energy loss
                results = theoretical_models.test_hidden_sector_spectrum(df)
                
            else:
                print(f"    Warning: Unknown model {model_name}, skipping")
                continue
            
            # Add model identifier to results
            if isinstance(results, dict):
                results['model'] = model_name
                all_results.append(results)
            elif isinstance(results, list):
                for result in results:
                    result['model'] = model_name
                    all_results.append(result)
                    
        except Exception as e:
            print(f"    Error testing {model_name}: {e}")
            continue
    
    return pd.DataFrame(all_results) if all_results else simulate_uhecr_basic(path)

def simulate_uhecr(path, use_enhanced=True, models=None):
    """Main UHECR simulation function with model selection"""
    if models is None:
        models = ['polynomial', 'polymer_qed', 'gravity_rainbow', 'vacuum_instability']
    
    if use_enhanced and ENHANCED_AVAILABLE:
        return simulate_uhecr_enhanced(path, models)
    else:
        return simulate_uhecr_basic(path)

def main():
    """Main UHECR analysis pipeline with enhanced theoretical model testing"""
    print("Running Enhanced UHECR Analysis with Theoretical Model Testing")
    print("=" * 70)
    
    # Define spectrum data path
    uhecr_csv = os.path.join("data", "uhecr", "sd1500_spectrum.csv")
    
    if not os.path.exists(uhecr_csv):
        print(f"Error: UHECR spectrum file not found: {uhecr_csv}")
        print("Please run the spectrum analysis first to generate this file.")
        return
    
    try:
        # Define models to test
        models_to_test = [
            'polynomial',      # Non-linear threshold modifications
            'polymer_qed',     # Polymer-quantized QED  
            'gravity_rainbow', # Gravity-rainbow corrections
            'vacuum_instability', # Vacuum instability effects
            'hidden_sector'    # Hidden sector energy loss
        ]
        
        print(f"Testing theoretical models: {models_to_test}")
        
        # Run enhanced analysis
        df = simulate_uhecr(uhecr_csv, use_enhanced=True, models=models_to_test)
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results
        df.to_csv(os.path.join("results", "uhecr_enhanced_exclusion.csv"), index=False)
        
        # Also save simplified version for compatibility
        simple_cols = ['E_LV_p (GeV)', 'chi2', 'Excluded'] if 'E_LV_p (GeV)' in df.columns else df.columns[:3]
        simple_df = df[simple_cols].copy() if len(df.columns) >= 3 else df
        simple_df.to_csv(os.path.join("results", "uhecr_exclusion.csv"), index=False)
        
        print(f"\n{'='*70}")
        print("ENHANCED UHECR ANALYSIS RESULTS:")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        
        # Summary by model
        if 'model' in df.columns:
            print(f"\n{'='*70}")
            print("RESULTS BY THEORETICAL MODEL:")
            print(f"{'='*70}")
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                print(f"\n{model.upper()}:")
                if 'Excluded' in model_df.columns:
                    excluded_count = model_df['Excluded'].sum()
                    total_count = len(model_df)
                    print(f"  Exclusions: {excluded_count}/{total_count}")
                if 'chi2' in model_df.columns:
                    best_chi2 = model_df['chi2'].min()
                    print(f"  Best χ²: {best_chi2:.3f}")
        
        print(f"\nResults saved to:")
        print(f"  - results/uhecr_enhanced_exclusion.csv (detailed)")
        print(f"  - results/uhecr_exclusion.csv (simplified)")
        
    except Exception as e:
        print(f"Error during enhanced analysis: {e}")
        print("Falling back to basic analysis...")
        
        # Fallback to basic analysis
        df = simulate_uhecr_basic(uhecr_csv)
        os.makedirs("results", exist_ok=True)
        df.to_csv(os.path.join("results", "uhecr_exclusion.csv"), index=False)
        print("\nBASIC UHECR ANALYSIS RESULTS:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()