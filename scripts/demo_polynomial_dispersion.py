#!/usr/bin/env python3
"""
Demonstration: Polynomial Dispersion Relations in LIV Analysis

This script demonstrates the key enhancement from linear to polynomial 
dispersion relations, directly testing theoretical predictions like:

Δt = D(z)[α₁(E/E_Pl) + α₂(E²/E_Pl²) + α₃(E³/E_Pl³) + ...]

Key Features:
1. Polynomial vs linear model comparison
2. Theoretical model predictions (polymer-QED, gravity-rainbow)
3. Model selection using AIC
4. Physical constraint validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Physical constants
E_PLANCK = 1.22e19  # GeV

def simulate_theoretical_data(model='polymer_qed', alpha1=1.0, alpha2=0.1, noise_level=0.05):
    """Generate synthetic GRB data with theoretical dispersion relations."""
    energies = np.logspace(np.log10(0.1), np.log10(100), 20)  # 0.1 to 100 GeV
    distance_factor = 1e17  # seconds
    
    if model == 'linear':
        # Simple linear dispersion: Δt = α₁(E/E_Pl)D(z)
        time_delays = distance_factor * alpha1 * (energies / E_PLANCK)
        
    elif model == 'polymer_qed':
        # Polymer-QED: Δt = D(z)[α₁(E/E_Pl) + α₂(E²/E_Pl²)]
        x = energies / E_PLANCK
        time_delays = distance_factor * (alpha1 * x + alpha2 * x**2)
        
    elif model == 'gravity_rainbow':
        # Gravity-rainbow with f(x) = (1 + ηx)^(-1)
        eta = alpha1  # Use alpha1 as eta parameter
        x = energies / E_PLANCK  
        rainbow_factor = (1 + eta * x)**(-1)
        time_delays = distance_factor * alpha1 * x * rainbow_factor
        
    # Add realistic noise
    noise = np.random.normal(0, noise_level * np.mean(time_delays), len(energies))
    time_delays += noise
    
    return energies, time_delays

def fit_polynomial(energies, times, order=2):
    """Fit polynomial dispersion relation."""
    x = energies / E_PLANCK
    
    # Fit polynomial: Δt = D(z)[α₁x + α₂x² + α₃x³ + ...]
    coeffs = np.polyfit(x, times, order)
    fitted_times = np.polyval(coeffs, x)
    
    # Calculate goodness of fit
    chi2 = np.sum((times - fitted_times)**2) / len(times)
    
    # Calculate AIC (approximate)
    n = len(times)
    k = order + 1  # number of parameters
    aic = 2*k + n*np.log(chi2)
    
    return coeffs, fitted_times, chi2, aic

def demonstrate_polynomial_fitting():
    """Demonstrate polynomial dispersion fitting capabilities."""
    print("POLYNOMIAL DISPERSION RELATION DEMONSTRATION")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Test different theoretical models
    models = {
        'linear': {'alpha1': 1.0, 'alpha2': 0.0},
        'polymer_qed': {'alpha1': 1.0, 'alpha2': 0.2}, 
        'gravity_rainbow': {'alpha1': 0.8, 'alpha2': 0.0}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    results = []
    
    for i, (model_name, params) in enumerate(models.items()):
        print(f"\n{model_name.upper()} MODEL:")
        print("-" * 30)
        
        # Generate theoretical data
        energies, times = simulate_theoretical_data(
            model=model_name, 
            alpha1=params['alpha1'], 
            alpha2=params['alpha2']
        )
        
        # Test polynomial fits of different orders
        orders = [1, 2, 3]
        best_aic = float('inf')
        best_order = 1
        
        for order in orders:
            coeffs, fitted_times, chi2, aic = fit_polynomial(energies, times, order)
            
            print(f"  Order {order}: χ² = {chi2:.6f}, AIC = {aic:.2f}")
            print(f"    Coefficients: {coeffs}")
            
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_coeffs = coeffs
                best_fitted = fitted_times
        
        print(f"  → Best model: Order {best_order} (AIC = {best_aic:.2f})")
        
        # Calculate LIV energy scale from leading coefficient
        E_LV = 1e17 / abs(best_coeffs[-1])  # From leading coefficient
        print(f"  → E_LV estimate: {E_LV:.2e} GeV")
        
        # Store results
        results.append({
            'Model': model_name,
            'Best_Order': best_order,
            'AIC': best_aic,
            'Chi2': chi2,
            'E_LV_GeV': E_LV,
            'Coefficients': str(best_coeffs.tolist())
        })
        
        # Plot results
        ax = axes[i]
        ax.scatter(energies, times, alpha=0.7, label='Data', color='blue')
        ax.plot(energies, best_fitted, 'r-', linewidth=2, 
                label=f'Order {best_order} fit')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('Time Delay (s)')
        ax.set_title(f'{model_name.title()} Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplot
    fig.delaxes(axes[3])
    
    plt.tight_layout()
    plt.savefig('results/polynomial_dispersion_demo.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: results/polynomial_dispersion_demo.png")
    
    # Save results summary
    df = pd.DataFrame(results)
    df.to_csv('results/polynomial_demo_results.csv', index=False)
    print(f"Results saved: results/polynomial_demo_results.csv")
    
    print(f"\n{'=' * 60}")
    print("SUMMARY: POLYNOMIAL vs LINEAR DISPERSION")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))
    
    return df

def compare_theoretical_models():
    """Compare different theoretical LIV models."""
    print(f"\n{'=' * 60}")
    print("THEORETICAL MODEL COMPARISON")  
    print(f"{'=' * 60}")
    
    # Test energy range (GeV)
    energies = np.logspace(15, 20, 6)  # 10^15 to 10^20 GeV
    
    print("Energy-dependent group velocities (v/c):")
    print("-" * 40)
    print("Energy (GeV)     Standard    Polymer-QED   Rainbow")
    print("-" * 40)
    
    for E in energies:
        # Standard relativity
        v_standard = 1.0
        
        # Polymer-QED: v ≈ 1 + α(E/E_Pl)  
        alpha = 0.5
        v_polymer = 1.0 + alpha * (E / E_PLANCK)
        
        # Gravity-rainbow: v ≈ 1 + η(E/E_Pl)/(1 + η(E/E_Pl))
        eta = 1.0
        x = E / E_PLANCK
        v_rainbow = 1.0 + eta * x / (1 + eta * x)
        
        print(f"{E:.1e}    {v_standard:.6f}    {v_polymer:.6f}    {v_rainbow:.6f}")
        
        # Check for superluminal propagation
        if v_polymer > 1.1 or v_rainbow > 1.1:
            print("                                      ⚠️  Superluminal!")
    
    print("-" * 40)
    
    return energies

def main():
    """Run the complete polynomial dispersion demonstration."""
    print("ENHANCED LIV FRAMEWORK: POLYNOMIAL DISPERSION DEMONSTRATION")
    print("=" * 70)
    print("This demonstrates the key advancement from linear to polynomial")
    print("dispersion relations for testing theoretical LIV models.")
    print("=" * 70)
    
    # Demonstrate polynomial fitting
    results_df = demonstrate_polynomial_fitting()
    
    # Compare theoretical models
    energies = compare_theoretical_models()
    
    print(f"\n{'=' * 70}")
    print("KEY ACHIEVEMENTS:")
    print("✅ Polynomial dispersion fitting (replaces linear-only)")
    print("✅ Theoretical model testing (polymer-QED, gravity-rainbow)")  
    print("✅ Model selection using AIC")
    print("✅ LIV energy scale extraction from polynomial coefficients")
    print("✅ Physical constraint validation")
    print("=" * 70)
    
    print("\nNext steps:")
    print("• Run the full enhanced pipeline: python scripts/run_full_analysis.py")
    print("• Test specific models: python scripts/enhanced_grb_analysis.py")
    print("• Validate theory: python scripts/theoretical_liv_models.py")

if __name__ == "__main__":
    main()
