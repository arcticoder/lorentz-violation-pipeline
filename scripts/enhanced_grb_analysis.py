#!/usr/bin/env python3
"""
Enhanced GRB Analysis with Polynomial Dispersion Fitting

This module extends the basic GRB time-delay analysis to test sophisticated
theoretical models including polynomial expansions in E/E_Pl:

Δt = D(z)[α₁(E/E_Pl) + α₂(E²/E_Pl²) + α₃(E³/E_Pl³) + ...]

This directly tests:
- Polymer-quantized QED predictions  
- Gravity-rainbow model signatures
- Higher-order Planck-scale corrections
- Non-linear LIV effects

Key improvements over basic linear fitting:
1. Multi-parameter polynomial fits
2. Model selection and comparison
3. Theoretical model constraints
4. Systematic uncertainty analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import glob
import os

# Physical constants
E_PLANCK = 1.22e19  # GeV
LIGHT_SPEED = 2.998e8  # m/s

class PolynomialDispersionFitter:
    """
    Fit polynomial expansions to GRB time-delay data.
    
    Models tested:
    1. Linear: Δt = α₁(E/E_Pl)D(z)
    2. Quadratic: Δt = [α₁(E/E_Pl) + α₂(E/E_Pl)²]D(z) 
    3. Cubic: Δt = [α₁(E/E_Pl) + α₂(E/E_Pl)² + α₃(E/E_Pl)³]D(z)
    4. Quartic: Full 4th-order expansion
    """
    
    def __init__(self, distance_factor=1e17):
        """
        Initialize the fitter.
        
        Parameters:
        -----------
        distance_factor : float
            Characteristic distance scale D(z) in seconds
            For cosmological GRBs: D ~ ct_H ~ 10^17 s
        """
        self.D = distance_factor
        self.fit_results = {}
        self.model_comparison = {}
    
    def linear_model(self, E, alpha1):
        """Linear LIV model: Δt = α₁(E/E_Pl)D."""
        return alpha1 * (E / E_PLANCK) * self.D
    
    def quadratic_model(self, E, alpha1, alpha2):
        """Quadratic LIV model: Δt = [α₁(E/E_Pl) + α₂(E/E_Pl)²]D."""
        x = E / E_PLANCK
        return (alpha1 * x + alpha2 * x**2) * self.D
    
    def cubic_model(self, E, alpha1, alpha2, alpha3):
        """Cubic LIV model: Δt = [α₁(E/E_Pl) + α₂(E/E_Pl)² + α₃(E/E_Pl)³]D."""
        x = E / E_PLANCK
        return (alpha1 * x + alpha2 * x**2 + alpha3 * x**3) * self.D
    
    def quartic_model(self, E, alpha1, alpha2, alpha3, alpha4):
        """Quartic LIV model: 4th-order polynomial expansion."""
        x = E / E_PLANCK
        return (alpha1 * x + alpha2 * x**2 + alpha3 * x**3 + alpha4 * x**4) * self.D
    
    def polymer_qed_model(self, E, alpha1, alpha2):
        """
        Polymer-QED theoretical prediction.
        
        For polymer field theory:
        Δt ≈ (c/2)[α₁(E/E_Pl) + α₂(E/E_Pl)²]D(z)
        
        where α₁ ~ γ (polymer parameter)
              α₂ ~ γ²/2 (second-order correction)
        """
        return self.quadratic_model(E, alpha1, alpha2)
    
    def rainbow_gravity_model(self, E, eta, n):
        """
        Gravity-rainbow theoretical prediction.
        
        For rainbow dispersion ω² = k²(1-ηk/E_Pl)^n:
        Δt ≈ (D(z)/2c) * η * n * (E/E_Pl)
        
        Leading order gives linear dependence.
        """
        return eta * n * (E / E_PLANCK) * self.D / 2
    
    def fit_grb_data(self, energy_data, time_data, models='all'):
        """
        Fit multiple polynomial models to GRB time-delay data.
        
        Parameters:
        -----------
        energy_data : array
            Photon energies in GeV
        time_data : array  
            Time delays in seconds
        models : str or list
            Which models to fit ('all', 'basic', or list of model names)
        
        Returns:
        --------
        results : dict
            Fit results for each model including parameters, errors, chi²
        """
        
        if models == 'all':
            models_to_fit = ['linear', 'quadratic', 'cubic', 'quartic', 'polymer_qed', 'rainbow']
        elif models == 'basic':
            models_to_fit = ['linear', 'quadratic', 'cubic']
        else:
            models_to_fit = models
        
        results = {}
        
        for model_name in models_to_fit:
            try:
                result = self._fit_single_model(energy_data, time_data, model_name)
                results[model_name] = result
                print(f"  {model_name}: χ² = {result['chi2']:.2f}, AIC = {result['aic']:.2f}")
            except Exception as e:
                print(f"  {model_name}: Fit failed - {e}")
                results[model_name] = None
        
        return results
    
    def _fit_single_model(self, E, t, model_name):
        """Fit a single model to the data."""
        
        # Select the appropriate model function
        if model_name == 'linear':
            model_func = self.linear_model
            p0 = [1.0]
        elif model_name == 'quadratic':
            model_func = self.quadratic_model  
            p0 = [1.0, 0.1]
        elif model_name == 'cubic':
            model_func = self.cubic_model
            p0 = [1.0, 0.1, 0.01]
        elif model_name == 'quartic':
            model_func = self.quartic_model
            p0 = [1.0, 0.1, 0.01, 0.001]
        elif model_name == 'polymer_qed':
            model_func = self.polymer_qed_model
            p0 = [1.0, 0.5]  # Theory suggests α₂ ~ α₁²/2
        elif model_name == 'rainbow':
            model_func = self.rainbow_gravity_model
            p0 = [1.0, 1.0]  # eta, n parameters
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Perform the fit
        popt, pcov = curve_fit(model_func, E, t, p0=p0, maxfev=5000)
        
        # Calculate fit statistics
        t_pred = model_func(E, *popt)
        residuals = t - t_pred
        chi2 = np.sum(residuals**2) / (len(t) - len(popt))
        
        # Akaike Information Criterion for model comparison
        n = len(t)
        k = len(popt)
        aic = 2*k + n*np.log(np.sum(residuals**2)/n)
        
        # Parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.inf)
        
        # Calculate theoretical energy scales
        energy_scales = self._extract_energy_scales(popt, param_errors, model_name)
        
        return {
            'model': model_name,
            'parameters': popt,
            'param_errors': param_errors,
            'chi2': chi2,
            'aic': aic,
            'residuals': residuals,
            'prediction': t_pred,
            'energy_scales': energy_scales
        }
    
    def _extract_energy_scales(self, params, errors, model_name):
        """Extract characteristic energy scales from fit parameters."""
        
        scales = {}
        
        if model_name in ['linear', 'quadratic', 'cubic', 'quartic']:
            # For polynomial models: E_LV ~ E_Pl / |α₁|
            if len(params) >= 1 and params[0] != 0:
                scales['E_LV_linear'] = E_PLANCK / abs(params[0])
                if len(errors) >= 1:
                    scales['E_LV_linear_error'] = E_PLANCK * errors[0] / params[0]**2
            
            # Quadratic scale: E_LV2 ~ √(E_Pl / |α₂|)  
            if len(params) >= 2 and params[1] != 0:
                scales['E_LV_quadratic'] = np.sqrt(E_PLANCK / abs(params[1]))
                if len(errors) >= 2:
                    scales['E_LV_quadratic_error'] = 0.5 * scales['E_LV_quadratic'] * errors[1] / abs(params[1])
        
        elif model_name == 'polymer_qed':
            # Polymer parameter γ ~ α₁
            scales['gamma_polymer'] = params[0]
            if len(errors) >= 1:
                scales['gamma_polymer_error'] = errors[0]
        
        elif model_name == 'rainbow':
            # Rainbow parameters η and n
            scales['eta_rainbow'] = params[0]
            scales['n_rainbow'] = params[1]
            if len(errors) >= 2:
                scales['eta_rainbow_error'] = errors[0]
                scales['n_rainbow_error'] = errors[1]
        
        return scales
    
    def model_comparison_analysis(self, results):
        """
        Perform statistical model comparison using AIC and F-tests.
        
        Determines which polynomial order is statistically preferred.
        """
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) < 2:
            return "Insufficient models for comparison"
        
        # Sort by AIC (lower is better)
        sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['aic'])
        
        comparison = {
            'best_model': sorted_models[0][0],
            'aic_ranking': [(name, result['aic']) for name, result in sorted_models],
            'evidence_ratios': {}
        }
        
        # Calculate AIC evidence ratios
        best_aic = sorted_models[0][1]['aic']
        for name, result in sorted_models[1:]:
            delta_aic = result['aic'] - best_aic
            evidence_ratio = np.exp(delta_aic / 2)
            comparison['evidence_ratios'][name] = evidence_ratio
        
        return comparison
    
    def plot_fit_results(self, energy_data, time_data, results, save_path=None):
        """Create comprehensive plots of polynomial fit results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Data and all model fits
        ax1 = axes[0, 0]
        ax1.scatter(energy_data, time_data, alpha=0.7, s=50, c='black', label='GRB data')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        energy_smooth = np.logspace(np.log10(energy_data.min()), 
                                   np.log10(energy_data.max()), 200)
        
        for i, (model_name, result) in enumerate(results.items()):
            if result is not None:
                if model_name == 'linear':
                    t_smooth = self.linear_model(energy_smooth, *result['parameters'])
                elif model_name == 'quadratic':
                    t_smooth = self.quadratic_model(energy_smooth, *result['parameters'])
                elif model_name == 'cubic':
                    t_smooth = self.cubic_model(energy_smooth, *result['parameters'])
                # Add other models as needed
                
                ax1.plot(energy_smooth, t_smooth, '--', color=colors[i % len(colors)],
                        label=f"{model_name} (χ²={result['chi2']:.2f})")
        
        ax1.set_xlabel('Energy (GeV)')
        ax1.set_ylabel('Time Delay (s)')
        ax1.set_xscale('log')
        ax1.set_title('GRB Time Delays: Polynomial Model Fits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residuals comparison
        ax2 = axes[0, 1]
        for i, (model_name, result) in enumerate(results.items()):
            if result is not None:
                ax2.scatter(energy_data, result['residuals'], alpha=0.7, 
                           color=colors[i % len(colors)], label=model_name)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Energy (GeV)')
        ax2.set_ylabel('Residuals (s)')
        ax2.set_xscale('log')
        ax2.set_title('Fit Residuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model comparison (AIC)
        ax3 = axes[1, 0]
        valid_results = {k: v for k, v in results.items() if v is not None}
        model_names = list(valid_results.keys())
        aic_values = [result['aic'] for result in valid_results.values()]
        
        bars = ax3.bar(model_names, aic_values, alpha=0.7)
        ax3.set_ylabel('AIC')
        ax3.set_title('Model Comparison (lower AIC = better)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Highlight best model
        min_aic_idx = np.argmin(aic_values)
        bars[min_aic_idx].set_color('red')
        
        # Plot 4: Energy scale constraints
        ax4 = axes[1, 1]
        energy_scales = []
        scale_errors = []
        model_labels = []
        
        for model_name, result in valid_results.items():
            if 'E_LV_linear' in result['energy_scales']:
                energy_scales.append(result['energy_scales']['E_LV_linear'])
                scale_errors.append(result['energy_scales'].get('E_LV_linear_error', 0))
                model_labels.append(f"{model_name}\n(linear)")
        
        if energy_scales:
            ax4.errorbar(range(len(energy_scales)), energy_scales, yerr=scale_errors, 
                        fmt='o', capsize=5, markersize=8)
            ax4.set_yscale('log')
            ax4.set_ylabel('LIV Energy Scale (GeV)')
            ax4.set_title('Energy Scale Constraints')
            ax4.set_xticks(range(len(model_labels)))
            ax4.set_xticklabels(model_labels, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Polynomial fit analysis saved to {save_path}")
        
        plt.show()

def analyze_grb_with_polynomials(grb_file, distance_factor=1e17):
    """
    Analyze a single GRB file with polynomial dispersion models.
    
    This is the enhanced version of the basic GRB analysis that tests
    sophisticated theoretical predictions.
    """
    
    print(f"\nPolynomial Dispersion Analysis: {grb_file}")
    print("=" * 60)
    
    # Load GRB data
    df = pd.read_csv(grb_file)
    delta_E = df['delta_E_GeV'].values
    delta_t = df['delta_t_s'].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(delta_E) | np.isnan(delta_t))
    delta_E = delta_E[valid_mask]
    delta_t = delta_t[valid_mask]
    
    print(f"Data points: {len(delta_E)}")
    print(f"Energy range: {delta_E.min():.2f} - {delta_E.max():.2f} GeV")
    print(f"Time delay range: {delta_t.min():.4f} - {delta_t.max():.4f} s")
    
    # Initialize polynomial fitter
    fitter = PolynomialDispersionFitter(distance_factor=distance_factor)
    
    # Fit all polynomial models
    print(f"\nFitting polynomial dispersion models:")
    results = fitter.fit_grb_data(delta_E, delta_t, models='all')
    
    # Perform model comparison
    print(f"\nModel comparison analysis:")
    comparison = fitter.model_comparison_analysis(results)
    
    if isinstance(comparison, dict):
        print(f"Best model: {comparison['best_model']}")
        print(f"AIC ranking:")
        for model, aic in comparison['aic_ranking'][:3]:
            print(f"  {model}: AIC = {aic:.2f}")
    
    # Create plots
    grb_name = os.path.basename(grb_file).replace('.csv', '')
    plot_file = f"results/{grb_name}_polynomial_analysis.png"
    fitter.plot_fit_results(delta_E, delta_t, results, save_path=plot_file)
    
    return results, comparison

def main():
    """Main function for enhanced GRB analysis."""
    
    print("Enhanced GRB Analysis with Polynomial Dispersion Fitting")
    print("=" * 60)
    print("Testing theoretical models:")
    print("  - Polymer-quantized QED")
    print("  - Gravity-rainbow dispersion")  
    print("  - Higher-order Planck corrections")
    print()
    
    # Analyze all GRB files
    grb_files = glob.glob(os.path.join("data", "grbs", "*.csv"))
    
    if not grb_files:
        print("No GRB data files found!")
        return
    
    all_results = {}
    
    for grb_file in grb_files:
        try:
            results, comparison = analyze_grb_with_polynomials(grb_file)
            grb_name = os.path.basename(grb_file).replace('.csv', '')
            all_results[grb_name] = {'fits': results, 'comparison': comparison}
        except Exception as e:
            print(f"Error analyzing {grb_file}: {e}")
    
    # Save combined results
    print(f"\nSaving enhanced analysis results...")
    os.makedirs("results", exist_ok=True)
    
    # Create summary of best-fit models and energy scales
    summary_data = []
    for grb_name, data in all_results.items():
        if isinstance(data['comparison'], dict):
            best_model = data['comparison']['best_model']
            best_result = data['fits'][best_model]
            
            summary_data.append({
                'GRB': grb_name,
                'Best_Model': best_model,
                'Chi2': best_result['chi2'],
                'AIC': best_result['aic'],
                'Parameters': str(best_result['parameters']),
                'Energy_Scales': str(best_result['energy_scales'])
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("results/grb_polynomial_analysis.csv", index=False)
        print(f"Summary saved to results/grb_polynomial_analysis.csv")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
