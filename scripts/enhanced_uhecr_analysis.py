#!/usr/bin/env python3
"""
Enhanced UHECR Analysis with Theoretical LIV Models

This module implements sophisticated theoretical predictions for UHECR interactions
in the presence of Lorentz Invariance Violation, going beyond simple threshold shifts.

Key theoretical frameworks:
1. Polymer-QED modified photopion thresholds
2. Gravity-rainbow altered kinematics  
3. Vacuum instability effects on propagation
4. Hidden sector energy loss mechanisms
5. Non-linear threshold modifications

Enhanced threshold calculations:
- Standard: E_th = m_π(m_π + 2m_p)/(4E_CMB)
- LIV: E_th → E_th * f(E/E_LV, model_parameters)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os

# Physical constants
PROTON_MASS = 0.938  # GeV
PION_MASS = 0.140   # GeV  
CMB_ENERGY = 2.7e-4  # GeV (CMB photon energy)
E_PLANCK = 1.22e19   # GeV

class EnhancedUHECRAnalysis:
    """
    Enhanced UHECR analysis with theoretical LIV predictions.
    
    Implements various LIV models for cosmic ray interactions:
    - Modified photopion production thresholds
    - Altered GZK cutoff predictions
    - Vacuum instability effects
    - Hidden sector energy losses
    """
    
    def __init__(self):
        self.models = {}
        self.spectrum_data = None
        self.theoretical_predictions = {}
    
    def load_spectrum(self, spectrum_file):
        """Load UHECR spectrum data."""
        self.spectrum_data = pd.read_csv(spectrum_file)
        print(f"Loaded spectrum with {len(self.spectrum_data)} energy bins")
    
    def standard_photopion_threshold(self):
        """
        Calculate standard photopion production threshold.
        
        p + γ_CMB → p + π⁰ (threshold reaction)
        E_th = m_π(m_π + 2m_p)/(4E_CMB) ≈ 6×10¹⁹ eV
        """
        return PION_MASS * (PION_MASS + 2*PROTON_MASS) / (4*CMB_ENERGY)
    
    def polymer_qed_threshold(self, alpha1=0, alpha2=0):
        """
        Modified photopion threshold in polymer-QED.
        
        The modified dispersion relation ω² = k²[1 + α(k/E_Pl)] + m²
        alters the kinematic threshold conditions.
        
        Leading correction: E_th → E_th[1 + α₁E_th/(2E_Pl)]
        """
        E_th_standard = self.standard_photopion_threshold()
        
        # First-order polymer correction
        polymer_correction = 1 + alpha1 * E_th_standard / (2 * E_PLANCK)
        
        # Second-order correction
        if alpha2 != 0:
            polymer_correction += alpha2 * (E_th_standard / E_PLANCK)**2
        
        return E_th_standard * polymer_correction
    
    def rainbow_gravity_threshold(self, eta=1.0, n=1):
        """
        Modified threshold in gravity-rainbow models.
        
        With ω² = k²(1 - ηk/E_Pl)^n + m², the threshold energy becomes:
        E_th → E_th / [1 - η*E_th/E_Pl]^(n/2)
        """
        E_th_standard = self.standard_photopion_threshold()
        
        # Rainbow correction factor
        x = eta * E_th_standard / E_PLANCK
        if x < 1:
            rainbow_factor = (1 - x)**(n/2)
            return E_th_standard / rainbow_factor
        else:
            # Threshold becomes inaccessible
            return float('inf')
    
    def vacuum_instability_threshold(self, field_strength_param=1e-6):
        """
        Vacuum instability effects on UHECR propagation.
        
        In some LIV models, vacuum becomes unstable at high energies,
        leading to modified propagation and energy losses.
        """
        E_th_standard = self.standard_photopion_threshold()
        
        # Schwinger-like instability threshold
        E_instability = np.sqrt(PROTON_MASS / field_strength_param) * E_PLANCK
        
        if E_th_standard > E_instability:
            # Energy loss due to vacuum pair production
            loss_factor = np.exp(-E_th_standard / E_instability)
            return E_th_standard * loss_factor
        else:
            return E_th_standard
    
    def hidden_sector_threshold(self, coupling=1e-6, hidden_mass=1e-3):
        """
        Hidden sector interactions modifying UHECR thresholds.
        
        High-energy cosmic rays can interact with hidden sector fields,
        modifying effective interaction thresholds.
        """
        E_th_standard = self.standard_photopion_threshold()
        
        # Energy loss to hidden sector
        if E_th_standard > hidden_mass:
            # Dimensional analysis for energy transfer rate
            transfer_rate = coupling**2 * E_th_standard / hidden_mass
            effective_threshold = E_th_standard * (1 + transfer_rate)
            return effective_threshold
        else:
            return E_th_standard
    
    def calculate_theoretical_spectra(self):
        """
        Calculate theoretical UHECR spectra for various LIV models.
        
        Each model predicts a different GZK cutoff position and shape.
        """
        
        if self.spectrum_data is None:
            raise ValueError("No spectrum data loaded")
        
        energies = self.spectrum_data['E_center_eV'].values
        
        models = {
            'standard': {'threshold': self.standard_photopion_threshold()},
            
            'polymer_linear': {
                'threshold': self.polymer_qed_threshold(alpha1=1.0),
                'params': {'alpha1': 1.0}
            },
            
            'polymer_quadratic': {
                'threshold': self.polymer_qed_threshold(alpha1=0.5, alpha2=0.1),
                'params': {'alpha1': 0.5, 'alpha2': 0.1}
            },
            
            'rainbow_linear': {
                'threshold': self.rainbow_gravity_threshold(eta=1.0, n=1),
                'params': {'eta': 1.0, 'n': 1}
            },
            
            'rainbow_quadratic': {
                'threshold': self.rainbow_gravity_threshold(eta=1.0, n=2),
                'params': {'eta': 1.0, 'n': 2}
            },
            
            'vacuum_instability': {
                'threshold': self.vacuum_instability_threshold(),
                'params': {'field_strength': 1e-6}
            },
            
            'hidden_sector': {
                'threshold': self.hidden_sector_threshold(),
                'params': {'coupling': 1e-6, 'hidden_mass': 1e-3}
            }
        }
        
        predictions = {}
        
        for model_name, model_data in models.items():
            threshold = model_data['threshold']
            
            # Simple GZK suppression model
            # Flux suppression factor: exp(-(E/E_th)^β) for E > E_th
            suppression = np.ones_like(energies)
            high_energy = energies > threshold
            
            if np.any(high_energy):
                # Exponential suppression above threshold
                beta = 2.0  # Steepness parameter
                suppression[high_energy] = np.exp(-((energies[high_energy]/threshold)**beta))
            
            predictions[model_name] = {
                'threshold_energy': threshold,
                'suppression_factor': suppression,
                'parameters': model_data.get('params', {}),
                'predicted_flux': self.spectrum_data['flux'].values * suppression
            }
        
        self.theoretical_predictions = predictions
        return predictions
    
    def chi_squared_analysis(self, models_to_test=None):
        """
        Perform chi-squared analysis comparing theoretical predictions to data.
        
        Tests which LIV models are consistent with observed UHECR spectrum.
        """
        
        if not self.theoretical_predictions:
            self.calculate_theoretical_spectra()
        
        if models_to_test is None:
            models_to_test = list(self.theoretical_predictions.keys())
        
        results = {}
        observed_flux = self.spectrum_data['flux'].values
        flux_errors = self.spectrum_data['flux_err_low'].values
        
        # Handle zero errors
        flux_errors = np.where(flux_errors > 0, flux_errors, 
                              np.mean(flux_errors[flux_errors > 0]))
        
        for model_name in models_to_test:
            if model_name in self.theoretical_predictions:
                predicted_flux = self.theoretical_predictions[model_name]['predicted_flux']
                
                # Chi-squared calculation
                valid_data = (observed_flux > 0) & (flux_errors > 0)
                if np.any(valid_data):
                    chi2 = np.sum(((observed_flux[valid_data] - predicted_flux[valid_data]) / 
                                  flux_errors[valid_data])**2)
                    dof = np.sum(valid_data) - 1  # Degrees of freedom
                    
                    results[model_name] = {
                        'chi2': chi2,
                        'dof': dof,
                        'chi2_reduced': chi2 / dof if dof > 0 else float('inf'),
                        'threshold': self.theoretical_predictions[model_name]['threshold_energy'],
                        'parameters': self.theoretical_predictions[model_name]['parameters']
                    }
                    
                    print(f"{model_name}: χ² = {chi2:.2f}, χ²/dof = {chi2/dof:.2f}")
        
        return results
    
    def exclusion_analysis(self, confidence_level=0.95):
        """
        Determine which LIV models are excluded by the data.
        
        Uses chi-squared test to determine if theoretical predictions
        are consistent with observations.
        """
        
        chi2_results = self.chi_squared_analysis()
        
        # Critical chi-squared value for given confidence level
        from scipy.stats import chi2 as chi2_dist
        exclusions = {}
        
        for model_name, result in chi2_results.items():
            dof = result['dof']
            chi2_critical = chi2_dist.ppf(confidence_level, dof)
            
            excluded = result['chi2'] > chi2_critical
            p_value = 1 - chi2_dist.cdf(result['chi2'], dof)
            
            exclusions[model_name] = {
                'excluded': excluded,
                'p_value': p_value,
                'chi2': result['chi2'],
                'chi2_critical': chi2_critical,
                'confidence_level': confidence_level
            }
        
        return exclusions
    
    def plot_theoretical_comparison(self, save_path=None):
        """Create comprehensive plots comparing theoretical predictions."""
        
        if not self.theoretical_predictions:
            self.calculate_theoretical_spectra()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        energies = self.spectrum_data['E_center_eV'].values / 1e18  # Convert to EeV
        observed_flux = self.spectrum_data['flux'].values
        flux_errors = self.spectrum_data['flux_err_low'].values
        
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Plot 1: Flux comparison
        ax1 = axes[0, 0]
        ax1.errorbar(energies, observed_flux * (energies*1e18)**3, 
                    yerr=flux_errors * (energies*1e18)**3,
                    fmt='o', color='black', label='Auger data', markersize=4)
        
        for i, (model_name, prediction) in enumerate(self.theoretical_predictions.items()):
            predicted_flux = prediction['predicted_flux']
            ax1.plot(energies, predicted_flux * (energies*1e18)**3, '--', 
                    color=colors[i % len(colors)], label=model_name, linewidth=2)
        
        ax1.set_xlabel('Energy (EeV)')
        ax1.set_ylabel('E³ × J(E) [eV² km⁻² sr⁻¹ yr⁻¹]')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('UHECR Spectrum: LIV Model Predictions')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Suppression factors
        ax2 = axes[0, 1]
        for i, (model_name, prediction) in enumerate(self.theoretical_predictions.items()):
            if model_name != 'standard':
                ax2.plot(energies, prediction['suppression_factor'], 
                        color=colors[i % len(colors)], label=model_name, linewidth=2)
        
        ax2.set_xlabel('Energy (EeV)')
        ax2.set_ylabel('Suppression Factor')
        ax2.set_xscale('log')
        ax2.set_title('GZK Suppression: Model Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Threshold energies
        ax3 = axes[1, 0]
        model_names = []
        thresholds = []
        
        for model_name, prediction in self.theoretical_predictions.items():
            model_names.append(model_name)
            thresholds.append(prediction['threshold_energy'] / 1e18)  # EeV
        
        bars = ax3.bar(model_names, thresholds, alpha=0.7)
        ax3.set_ylabel('Threshold Energy (EeV)')
        ax3.set_title('Photopion Thresholds: LIV Modifications')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        
        # Highlight standard threshold
        standard_idx = model_names.index('standard')
        bars[standard_idx].set_color('red')
        
        # Plot 4: Chi-squared comparison
        ax4 = axes[1, 1]
        chi2_results = self.chi_squared_analysis()
        
        if chi2_results:
            models = list(chi2_results.keys())
            chi2_values = [result['chi2_reduced'] for result in chi2_results.values()]
            
            bars = ax4.bar(models, chi2_values, alpha=0.7)
            ax4.axhline(y=1, color='red', linestyle='--', label='χ²/dof = 1')
            ax4.set_ylabel('χ² / dof')
            ax4.set_title('Model Goodness of Fit')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
            
            # Color bars by fit quality
            for i, bar in enumerate(bars):
                if chi2_values[i] < 1.5:
                    bar.set_color('green')  # Good fit
                elif chi2_values[i] < 3.0:
                    bar.set_color('orange')  # Acceptable
                else:
                    bar.set_color('red')  # Poor fit
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Theoretical comparison saved to {save_path}")
        
        plt.show()

def main():
    """Main function for enhanced UHECR analysis."""
    
    print("Enhanced UHECR Analysis with Theoretical LIV Models")
    print("=" * 60)
    print("Testing sophisticated theoretical frameworks:")
    print("  - Polymer-quantized QED")
    print("  - Gravity-rainbow dispersion")
    print("  - Vacuum instability effects")
    print("  - Hidden sector interactions")
    print()
    
    # Initialize analyzer
    analyzer = EnhancedUHECRAnalysis()
    
    # Load spectrum data
    spectrum_file = os.path.join("data", "uhecr", "sd1500_spectrum.csv")
    
    try:
        analyzer.load_spectrum(spectrum_file)
    except FileNotFoundError:
        print(f"Spectrum file {spectrum_file} not found!")
        print("Run uhecr_spectrum.py first to generate spectrum data.")
        return
    
    # Calculate theoretical predictions
    print("Calculating theoretical predictions...")
    predictions = analyzer.calculate_theoretical_spectra()
    
    print(f"\nThreshold energies (EeV):")
    for model_name, prediction in predictions.items():
        threshold_eev = prediction['threshold_energy'] / 1e18
        print(f"  {model_name}: {threshold_eev:.2f}")
    
    # Perform chi-squared analysis
    print(f"\nChi-squared analysis:")
    chi2_results = analyzer.chi_squared_analysis()
    
    # Exclusion analysis
    print(f"\nExclusion analysis (95% CL):")
    exclusions = analyzer.exclusion_analysis()
    
    for model_name, result in exclusions.items():
        status = "EXCLUDED" if result['excluded'] else "allowed"
        print(f"  {model_name}: {status} (p = {result['p_value']:.4f})")
    
    # Create plots
    print(f"\nGenerating comparison plots...")
    plot_file = "results/uhecr_theoretical_comparison.png"
    os.makedirs("results", exist_ok=True)
    analyzer.plot_theoretical_comparison(save_path=plot_file)
    
    # Save results
    results_data = []
    for model_name in predictions.keys():
        if model_name in chi2_results and model_name in exclusions:
            results_data.append({
                'Model': model_name,
                'Threshold_EeV': predictions[model_name]['threshold_energy'] / 1e18,
                'Chi2_reduced': chi2_results[model_name]['chi2_reduced'],
                'Excluded_95CL': exclusions[model_name]['excluded'],
                'P_value': exclusions[model_name]['p_value'],
                'Parameters': str(predictions[model_name]['parameters'])
            })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df.to_csv("results/uhecr_theoretical_analysis.csv", index=False)
        print(f"\nResults saved to results/uhecr_theoretical_analysis.csv")
        print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
