#!/usr/bin/env python3
"""
Hidden Sector Coupling Module: Photon→Dark-Photon Conversion

This module implements theoretical predictions for photon→dark-photon conversion
probabilities as a function of energy, with applications to:
1. GRB spectral attenuation analysis
2. Polymer-quantum gravity models
3. Rainbow gravity scenarios
4. Laboratory dark photon searches

Key Physics:
- Photon-dark photon oscillations: P(γ→γ') = sin²(2θ) × sin²(πL/L_osc)
- Energy-dependent mixing: θ(E) depends on LIV model parameters
- Attenuation effects: Observable signatures in GRB spectra
- Laboratory predictions: Dark photon detection rates

Integration with LIV Pipeline:
- Connects to GRB analysis for parameter constraints
- Links to vacuum instability predictions
- Provides unified model testing framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# Physical constants and scales
class HiddenSectorConstants:
    """Constants for hidden sector physics calculations."""
    
    # Fundamental constants
    SPEED_OF_LIGHT = 2.998e8          # m/s
    PLANCK_LENGTH = 1.616e-35         # m
    PLANCK_ENERGY = 1.22e19           # GeV
    ELECTRON_MASS = 0.511e-3          # GeV
    
    # Astrophysical scales
    KPC_TO_METERS = 3.086e19          # m
    GRB_DISTANCE_TYPICAL = 1e9 * KPC_TO_METERS  # ~1 Gpc in meters
    
    # Energy scales
    GEV_TO_HZ = 2.418e23              # Hz per GeV
    COMPTON_WAVELENGTH = 2.426e-12    # m (electron)
    
    # Laboratory scales
    LAB_BASELINE_KM = 1.0             # km baseline for dark photon searches
    LAB_LASER_POWER_MW = 1.0          # MW laser power

CONST = HiddenSectorConstants()

class HiddenSectorCouplingModel:
    """
    Photon-dark photon coupling model with energy-dependent mixing.
    
    This class implements various theoretical models for how photons
    couple to dark sector particles, with predictions for both
    astrophysical observations and laboratory experiments.
    """
    
    def __init__(self, model_type='polymer_quantum', base_coupling=1e-10):
        """
        Initialize hidden sector coupling model.
        
        Parameters:
        -----------
        model_type : str
            Type of LIV model ('polymer_quantum', 'rainbow_gravity', 'string_theory')
        base_coupling : float
            Base coupling strength (dimensionless)
        """
        self.model_type = model_type
        self.base_coupling = base_coupling
        self.model_params = {}
        
        print(f"🌑 Hidden Sector Model Initialized: {model_type}")
        print(f"   Base coupling: {base_coupling:.2e}")
    
    def energy_dependent_mixing_angle(self, E_GeV, mu_LIV_GeV):
        """
        Compute energy-dependent photon-dark photon mixing angle.
        
        Parameters:
        -----------
        E_GeV : float or array
            Photon energy in GeV
        mu_LIV_GeV : float
            LIV energy scale in GeV
            
        Returns:
        --------
        theta : float or array
            Mixing angle θ(E) in radians
        """
        E = np.asarray(E_GeV)
        
        # Dimensionless energy parameter
        x = E / mu_LIV_GeV
        
        if self.model_type == 'polymer_quantum':
            # Polymer quantum gravity: θ(E) ∝ (E/μ)^n with n=1,2,3
            n_power = 2  # Quadratic dependence typical
            theta = self.base_coupling * x**n_power
            
        elif self.model_type == 'rainbow_gravity':
            # Rainbow gravity: θ(E) ∝ 1 - (E/μ)^α with threshold behavior
            alpha = 1.0
            theta = self.base_coupling * (1 - np.exp(-x**alpha))
            
        elif self.model_type == 'string_theory':
            # String theory: θ(E) ∝ sin(E/M_string) oscillatory behavior
            theta = self.base_coupling * np.abs(np.sin(x))
            
        elif self.model_type == 'axion_like':
            # Axion-like particles: θ(E) ∝ 1/(1 + (E/μ)²) resonant
            theta = self.base_coupling / (1 + x**2)
            
        elif self.model_type == 'extra_dimensions':
            # Extra dimensions: θ(E) ∝ (E/μ)^n × exp(-E/μ) 
            n_power = 1
            theta = self.base_coupling * x**n_power * np.exp(-x)
            
        else:
            # Constant mixing angle
            theta = self.base_coupling * np.ones_like(x)
        
        # Ensure physical bounds: 0 < θ < π/2
        theta = np.clip(theta, 1e-20, np.pi/2 - 1e-6)
        
        return theta
    
    def oscillation_length(self, E_GeV, mass_dark_photon_eV=1e-6):
        """
        Compute photon-dark photon oscillation length.
        
        Parameters:
        -----------
        E_GeV : float or array
            Photon energy in GeV
        mass_dark_photon_eV : float
            Dark photon mass in eV
            
        Returns:
        --------
        L_osc : float or array
            Oscillation length in meters
        """
        E = np.asarray(E_GeV)
        
        # Convert mass to GeV
        m_gamma_prime = mass_dark_photon_eV * 1e-9  # eV to GeV
        
        # Oscillation length: L_osc = 4πE / Δm²
        # For small mass splitting: Δm² ≈ m_γ'²
        if m_gamma_prime > 0:
            L_osc = 4 * np.pi * E / (m_gamma_prime**2)
            # Convert to meters (using ℏc)
            L_osc *= 1.973e-13  # GeV⁻¹ to meters
        else:
            # Massless case - very long oscillation length
            L_osc = 1e20 * np.ones_like(E)  # Effectively infinite
        
        return L_osc
    
    def photon_conversion_probability(self, E_GeV, distance_meters, 
                                    mu_LIV_GeV, mass_dark_photon_eV=1e-6):
        """
        Calculate photon→dark-photon conversion probability.
        
        This is the core observable: P(γ→γ') as function of energy.
        
        Parameters:
        -----------
        E_GeV : float or array
            Photon energy in GeV
        distance_meters : float
            Propagation distance in meters
        mu_LIV_GeV : float
            LIV energy scale in GeV
        mass_dark_photon_eV : float
            Dark photon mass in eV
            
        Returns:
        --------
        P_conversion : float or array
            Conversion probability (0 to 1)
        """
        E = np.asarray(E_GeV)
        
        # Get mixing angle and oscillation length
        theta = self.energy_dependent_mixing_angle(E, mu_LIV_GeV)
        L_osc = self.oscillation_length(E, mass_dark_photon_eV)
        
        # Conversion probability: P = sin²(2θ) × sin²(πL/L_osc)
        sin2_2theta = np.sin(2 * theta)**2
        oscillation_phase = np.pi * distance_meters / L_osc
        sin2_phase = np.sin(oscillation_phase)**2
        
        P_conversion = sin2_2theta * sin2_phase
        
        # Handle numerical issues
        P_conversion = np.clip(P_conversion, 0, 1)
        
        return P_conversion
    
    def grb_attenuation_spectrum(self, E_GeV, distance_meters, mu_LIV_GeV,
                               intrinsic_spectrum_func=None):
        """
        Predict GRB spectrum after dark photon conversion losses.
        
        Parameters:
        -----------
        E_GeV : array
            Energy bins in GeV
        distance_meters : float
            GRB distance in meters
        mu_LIV_GeV : float
            LIV scale in GeV
        intrinsic_spectrum_func : callable, optional
            Intrinsic GRB spectrum dN/dE (if None, use power law)
            
        Returns:
        --------
        observed_spectrum : array
            Attenuated spectrum dN/dE
        """
        E = np.asarray(E_GeV)
        
        # Default intrinsic spectrum: dN/dE ∝ E^(-2) exp(-E/E_cut)
        if intrinsic_spectrum_func is None:
            E_cut = 1.0  # GeV cutoff
            intrinsic_spectrum = E**(-2) * np.exp(-E / E_cut)
        else:
            intrinsic_spectrum = intrinsic_spectrum_func(E)
        
        # Conversion probability
        P_conv = self.photon_conversion_probability(E, distance_meters, mu_LIV_GeV)
        
        # Observed spectrum: N_obs = N_intrinsic × (1 - P_conversion)
        observed_spectrum = intrinsic_spectrum * (1 - P_conv)
        
        return observed_spectrum
    
    def laboratory_conversion_rate(self, E_GeV, baseline_meters, laser_power_MW,
                                 mu_LIV_GeV, detection_efficiency=0.1):
        """
        Predict laboratory dark photon conversion rate.
        
        Parameters:
        -----------
        E_GeV : float or array
            Laser photon energy in GeV
        baseline_meters : float
            Laboratory baseline in meters
        laser_power_MW : float
            Laser power in MW
        mu_LIV_GeV : float
            LIV scale in GeV
        detection_efficiency : float
            Detection efficiency
            
        Returns:
        --------
        conversion_rate : float or array
            Dark photon production rate (Hz)
        """
        E = np.asarray(E_GeV)
        
        # Number of photons per second
        photon_energy_J = E * 1.602e-10  # GeV to Joules
        laser_power_W = laser_power_MW * 1e6
        photon_rate_Hz = laser_power_W / photon_energy_J
        
        # Conversion probability
        P_conv = self.photon_conversion_probability(E, baseline_meters, mu_LIV_GeV)
        
        # Dark photon production rate
        conversion_rate = photon_rate_Hz * P_conv * detection_efficiency
        
        return conversion_rate


class GRBHiddenSectorAnalysis:
    """
    GRB spectral analysis for hidden sector coupling constraints.
    
    This class fits hidden sector models to GRB attenuation data
    and extracts constraints on model parameters.
    """
    
    def __init__(self, grb_data_file=None):
        """
        Initialize GRB analysis.
        
        Parameters:
        -----------
        grb_data_file : str, optional
            Path to GRB spectral data file
        """
        self.grb_data = None
        self.model_fits = {}
        
        if grb_data_file and os.path.exists(grb_data_file):
            self.load_grb_data(grb_data_file)
        else:
            self.generate_mock_grb_data()
            
        print(f"📡 GRB Hidden Sector Analysis Initialized")
    
    def generate_mock_grb_data(self):
        """Generate mock GRB spectral data for testing."""
        
        # Energy range: 0.1 keV to 10 GeV
        E_keV = np.logspace(-1, 10, 50)  # keV
        E_GeV = E_keV * 1e-6  # Convert to GeV
        
        # Mock intrinsic spectrum: Band function approximation
        alpha = -1.0  # Low energy index
        beta = -2.5   # High energy index
        E_peak = 300  # keV
        
        # Band function (simplified)
        E_norm = E_keV / E_peak
        spectrum = np.where(E_keV < E_peak,
                          E_norm**alpha * np.exp(-E_norm),
                          E_norm**beta)
        
        # Add some mock attenuation at high energies (simulate dark photon effect)
        attenuation = 1 - 0.1 * np.exp(-(E_keV/1000)**2)  # 10% attenuation around 1 MeV
        observed_spectrum = spectrum * attenuation
        
        # Add realistic uncertainties
        uncertainties = 0.1 * observed_spectrum + 0.01 * np.max(observed_spectrum)
        
        self.grb_data = pd.DataFrame({
            'energy_keV': E_keV,
            'energy_GeV': E_GeV,
            'intrinsic_spectrum': spectrum,
            'observed_spectrum': observed_spectrum,
            'uncertainty': uncertainties,
            'grb_distance_Gpc': 1.5  # Typical GRB distance
        })
        
        print(f"   Generated mock GRB data: {len(E_keV)} energy bins")
    
    def fit_hidden_sector_model(self, model_type='polymer_quantum', 
                               mu_range=(1e15, 1e19)):
        """
        Fit hidden sector model to GRB attenuation data.
        
        Parameters:
        -----------
        model_type : str
            Hidden sector model type
        mu_range : tuple
            Range of LIV scales to test (GeV)
            
        Returns:
        --------
        fit_results : dict
            Best-fit parameters and goodness of fit
        """
        print(f"\n🔍 Fitting {model_type} model to GRB data...")
        
        # Initialize model
        model = HiddenSectorCouplingModel(model_type=model_type)
        
        # GRB distance in meters
        distance_m = self.grb_data['grb_distance_Gpc'].iloc[0] * 1e9 * CONST.KPC_TO_METERS
        
        # Energy and observed spectrum
        E_GeV = self.grb_data['energy_GeV'].values
        observed = self.grb_data['observed_spectrum'].values
        uncertainty = self.grb_data['uncertainty'].values
        intrinsic = self.grb_data['intrinsic_spectrum'].values
        
        def intrinsic_func(E):
            # Interpolate intrinsic spectrum
            return np.interp(E, E_GeV, intrinsic)
        
        def chi_squared(params):
            """Chi-squared objective function."""
            log_mu, log_coupling = params
            mu_LIV = 10**log_mu
            coupling = 10**log_coupling
            
            # Update model parameters
            model.base_coupling = coupling
            
            # Predict attenuated spectrum
            predicted = model.grb_attenuation_spectrum(
                E_GeV, distance_m, mu_LIV, intrinsic_func
            )
            
            # Chi-squared calculation
            chi2 = np.sum(((observed - predicted) / uncertainty)**2)
            return chi2
        
        # Parameter bounds: log10(mu) in [15, 19], log10(coupling) in [-15, -5]
        from scipy.optimize import minimize
        
        # Grid search for initial guess
        log_mu_test = np.linspace(np.log10(mu_range[0]), np.log10(mu_range[1]), 10)
        log_coupling_test = np.linspace(-15, -5, 10)
        
        best_chi2 = np.inf
        best_params = None
        
        for log_mu in log_mu_test:
            for log_coupling in log_coupling_test:
                chi2 = chi_squared([log_mu, log_coupling])
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = [log_mu, log_coupling]
        
        # Refine with optimization
        result = minimize(chi_squared, best_params, 
                         bounds=[(np.log10(mu_range[0]), np.log10(mu_range[1])),
                                (-15, -5)],
                         method='L-BFGS-B')
        
        if result.success:
            best_log_mu, best_log_coupling = result.x
            best_mu = 10**best_log_mu
            best_coupling = 10**best_log_coupling
            final_chi2 = result.fun
            
            # Calculate degrees of freedom and reduced chi-squared
            n_data = len(observed)
            n_params = 2
            dof = n_data - n_params
            reduced_chi2 = final_chi2 / dof
            
            fit_results = {
                'model_type': model_type,
                'best_mu_GeV': best_mu,
                'best_coupling': best_coupling,
                'chi_squared': final_chi2,
                'reduced_chi_squared': reduced_chi2,
                'degrees_of_freedom': dof,
                'fit_success': True,
                'optimization_result': result
            }
            
            print(f"   ✅ Fit successful:")
            print(f"      Best μ: {best_mu:.2e} GeV")
            print(f"      Best coupling: {best_coupling:.2e}")
            print(f"      χ²/dof: {reduced_chi2:.2f}")
            
        else:
            fit_results = {
                'model_type': model_type,
                'fit_success': False,
                'optimization_result': result
            }
            print(f"   ❌ Fit failed: {result.message}")
        
        self.model_fits[model_type] = fit_results
        return fit_results
    
    def compare_models(self):
        """
        Compare different hidden sector models using AIC/BIC criteria.
        """
        if not self.model_fits:
            print("❌ No model fits available. Run fit_hidden_sector_model() first.")
            return
        
        print(f"\n📊 MODEL COMPARISON")
        print("=" * 60)
        
        # Calculate AIC and BIC for each successful fit
        n_data = len(self.grb_data)
        
        comparison_results = []
        
        for model_type, fit_result in self.model_fits.items():
            if fit_result['fit_success']:
                chi2 = fit_result['chi_squared']
                k = 2  # Number of parameters
                
                # Akaike Information Criterion
                AIC = chi2 + 2 * k
                
                # Bayesian Information Criterion  
                BIC = chi2 + k * np.log(n_data)
                
                comparison_results.append({
                    'model': model_type,
                    'chi2': chi2,
                    'chi2_reduced': fit_result['reduced_chi_squared'],
                    'AIC': AIC,
                    'BIC': BIC,
                    'mu_GeV': fit_result['best_mu_GeV'],
                    'coupling': fit_result['best_coupling']
                })
        
        if comparison_results:
            # Sort by AIC (lower is better)
            comparison_results.sort(key=lambda x: x['AIC'])
            
            print(f"{'Model':<20} {'χ²/dof':<10} {'AIC':<10} {'BIC':<10} {'μ (GeV)':<12} {'Coupling':<12}")
            print("-" * 80)
            
            for result in comparison_results:
                print(f"{result['model']:<20} "
                      f"{result['chi2_reduced']:<10.2f} "
                      f"{result['AIC']:<10.1f} "
                      f"{result['BIC']:<10.1f} "
                      f"{result['mu_GeV']:<12.2e} "
                      f"{result['coupling']:<12.2e}")
            
            # Best model
            best_model = comparison_results[0]
            print(f"\n🏆 BEST MODEL: {best_model['model']}")
            print(f"   Preferred parameters:")
            print(f"   μ = {best_model['mu_GeV']:.2e} GeV")
            print(f"   g = {best_model['coupling']:.2e}")
            
            return comparison_results
        
        else:
            print("❌ No successful fits to compare")
            return []


class HiddenSectorLaboratoryPredictor:
    """
    Predict laboratory signatures from GRB-constrained hidden sector models.
    """
    
    def __init__(self, grb_analysis):
        """
        Initialize laboratory predictor.
        
        Parameters:
        -----------
        grb_analysis : GRBHiddenSectorAnalysis
            GRB analysis with fitted models
        """
        self.grb_analysis = grb_analysis
        self.lab_predictions = {}
        
        print(f"🔬 Laboratory Predictor Initialized")
    
    def predict_dark_photon_searches(self):
        """
        Predict dark photon search sensitivity for constrained models.
        """
        print(f"\n🔍 LABORATORY DARK PHOTON SEARCH PREDICTIONS")
        print("=" * 60)
        
        # Laboratory parameters
        lab_scenarios = {
            'Optical Laser': {
                'energy_eV': 2.0,  # Optical photon
                'baseline_m': 1000,  # 1 km baseline
                'power_MW': 1.0,
                'detection_efficiency': 0.1
            },
            'X-ray FEL': {
                'energy_eV': 10000,  # 10 keV X-rays
                'baseline_m': 100,   # 100 m baseline
                'power_MW': 0.001,   # 1 kW
                'detection_efficiency': 0.01
            },
            'Gamma-ray Laser': {
                'energy_eV': 1e6,    # 1 MeV gamma rays
                'baseline_m': 10,    # 10 m baseline
                'power_MW': 0.0001,  # 100 W
                'detection_efficiency': 0.001
            }
        }
        
        for model_type, fit_result in self.grb_analysis.model_fits.items():
            if not fit_result['fit_success']:
                continue
                
            print(f"\n📍 {model_type.upper()} MODEL PREDICTIONS")
            print("-" * 40)
            
            # Extract best-fit parameters
            mu_GeV = fit_result['best_mu_GeV']
            coupling = fit_result['best_coupling']
            
            # Initialize model with best-fit parameters
            model = HiddenSectorCouplingModel(model_type=model_type, 
                                            base_coupling=coupling)
            
            predictions = []
            
            for scenario_name, params in lab_scenarios.items():
                E_GeV = params['energy_eV'] * 1e-9  # eV to GeV
                
                # Predict conversion rate
                conversion_rate = model.laboratory_conversion_rate(
                    E_GeV, params['baseline_m'], params['power_MW'],
                    mu_GeV, params['detection_efficiency']
                )
                
                print(f"  {scenario_name}:")
                print(f"    Energy: {params['energy_eV']:.1e} eV")
                print(f"    Conversion rate: {conversion_rate:.2e} Hz")
                print(f"    Events/day: {conversion_rate * 86400:.2e}")
                
                # Assessment
                if conversion_rate * 86400 > 1:
                    assessment = "🚀 DETECTABLE"
                elif conversion_rate * 86400 > 0.01:
                    assessment = "✅ Marginal"
                else:
                    assessment = "❌ Below threshold"
                
                print(f"    Assessment: {assessment}")
                
                predictions.append({
                    'scenario': scenario_name,
                    'model': model_type,
                    'energy_eV': params['energy_eV'],
                    'conversion_rate_Hz': conversion_rate,
                    'events_per_day': conversion_rate * 86400,
                    'detectable': conversion_rate * 86400 > 1
                })
            
            self.lab_predictions[model_type] = predictions
        
        return self.lab_predictions
    
    def vacuum_instability_cross_check(self):
        """
        Cross-check hidden sector predictions with vacuum instability analysis.
        """
        print(f"\n🔗 VACUUM INSTABILITY CROSS-CHECK")
        print("=" * 50)
        
        # Import vacuum instability system
        try:
            from vacuum_instability_final import VacuumInstabilitySystem
            
            for model_type, fit_result in self.grb_analysis.model_fits.items():
                if not fit_result['fit_success']:
                    continue
                
                print(f"\n📍 {model_type.upper()} MODEL")
                print("-" * 30)
                
                mu_GeV = fit_result['best_mu_GeV']
                
                # Initialize vacuum instability system
                vacuum_system = VacuumInstabilitySystem(model='resonant', coupling=1.0)
                
                # Test at laboratory field strengths
                lab_fields = [1e13, 1e15, 1e16]  # V/m
                
                for E_field in lab_fields:
                    enhancement = vacuum_system.compute_liv_enhancement(E_field, mu_GeV)
                    log_gamma = vacuum_system.compute_gamma_enhanced(E_field, mu_GeV)
                    
                    print(f"  E = {E_field:.1e} V/m:")
                    print(f"    Enhancement: {enhancement:.2e}×")
                    print(f"    log₁₀(Γ): {log_gamma:.1f}")
                    
                    if enhancement > 10:
                        print(f"    🚀 Significant vacuum enhancement!")
                    elif enhancement > 2:
                        print(f"    ✅ Moderate enhancement")
                    else:
                        print(f"    ❌ Minimal enhancement")
                
        except ImportError:
            print("   Vacuum instability module not available for cross-check")


class HiddenSectorVisualizer:
    """
    Comprehensive visualization for hidden sector analysis.
    """
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_grb_spectral_fits(self, grb_analysis):
        """Plot GRB spectral data and model fits."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get GRB data
        grb_data = grb_analysis.grb_data
        E_keV = grb_data['energy_keV'].values
        E_GeV = grb_data['energy_GeV'].values
        observed = grb_data['observed_spectrum'].values
        intrinsic = grb_data['intrinsic_spectrum'].values
        uncertainty = grb_data['uncertainty'].values
        
        # Plot 1: Observed vs intrinsic spectrum
        ax = axes[0, 0]
        ax.loglog(E_keV, intrinsic, 'b--', label='Intrinsic Spectrum', alpha=0.7)
        ax.errorbar(E_keV, observed, yerr=uncertainty, fmt='ro', 
                   label='Observed Data', alpha=0.7, markersize=3)
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('dN/dE (arbitrary units)')
        ax.set_title('GRB Spectral Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Model fits overlay
        ax = axes[0, 1]
        ax.errorbar(E_keV, observed, yerr=uncertainty, fmt='ko', 
                   label='Data', alpha=0.7, markersize=3)
        
        # Plot fitted models
        distance_m = grb_data['grb_distance_Gpc'].iloc[0] * 1e9 * CONST.KPC_TO_METERS
        
        def intrinsic_func(E):
            return np.interp(E, E_GeV, intrinsic)
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_type, fit_result) in enumerate(grb_analysis.model_fits.items()):
            if fit_result['fit_success']:
                model = HiddenSectorCouplingModel(model_type=model_type,
                                                base_coupling=fit_result['best_coupling'])
                
                predicted = model.grb_attenuation_spectrum(
                    E_GeV, distance_m, fit_result['best_mu_GeV'], intrinsic_func
                )
                
                ax.loglog(E_keV, predicted, color=colors[i % len(colors)], 
                         label=f'{model_type} fit', alpha=0.8)
        
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('dN/dE (arbitrary units)')
        ax.set_title('Model Fits to GRB Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Conversion probability vs energy
        ax = axes[1, 0]
        
        for i, (model_type, fit_result) in enumerate(grb_analysis.model_fits.items()):
            if fit_result['fit_success']:
                model = HiddenSectorCouplingModel(model_type=model_type,
                                                base_coupling=fit_result['best_coupling'])
                
                P_conv = model.photon_conversion_probability(
                    E_GeV, distance_m, fit_result['best_mu_GeV']
                )
                
                ax.loglog(E_keV, P_conv, color=colors[i % len(colors)], 
                         label=f'{model_type}', alpha=0.8)
        
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Conversion Probability')
        ax.set_title('Photon→Dark Photon Conversion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Parameter constraints
        ax = axes[1, 1]
        
        successful_fits = [(name, fit) for name, fit in grb_analysis.model_fits.items() 
                          if fit['fit_success']]
        
        if successful_fits:
            model_names = [fit[0] for fit in successful_fits]
            mu_values = [fit[1]['best_mu_GeV'] for fit in successful_fits]
            coupling_values = [fit[1]['best_coupling'] for fit in successful_fits]
            
            scatter = ax.scatter(mu_values, coupling_values, 
                               c=range(len(mu_values)), s=100, 
                               cmap='viridis', alpha=0.8)
            
            for i, name in enumerate(model_names):
                ax.annotate(name, (mu_values[i], coupling_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('LIV Scale μ (GeV)')
            ax.set_ylabel('Coupling Strength g')
            ax.set_title('Best-Fit Parameters')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hidden_sector_grb_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GRB analysis plot saved: {self.output_dir}/hidden_sector_grb_analysis.png")
    
    def plot_laboratory_predictions(self, lab_predictions):
        """Plot laboratory dark photon search predictions."""
        
        if not lab_predictions:
            print("No laboratory predictions to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect all predictions
        all_predictions = []
        for model_type, predictions in lab_predictions.items():
            for pred in predictions:
                pred['model'] = model_type
                all_predictions.append(pred)
        
        df_pred = pd.DataFrame(all_predictions)
        
        # Plot 1: Conversion rates by scenario and model
        scenarios = df_pred['scenario'].unique()
        models = df_pred['model'].unique()
        
        x_pos = np.arange(len(scenarios))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = df_pred[df_pred['model'] == model]
            rates = [model_data[model_data['scenario'] == scenario]['conversion_rate_Hz'].iloc[0] 
                    if len(model_data[model_data['scenario'] == scenario]) > 0 else 0
                    for scenario in scenarios]
            
            ax1.bar(x_pos + i * width, rates, width, label=model, alpha=0.8)
        
        ax1.set_xlabel('Laboratory Scenario')
        ax1.set_ylabel('Conversion Rate (Hz)')
        ax1.set_title('Dark Photon Conversion Rates')
        ax1.set_yscale('log')
        ax1.set_xticks(x_pos + width * (len(models) - 1) / 2)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add detectability threshold line
        ax1.axhline(1/(24*3600), color='red', linestyle='--', 
                   alpha=0.7, label='Daily Detection Threshold')
        
        # Plot 2: Energy dependence of conversion probability
        E_range = np.logspace(-3, 6, 100)  # eV
        
        for model_type, fit_result in lab_predictions.items():
            if len(fit_result) > 0:
                # Use first prediction to get model parameters
                sample_pred = fit_result[0]
                
                # Recreate model (this is approximate - would need access to original fit)
                model = HiddenSectorCouplingModel(model_type=model_type, base_coupling=1e-10)
                
                # Calculate conversion probability vs energy
                E_GeV = E_range * 1e-9
                baseline_m = 1000  # 1 km baseline
                
                P_conv = model.photon_conversion_probability(E_GeV, baseline_m, 1e16)  # Approximate μ
                
                ax2.loglog(E_range, P_conv, label=model_type, alpha=0.8)
        
        ax2.set_xlabel('Photon Energy (eV)')
        ax2.set_ylabel('Conversion Probability')
        ax2.set_title('Energy Dependence of Conversion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark laboratory energy ranges
        ax2.axvspan(1, 10, alpha=0.2, color='red', label='Optical')
        ax2.axvspan(1e3, 1e5, alpha=0.2, color='blue', label='X-ray')
        ax2.axvspan(1e5, 1e7, alpha=0.2, color='green', label='Gamma-ray')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hidden_sector_lab_predictions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Laboratory predictions plot saved: {self.output_dir}/hidden_sector_lab_predictions.png")


def main():
    """
    Main execution: Complete hidden sector coupling analysis.
    """
    print("🌑 HIDDEN SECTOR COUPLING MODULE")
    print("=" * 60)
    print("Photon→Dark-Photon Conversion Analysis with GRB Constraints")
    print("=" * 60)
    
    # Initialize GRB analysis
    grb_analysis = GRBHiddenSectorAnalysis()
    
    # Test different hidden sector models
    models_to_test = ['polymer_quantum', 'rainbow_gravity', 'string_theory', 'axion_like']
    
    print(f"\n🔍 FITTING HIDDEN SECTOR MODELS TO GRB DATA")
    print("=" * 50)
    
    for model_type in models_to_test:
        grb_analysis.fit_hidden_sector_model(model_type)
    
    # Compare models
    comparison_results = grb_analysis.compare_models()
    
    # Laboratory predictions
    lab_predictor = HiddenSectorLaboratoryPredictor(grb_analysis)
    lab_predictions = lab_predictor.predict_dark_photon_searches()
    
    # Cross-check with vacuum instability
    lab_predictor.vacuum_instability_cross_check()
    
    # Generate visualizations
    visualizer = HiddenSectorVisualizer()
    visualizer.plot_grb_spectral_fits(grb_analysis)
    visualizer.plot_laboratory_predictions(lab_predictions)
    
    # Save results
    output_dir = 'results'
    
    # Save model fits
    if comparison_results:
        pd.DataFrame(comparison_results).to_csv(
            f'{output_dir}/hidden_sector_model_comparison.csv', index=False
        )
    
    # Save laboratory predictions
    all_lab_predictions = []
    for model_type, predictions in lab_predictions.items():
        for pred in predictions:
            pred['model'] = model_type
            all_lab_predictions.append(pred)
    
    if all_lab_predictions:
        pd.DataFrame(all_lab_predictions).to_csv(
            f'{output_dir}/hidden_sector_lab_predictions.csv', index=False
        )
    
    # Summary report
    print(f"\n{'=' * 60}")
    print("🎯 HIDDEN SECTOR ANALYSIS SUMMARY")
    print(f"{'=' * 60}")
    
    if comparison_results:
        best_model = comparison_results[0]
        print(f"\n🏆 BEST-FIT MODEL: {best_model['model'].upper()}")
        print(f"   LIV scale: μ = {best_model['mu_GeV']:.2e} GeV")
        print(f"   Coupling: g = {best_model['coupling']:.2e}")
        print(f"   Reduced χ²: {best_model['chi2_reduced']:.2f}")
        
        # Laboratory detectability
        detectable_predictions = [pred for pred in all_lab_predictions 
                                if pred['detectable']]
        
        if detectable_predictions:
            print(f"\n🚀 DETECTABLE LABORATORY SIGNATURES:")
            for pred in detectable_predictions:
                print(f"   {pred['scenario']} ({pred['model']}): "
                      f"{pred['events_per_day']:.1e} events/day")
        else:
            print(f"\n❌ No detectable laboratory signatures predicted")
    
    else:
        print(f"\n❌ No successful model fits obtained")
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   - {output_dir}/hidden_sector_model_comparison.csv")
    print(f"   - {output_dir}/hidden_sector_lab_predictions.csv")
    print(f"   - {output_dir}/hidden_sector_grb_analysis.png")
    print(f"   - {output_dir}/hidden_sector_lab_predictions.png")
    
    print(f"\n✅ HIDDEN SECTOR COUPLING ANALYSIS COMPLETE")
    print(f"   Ready for integration with LIV physics pipeline")
    
    return grb_analysis, lab_predictions, comparison_results


if __name__ == "__main__":
    grb_analysis, lab_predictions, model_comparison = main()
