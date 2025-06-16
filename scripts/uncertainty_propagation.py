#!/usr/bin/env python3
"""
Comprehensive Uncertainty Propagation for Multi-Channel LIV Analysis

This module implements detailed uncertainty propagation for all observational
and theoretical sources across the four LIV constraint channels:

1. GRB Time Delays: Redshift, energy calibration, timing uncertainties
2. UHECR Propagation: Energy reconstruction, stochastic losses, flux uncertainties  
3. Vacuum Instability: EFT parameter uncertainties, field calibration
4. Hidden Sector: Instrumental sensitivity, conversion rate uncertainties

Key features:
- Monte Carlo uncertainty propagation
- Analytic error propagation where tractable
- Systematic and statistical uncertainty separation
- Correlation handling across observational channels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, interpolate
from scipy.special import erfc
import warnings
warnings.filterwarnings('ignore')

class UncertaintyPropagation:
    """
    Comprehensive uncertainty propagation for multi-channel LIV analysis.
    
    This class handles all sources of observational and theoretical uncertainty,
    propagating them through to final parameter constraints using both Monte
    Carlo and analytic methods.
    """
    
    def __init__(self, n_mc_samples=10000, random_seed=42):
        """
        Initialize uncertainty propagation framework.
        
        Parameters:
        -----------
        n_mc_samples : int
            Number of Monte Carlo samples for uncertainty propagation
        random_seed : int
            Random seed for reproducibility
        """
        self.n_mc_samples = n_mc_samples
        np.random.seed(random_seed)
        
        # Physical constants with uncertainties
        self.constants = self._define_physical_constants()
        
        # Systematic uncertainty models
        self.systematic_models = self._define_systematic_models()
        
        # Correlation matrices
        self.correlation_matrices = {}
        
    def _define_physical_constants(self):
        """Define physical constants with uncertainties."""
        return {
            'planck_energy': {'value': 1.22e19, 'uncertainty': 0.01e19},  # GeV
            'schwinger_field': {'value': 1.32e16, 'uncertainty': 0.02e16},  # V/m
            'alpha_em': {'value': 1/137.036, 'uncertainty': 1e-9},
            'electron_mass': {'value': 0.511e-3, 'uncertainty': 1e-8}  # GeV
        }
    
    def _define_systematic_models(self):
        """Define systematic uncertainty models for each channel."""
        return {
            'grb': {
                'redshift_calibration': 0.05,  # 5% systematic in redshift
                'energy_calibration': 0.10,    # 10% energy scale uncertainty
                'timing_systematics': 0.1,     # 0.1s systematic timing offset
                'intrinsic_delays': 1.0,       # 1s intrinsic delay uncertainty
                'detector_response': 0.02       # 2% detector efficiency uncertainty
            },
            'uhecr': {
                'energy_reconstruction': 0.15,  # 15% energy reconstruction uncertainty
                'shower_fluctuations': 0.12,    # 12% shower development uncertainty
                'atmospheric_depth': 0.08,      # 8% atmospheric model uncertainty
                'detector_acceptance': 0.05,    # 5% detector acceptance uncertainty
                'composition_uncertainty': 0.20 # 20% composition-dependent uncertainty
            },
            'vacuum': {
                'field_calibration': 0.05,      # 5% field strength calibration
                'eft_parameters': 0.10,         # 10% EFT parameter uncertainty
                'higher_order_corrections': 0.15, # 15% from neglected terms
                'quantum_corrections': 0.08,    # 8% quantum loop corrections
                'finite_size_effects': 0.03    # 3% finite beam size effects
            },
            'hidden_sector': {
                'instrumental_sensitivity': 0.20, # 20% sensitivity uncertainty
                'background_subtraction': 0.15,   # 15% background uncertainty
                'conversion_efficiency': 0.10,    # 10% detection efficiency
                'dark_coupling_theory': 0.25,     # 25% theoretical uncertainty
                'mixing_angle_uncertainty': 0.30  # 30% mixing parameter uncertainty
            }
        }
    
    def generate_grb_uncertainties(self, grb_data):
        """
        Generate comprehensive uncertainty realizations for GRB data.
        
        Parameters:
        -----------
        grb_data : pd.DataFrame
            GRB observational data
            
        Returns:
        --------
        dict : Uncertainty realizations for all GRB parameters
        """
        n_grbs = len(grb_data)
        
        uncertainties = {
            'redshift_errors': np.zeros((self.n_mc_samples, n_grbs)),
            'energy_errors': np.zeros((self.n_mc_samples, n_grbs)),
            'timing_errors': np.zeros((self.n_mc_samples, n_grbs)),
            'intrinsic_delay_errors': np.zeros((self.n_mc_samples, n_grbs)),
            'detector_response_errors': np.zeros((self.n_mc_samples, n_grbs))
        }
        
        sys_models = self.systematic_models['grb']
        
        for i, grb in grb_data.iterrows():
            redshift = grb['redshift']
            energy = grb['energy_gev']
            time_error = grb['time_error_s']
            
            # Redshift uncertainties (correlated systematic + statistical)
            z_sys = np.random.normal(0, sys_models['redshift_calibration'] * redshift, 
                                   self.n_mc_samples)
            z_stat = np.random.normal(0, 0.01 * redshift, self.n_mc_samples)
            uncertainties['redshift_errors'][:, i] = z_sys + z_stat
            
            # Energy calibration uncertainties
            # Systematic: correlated across all photons
            e_sys = np.random.normal(0, sys_models['energy_calibration'] * energy, 
                                   self.n_mc_samples)
            # Statistical: uncorrelated per photon
            e_stat = np.random.normal(0, 0.05 * energy, self.n_mc_samples)
            uncertainties['energy_errors'][:, i] = e_sys + e_stat
            
            # Timing uncertainties
            # Systematic timing offset (correlated within GRB)
            t_sys = np.random.normal(0, sys_models['timing_systematics'], 
                                   self.n_mc_samples)
            # Statistical timing error
            t_stat = np.random.normal(0, time_error, self.n_mc_samples)
            uncertainties['timing_errors'][:, i] = t_sys + t_stat
            
            # Intrinsic delay uncertainties (astrophysical)
            intrinsic_scale = sys_models['intrinsic_delays'] * np.sqrt(energy)
            uncertainties['intrinsic_delay_errors'][:, i] = np.random.exponential(
                intrinsic_scale, self.n_mc_samples)
            
            # Detector response uncertainties
            det_eff = 1 + np.random.normal(0, sys_models['detector_response'], 
                                         self.n_mc_samples)
            uncertainties['detector_response_errors'][:, i] = det_eff
        
        return uncertainties
    
    def generate_uhecr_uncertainties(self, uhecr_data):
        """
        Generate comprehensive uncertainty realizations for UHECR data.
        
        Parameters:
        -----------
        uhecr_data : pd.DataFrame
            UHECR observational data
            
        Returns:
        --------
        dict : Uncertainty realizations for all UHECR parameters
        """
        n_energy_bins = len(uhecr_data)
        
        uncertainties = {
            'energy_reconstruction_errors': np.zeros((self.n_mc_samples, n_energy_bins)),
            'shower_fluctuation_errors': np.zeros((self.n_mc_samples, n_energy_bins)),
            'atmospheric_errors': np.zeros((self.n_mc_samples, n_energy_bins)),
            'detector_acceptance_errors': np.zeros((self.n_mc_samples, n_energy_bins)),
            'composition_errors': np.zeros((self.n_mc_samples, n_energy_bins)),
            'stochastic_loss_errors': np.zeros((self.n_mc_samples, n_energy_bins))
        }
        
        sys_models = self.systematic_models['uhecr']
        
        for i, point in uhecr_data.iterrows():
            energy = point['energy_ev']
            flux = point['flux']
            flux_error = point['flux_error']
            
            # Energy reconstruction uncertainties
            # Systematic: energy-dependent calibration
            e_sys_frac = sys_models['energy_reconstruction'] * (1 + np.log10(energy/1e19)/10)
            e_reconstruction = np.random.lognormal(0, e_sys_frac, self.n_mc_samples)
            uncertainties['energy_reconstruction_errors'][:, i] = e_reconstruction
            
            # Shower development fluctuations
            # Model shower-to-shower variations
            shower_var = sys_models['shower_fluctuations']
            shower_fluct = np.random.lognormal(0, shower_var, self.n_mc_samples)
            uncertainties['shower_fluctuation_errors'][:, i] = shower_fluct
            
            # Atmospheric depth uncertainties
            # Seasonal and geographical variations
            atm_uncertainty = sys_models['atmospheric_depth']
            atm_errors = np.random.normal(1, atm_uncertainty, self.n_mc_samples)
            uncertainties['atmospheric_errors'][:, i] = atm_errors
            
            # Detector acceptance uncertainties
            # Angular and energy-dependent acceptance
            acc_base = sys_models['detector_acceptance']
            acc_energy_dep = acc_base * (1 + 0.1 * np.log10(energy/1e19))
            acceptance_errors = np.random.normal(1, acc_energy_dep, self.n_mc_samples)
            uncertainties['detector_acceptance_errors'][:, i] = acceptance_errors
            
            # Composition uncertainties
            # Mass composition effects on shower development
            comp_uncertainty = sys_models['composition_uncertainty']
            # Energy-dependent composition evolution
            comp_evolution = comp_uncertainty * (1 + 0.2 * np.log10(energy/1e19))
            composition_errors = np.random.normal(1, comp_evolution, self.n_mc_samples)
            uncertainties['composition_errors'][:, i] = composition_errors
            
            # Stochastic energy loss uncertainties
            # Propagation through cosmic microwave background
            if energy > 5e19:  # GZK cutoff region
                loss_variance = 0.3 * (energy / 5e19)**0.5
                stochastic_losses = np.random.exponential(loss_variance, self.n_mc_samples)
            else:
                stochastic_losses = np.random.normal(1, 0.05, self.n_mc_samples)
            
            uncertainties['stochastic_loss_errors'][:, i] = stochastic_losses
        
        return uncertainties
    
    def generate_vacuum_uncertainties(self, field_strengths, theoretical_params):
        """
        Generate uncertainties for vacuum instability predictions.
        
        Parameters:
        -----------
        field_strengths : array
            Laboratory field strengths (V/m)
        theoretical_params : dict
            LIV model parameters
            
        Returns:
        --------
        dict : Uncertainty realizations for vacuum predictions
        """
        n_fields = len(field_strengths)
        
        uncertainties = {
            'field_calibration_errors': np.zeros((self.n_mc_samples, n_fields)),
            'eft_parameter_errors': np.zeros((self.n_mc_samples, n_fields)),
            'higher_order_errors': np.zeros((self.n_mc_samples, n_fields)),
            'quantum_correction_errors': np.zeros((self.n_mc_samples, n_fields)),
            'finite_size_errors': np.zeros((self.n_mc_samples, n_fields))
        }
        
        sys_models = self.systematic_models['vacuum']
        
        for i, field in enumerate(field_strengths):
            # Field strength calibration uncertainties
            # Systematic calibration error
            field_cal_error = sys_models['field_calibration']
            field_errors = np.random.lognormal(0, field_cal_error, self.n_mc_samples)
            uncertainties['field_calibration_errors'][:, i] = field_errors
            
            # EFT parameter uncertainties
            # Theoretical uncertainties in effective field theory coefficients
            eft_uncertainty = sys_models['eft_parameters']
            # Include correlation with field strength (higher fields → larger uncertainty)
            eft_scale = eft_uncertainty * (1 + 0.1 * np.log10(field / 1e15))
            eft_errors = np.random.lognormal(0, eft_scale, self.n_mc_samples)
            uncertainties['eft_parameter_errors'][:, i] = eft_errors
            
            # Higher-order correction uncertainties
            # Estimate uncertainty from neglected higher-order terms
            higher_order_scale = sys_models['higher_order_corrections']
            # Stronger field dependence for higher-order terms
            ho_field_dep = higher_order_scale * (field / 1e15)**0.5
            ho_errors = np.random.normal(1, ho_field_dep, self.n_mc_samples)
            uncertainties['higher_order_errors'][:, i] = ho_errors
            
            # Quantum loop correction uncertainties
            # One-loop and higher quantum corrections
            quantum_uncertainty = sys_models['quantum_corrections']
            # Loop corrections scale with coupling and field strength
            alpha_em = self.constants['alpha_em']['value']
            quantum_scale = quantum_uncertainty * alpha_em * np.log(field / 1e13)
            quantum_errors = np.random.normal(1, quantum_scale, self.n_mc_samples)
            uncertainties['quantum_correction_errors'][:, i] = quantum_errors
            
            # Finite beam size effects
            # Spatial averaging over finite focus region
            finite_size_uncertainty = sys_models['finite_size_effects']
            # Effect scales with field gradient
            gradient_effect = finite_size_uncertainty * np.sqrt(field / 1e15)
            finite_size_errors = np.random.normal(1, gradient_effect, self.n_mc_samples)
            uncertainties['finite_size_errors'][:, i] = finite_size_errors
        
        return uncertainties
    
    def generate_hidden_sector_uncertainties(self, energy_range, coupling_params):
        """
        Generate uncertainties for hidden sector predictions.
        
        Parameters:
        -----------
        energy_range : array
            Energy range for hidden sector searches (GeV)
        coupling_params : dict
            Hidden sector coupling parameters
            
        Returns:
        --------
        dict : Uncertainty realizations for hidden sector predictions
        """
        n_energies = len(energy_range)
        
        uncertainties = {
            'sensitivity_errors': np.zeros((self.n_mc_samples, n_energies)),
            'background_errors': np.zeros((self.n_mc_samples, n_energies)),
            'efficiency_errors': np.zeros((self.n_mc_samples, n_energies)),
            'coupling_theory_errors': np.zeros((self.n_mc_samples, n_energies)),
            'mixing_angle_errors': np.zeros((self.n_mc_samples, n_energies))
        }
        
        sys_models = self.systematic_models['hidden_sector']
        
        for i, energy in enumerate(energy_range):
            # Instrumental sensitivity uncertainties
            # Energy-dependent sensitivity variations
            sens_base = sys_models['instrumental_sensitivity']
            sens_energy_dep = sens_base * (1 + 0.1 * np.log10(energy))
            sensitivity_errors = np.random.lognormal(0, sens_energy_dep, self.n_mc_samples)
            uncertainties['sensitivity_errors'][:, i] = sensitivity_errors
            
            # Background subtraction uncertainties
            # Statistical and systematic background uncertainties
            bg_uncertainty = sys_models['background_subtraction']
            # Background typically worse at lower energies
            bg_energy_scale = bg_uncertainty * (1 + 1.0 / np.sqrt(energy))
            background_errors = np.random.lognormal(0, bg_energy_scale, self.n_mc_samples)
            uncertainties['background_errors'][:, i] = background_errors
            
            # Detection efficiency uncertainties
            # Conversion and detection efficiency uncertainties
            eff_uncertainty = sys_models['conversion_efficiency']
            efficiency_errors = np.random.normal(1, eff_uncertainty, self.n_mc_samples)
            uncertainties['efficiency_errors'][:, i] = efficiency_errors
            
            # Theoretical coupling uncertainties
            # Uncertainties in dark sector theoretical predictions
            theory_uncertainty = sys_models['dark_coupling_theory']
            # Theory uncertainty grows with energy due to running couplings
            theory_scale = theory_uncertainty * (1 + 0.05 * np.log10(energy))
            theory_errors = np.random.lognormal(0, theory_scale, self.n_mc_samples)
            uncertainties['coupling_theory_errors'][:, i] = theory_errors
            
            # Mixing angle uncertainties
            # Photon-dark photon mixing parameter uncertainties
            mixing_uncertainty = sys_models['mixing_angle_uncertainty']
            mixing_errors = np.random.lognormal(0, mixing_uncertainty, self.n_mc_samples)
            uncertainties['mixing_angle_errors'][:, i] = mixing_errors
        
        return uncertainties
    
    def propagate_grb_uncertainties(self, grb_data, grb_uncertainties, liv_params):
        """
        Propagate GRB uncertainties through to LIV parameter constraints.
        
        Parameters:
        -----------
        grb_data : pd.DataFrame
            GRB observational data
        grb_uncertainties : dict
            GRB uncertainty realizations
        liv_params : dict
            LIV model parameters
            
        Returns:
        --------
        array : Likelihood values with uncertainties propagated
        """
        log_likelihoods = np.zeros(self.n_mc_samples)
        
        mu = 10**liv_params['log_mu']
        coupling = 10**liv_params['log_coupling']
        
        for mc_sample in range(self.n_mc_samples):
            log_like_sample = 0.0
            
            for i, grb in grb_data.iterrows():
                # Apply uncertainty realizations
                z_true = grb['redshift'] + grb_uncertainties['redshift_errors'][mc_sample, i]
                e_true = grb['energy_gev'] * grb_uncertainties['energy_errors'][mc_sample, i]
                t_obs = grb['time_delay_s'] + grb_uncertainties['timing_errors'][mc_sample, i]
                
                # Include intrinsic delay
                t_intrinsic = grb_uncertainties['intrinsic_delay_errors'][mc_sample, i]
                
                # Detector response correction
                det_correction = grb_uncertainties['detector_response_errors'][mc_sample, i]
                
                # LIV prediction with uncertainties
                if z_true > 0 and e_true > 0:
                    # Luminosity distance (simplified)
                    d_luminosity = 3000 * z_true  # Mpc (approximate)
                    
                    # LIV time delay prediction
                    t_liv_pred = coupling * (e_true / mu) * d_luminosity * 1e-15
                    
                    # Total predicted delay
                    t_total_pred = t_liv_pred + t_intrinsic
                    
                    # Corrected observation
                    t_obs_corrected = t_obs * det_correction
                    
                    # Likelihood contribution
                    sigma_eff = grb['time_error_s'] * np.sqrt(1 + 0.1 * e_true)  # Energy-dependent error
                    log_like_sample += stats.norm.logpdf(t_obs_corrected, t_total_pred, sigma_eff)
            
            log_likelihoods[mc_sample] = log_like_sample
        
        return log_likelihoods
    
    def propagate_uhecr_uncertainties(self, uhecr_data, uhecr_uncertainties, liv_params):
        """
        Propagate UHECR uncertainties through to LIV parameter constraints.
        
        Parameters:
        -----------
        uhecr_data : pd.DataFrame
            UHECR observational data
        uhecr_uncertainties : dict
            UHECR uncertainty realizations
        liv_params : dict
            LIV model parameters
            
        Returns:
        --------
        array : Likelihood values with uncertainties propagated
        """
        log_likelihoods = np.zeros(self.n_mc_samples)
        
        mu = 10**liv_params['log_mu'] * 1e9  # Convert to eV
        coupling = 10**liv_params['log_coupling']
        
        for mc_sample in range(self.n_mc_samples):
            log_like_sample = 0.0
            
            for i, point in uhecr_data.iterrows():
                # Apply uncertainty realizations
                e_reco = point['energy_ev'] * uhecr_uncertainties['energy_reconstruction_errors'][mc_sample, i]
                shower_corr = uhecr_uncertainties['shower_fluctuation_errors'][mc_sample, i]
                atm_corr = uhecr_uncertainties['atmospheric_errors'][mc_sample, i]
                acc_corr = uhecr_uncertainties['detector_acceptance_errors'][mc_sample, i]
                comp_corr = uhecr_uncertainties['composition_errors'][mc_sample, i]
                loss_corr = uhecr_uncertainties['stochastic_loss_errors'][mc_sample, i]
                
                # True energy estimate
                e_true = e_reco * shower_corr * atm_corr
                
                # LIV modification to propagation
                if e_true > 0:
                    liv_modification = 1 + coupling * (e_true / mu)**2
                    
                    # Standard model flux expectation
                    flux_sm = 1e-8 * (e_true / 1e19)**(-2.7)
                    
                    # Apply all corrections
                    flux_pred = flux_sm * liv_modification * loss_corr * acc_corr * comp_corr
                    
                    # Observed flux with uncertainties
                    flux_obs = point['flux']
                    flux_error = point['flux_error']
                    
                    # Likelihood contribution
                    if flux_pred > 0:
                        log_like_sample += stats.norm.logpdf(flux_obs, flux_pred, flux_error)
            
            log_likelihoods[mc_sample] = log_like_sample
        
        return log_likelihoods
    
    def propagate_vacuum_uncertainties(self, field_strengths, vacuum_uncertainties, liv_params):
        """
        Propagate vacuum instability uncertainties.
        
        Parameters:
        -----------
        field_strengths : array
            Laboratory field strengths
        vacuum_uncertainties : dict
            Vacuum uncertainty realizations
        liv_params : dict
            LIV model parameters
            
        Returns:
        --------
        array : Enhancement factors with uncertainties
        """
        enhancement_factors = np.zeros((self.n_mc_samples, len(field_strengths)))
        
        mu = 10**liv_params['log_mu']
        coupling = 10**liv_params['log_coupling']
        
        for mc_sample in range(self.n_mc_samples):
            for i, field in enumerate(field_strengths):
                # Apply all uncertainty corrections
                field_corrected = field * vacuum_uncertainties['field_calibration_errors'][mc_sample, i]
                eft_correction = vacuum_uncertainties['eft_parameter_errors'][mc_sample, i]
                ho_correction = vacuum_uncertainties['higher_order_errors'][mc_sample, i]
                quantum_correction = vacuum_uncertainties['quantum_correction_errors'][mc_sample, i]
                finite_correction = vacuum_uncertainties['finite_size_errors'][mc_sample, i]
                
                # Base LIV enhancement
                if field_corrected > 0:
                    base_enhancement = 1 + coupling * (field_corrected / (mu * 1e9))**2
                    
                    # Apply all corrections
                    total_enhancement = (base_enhancement * eft_correction * 
                                       ho_correction * quantum_correction * finite_correction)
                    
                    enhancement_factors[mc_sample, i] = total_enhancement
                else:
                    enhancement_factors[mc_sample, i] = 1.0
        
        return enhancement_factors
    
    def propagate_hidden_sector_uncertainties(self, energy_range, hidden_uncertainties, liv_params):
        """
        Propagate hidden sector uncertainties.
        
        Parameters:
        -----------
        energy_range : array
            Energy range for searches
        hidden_uncertainties : dict
            Hidden sector uncertainty realizations
        liv_params : dict
            LIV model parameters
            
        Returns:
        --------
        array : Conversion rates with uncertainties
        """
        conversion_rates = np.zeros((self.n_mc_samples, len(energy_range)))
        
        mu = 10**liv_params['log_mu']
        coupling = 10**liv_params['log_coupling']
        
        for mc_sample in range(self.n_mc_samples):
            for i, energy in enumerate(energy_range):
                # Apply uncertainty corrections
                sens_corr = hidden_uncertainties['sensitivity_errors'][mc_sample, i]
                bg_corr = hidden_uncertainties['background_errors'][mc_sample, i]
                eff_corr = hidden_uncertainties['efficiency_errors'][mc_sample, i]
                theory_corr = hidden_uncertainties['coupling_theory_errors'][mc_sample, i]
                mixing_corr = hidden_uncertainties['mixing_angle_errors'][mc_sample, i]
                
                # Base conversion rate
                base_rate = coupling**2 * (energy / mu)**2 * 1e-8  # Hz
                
                # Apply all corrections
                total_rate = (base_rate * sens_corr * bg_corr * eff_corr * 
                            theory_corr * mixing_corr)
                
                conversion_rates[mc_sample, i] = max(total_rate, 0)
        
        return conversion_rates
    
    def analyze_uncertainty_contributions(self, uncertainties_dict, observable_name):
        """
        Analyze relative contributions of different uncertainty sources.
        
        Parameters:
        -----------
        uncertainties_dict : dict
            Dictionary of uncertainty arrays
        observable_name : str
            Name of the observable for reporting
            
        Returns:
        --------
        pd.DataFrame : Uncertainty contribution analysis
        """
        contributions = []
        
        for source, uncertainty_array in uncertainties_dict.items():
            # Calculate variance contribution
            if uncertainty_array.ndim == 2:
                # Average over all data points
                mean_variance = np.mean(np.var(uncertainty_array, axis=0))
            else:
                mean_variance = np.var(uncertainty_array)
            
            contributions.append({
                'observable': observable_name,
                'uncertainty_source': source,
                'variance_contribution': mean_variance,
                'rms_uncertainty': np.sqrt(mean_variance),
                'relative_uncertainty': np.sqrt(mean_variance) / (1 + np.sqrt(mean_variance))
            })
        
        df = pd.DataFrame(contributions)
        df['fractional_contribution'] = df['variance_contribution'] / df['variance_contribution'].sum()
        
        return df.sort_values('fractional_contribution', ascending=False)
    
    def generate_uncertainty_budget_plot(self, all_contributions, save_path=None):
        """
        Generate comprehensive uncertainty budget visualization.
        
        Parameters:
        -----------
        all_contributions : list
            List of uncertainty contribution DataFrames
        save_path : str, optional
            Path to save the plot
        """
        # Combine all contributions
        combined_df = pd.concat(all_contributions, ignore_index=True)
        
        # Create subplot for each observable
        observables = combined_df['observable'].unique()
        n_obs = len(observables)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        for i, obs in enumerate(observables):
            obs_data = combined_df[combined_df['observable'] == obs]
            
            # Pie chart of uncertainty contributions
            axes[i].pie(obs_data['fractional_contribution'], 
                       labels=obs_data['uncertainty_source'],
                       autopct='%1.1f%%',
                       colors=colors[:len(obs_data)])
            axes[i].set_title(f'{obs} Uncertainty Budget')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Uncertainty budget plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def run_comprehensive_uncertainty_analysis(self, data_dict, liv_params_list):
        """
        Run comprehensive uncertainty propagation analysis.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing all observational data
        liv_params_list : list
            List of LIV parameter dictionaries to analyze
            
        Returns:
        --------
        dict : Complete uncertainty analysis results
        """
        print("🔬 COMPREHENSIVE UNCERTAINTY PROPAGATION ANALYSIS")
        print("="*60)
        
        results = {}
        uncertainty_contributions = []
        
        for param_set in liv_params_list:
            model_name = param_set.get('model', 'unknown')
            print(f"\n📊 Analyzing {model_name} model uncertainties...")
            
            # Generate uncertainty realizations for all channels
            print("  Generating GRB uncertainties...")
            grb_uncertainties = self.generate_grb_uncertainties(data_dict['grb'])
            
            print("  Generating UHECR uncertainties...")
            uhecr_uncertainties = self.generate_uhecr_uncertainties(data_dict['uhecr'])
            
            print("  Generating vacuum uncertainties...")
            field_strengths = np.array([1e13, 1e15, 1e16])  # V/m
            vacuum_uncertainties = self.generate_vacuum_uncertainties(field_strengths, param_set)
            
            print("  Generating hidden sector uncertainties...")
            energy_range = np.logspace(-3, 3, 20)  # GeV
            hidden_uncertainties = self.generate_hidden_sector_uncertainties(energy_range, param_set)
            
            # Propagate uncertainties through to constraints
            print("  Propagating uncertainties...")
            grb_likelihoods = self.propagate_grb_uncertainties(
                data_dict['grb'], grb_uncertainties, param_set)
            
            uhecr_likelihoods = self.propagate_uhecr_uncertainties(
                data_dict['uhecr'], uhecr_uncertainties, param_set)
            
            vacuum_enhancements = self.propagate_vacuum_uncertainties(
                field_strengths, vacuum_uncertainties, param_set)
            
            hidden_rates = self.propagate_hidden_sector_uncertainties(
                energy_range, hidden_uncertainties, param_set)
            
            # Analyze uncertainty contributions
            grb_contrib = self.analyze_uncertainty_contributions(grb_uncertainties, 'GRB')
            uhecr_contrib = self.analyze_uncertainty_contributions(uhecr_uncertainties, 'UHECR')
            vacuum_contrib = self.analyze_uncertainty_contributions(vacuum_uncertainties, 'Vacuum')
            hidden_contrib = self.analyze_uncertainty_contributions(hidden_uncertainties, 'Hidden')
            
            uncertainty_contributions.extend([grb_contrib, uhecr_contrib, vacuum_contrib, hidden_contrib])
            
            # Store results
            results[model_name] = {
                'grb_likelihoods': grb_likelihoods,
                'uhecr_likelihoods': uhecr_likelihoods,
                'vacuum_enhancements': vacuum_enhancements,
                'hidden_conversion_rates': hidden_rates,
                'uncertainty_budgets': {
                    'grb': grb_contrib,
                    'uhecr': uhecr_contrib,
                    'vacuum': vacuum_contrib,
                    'hidden': hidden_contrib
                }
            }
            
            print(f"  ✓ {model_name} uncertainty analysis complete")
        
        # Generate comprehensive uncertainty budget visualization
        print("\n📈 Generating uncertainty budget plots...")
        self.generate_uncertainty_budget_plot(uncertainty_contributions, 
                                            'results/comprehensive_uncertainty_budget.png')
        
        # Save detailed results
        print("\n💾 Saving results...")
        for model_name, model_results in results.items():
            # Save uncertainty budgets
            for obs_type, budget_df in model_results['uncertainty_budgets'].items():
                filename = f"results/uncertainty_budget_{model_name}_{obs_type}.csv"
                budget_df.to_csv(filename, index=False)
        
        print("✓ Comprehensive uncertainty analysis complete!")
        return results

def main():
    """Demonstration of comprehensive uncertainty propagation."""
    # Initialize uncertainty propagation
    up = UncertaintyPropagation(n_mc_samples=1000)  # Reduced for demo
    
    # Generate mock data
    grb_data = pd.DataFrame({
        'redshift': [0.5, 1.0, 1.5, 2.0],
        'energy_gev': [1.0, 5.0, 10.0, 20.0],
        'time_delay_s': [0.1, 0.3, 0.5, 0.8],
        'time_error_s': [0.05, 0.1, 0.15, 0.2]
    })
    
    uhecr_data = pd.DataFrame({
        'energy_ev': np.logspace(18, 20.5, 10),
        'flux': np.logspace(-8, -6, 10),
        'flux_error': np.logspace(-9, -7, 10)
    })
    
    data_dict = {'grb': grb_data, 'uhecr': uhecr_data}
    
    # Test parameter sets
    liv_params_list = [
        {'model': 'string_theory', 'log_mu': 17.0, 'log_coupling': -8.0},
        {'model': 'axion_like', 'log_mu': 18.0, 'log_coupling': -7.0}
    ]
    
    # Run comprehensive analysis
    results = up.run_comprehensive_uncertainty_analysis(data_dict, liv_params_list)
    
    print("🎉 Uncertainty propagation demo complete!")

if __name__ == "__main__":
    main()
