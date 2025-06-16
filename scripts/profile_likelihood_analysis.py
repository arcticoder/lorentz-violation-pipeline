#!/usr/bin/env python3
"""
Profile Likelihood Analysis for Multi-Channel LIV Compatibility

This module implements profile likelihood analysis to identify regions of parameter
space where different LIV models are jointly compatible across all observational
channels after marginalizing over nuisance parameters.

Mathematical Framework:
For parameter of interest θ_k, the profile likelihood is:
    L_profile(θ_k) = max_{θ_{j≠k}} L(θ)

This identifies the maximum likelihood achievable for each value of θ_k after
optimizing over all other parameters (including nuisance parameters).

Key Features:
- Multi-dimensional parameter space profiling
- Nuisance parameter marginalization
- Cross-channel compatibility assessment
- Joint constraint regions
- Model overlap identification
- Statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, stats
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.special import erfc
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

class ProfileLikelihoodAnalysis:
    """
    Comprehensive profile likelihood analysis for multi-channel LIV constraints.
    
    This class implements profile likelihood calculations to identify regions
    of parameter space where different models are jointly compatible across
    multiple observational channels.
    """
    
    def __init__(self, confidence_levels=[0.68, 0.95, 0.99]):
        """
        Initialize profile likelihood analysis.
        
        Parameters:
        -----------
        confidence_levels : list
            Confidence levels for contour generation (default: 68%, 95%, 99%)
        """
        self.confidence_levels = confidence_levels
        self.parameter_grids = {}
        self.profile_likelihoods = {}
        self.compatibility_regions = {}
        self.nuisance_parameters = {}
        
        # Physical parameter ranges for LIV models
        self.parameter_ranges = {
            'log_mu': (15.0, 20.0),      # Log10 LIV scale in GeV
            'log_coupling': (-10.0, -5.0), # Log10 coupling strength
            'spectral_index': (1.0, 3.0),   # Dispersion relation power
            'threshold_factor': (0.1, 10.0), # Threshold enhancement
            'ln_norm': (-2.0, 2.0),         # Log normalization factor
            'background_level': (0.5, 2.0)  # Background normalization
        }
        
        # Chi-square thresholds for confidence levels
        self.chi2_thresholds = {
            0.68: 2.30,  # 68% CL for 2 DOF
            0.95: 5.99,  # 95% CL for 2 DOF  
            0.99: 9.21   # 99% CL for 2 DOF
        }
        
    def setup_parameter_grid(self, parameters_of_interest, grid_resolution=50):
        """
        Set up parameter grids for profile likelihood calculation.
        
        Parameters:
        -----------
        parameters_of_interest : list
            List of parameter names to profile over
        grid_resolution : int
            Number of grid points per parameter dimension
        """
        print(f"📐 Setting up parameter grids for profile likelihood...")
        
        self.parameters_of_interest = parameters_of_interest
        self.grid_resolution = grid_resolution
        
        # Create 1D grids for each parameter
        for param_name in parameters_of_interest:
            if param_name in self.parameter_ranges:
                param_min, param_max = self.parameter_ranges[param_name]
                grid = np.linspace(param_min, param_max, grid_resolution)
                self.parameter_grids[param_name] = grid
                print(f"   ✓ {param_name}: {len(grid)} points from {param_min:.2f} to {param_max:.2f}")
            else:
                print(f"   ⚠ Warning: {param_name} not in predefined ranges")
        
        # Create 2D parameter meshes for contour plotting
        if len(parameters_of_interest) >= 2:
            param1, param2 = parameters_of_interest[0], parameters_of_interest[1]
            X, Y = np.meshgrid(self.parameter_grids[param1], self.parameter_grids[param2])
            self.parameter_meshes = {
                param1: X,
                param2: Y,
                'shape': X.shape
            }
        
        print(f"   ✓ Parameter grid setup complete")
    
    def define_nuisance_parameters(self, channel_nuisance_params):
        """
        Define nuisance parameters for each observational channel.
        
        Parameters:
        -----------
        channel_nuisance_params : dict
            Dictionary of nuisance parameters by channel
        """
        print(f"🔧 Defining nuisance parameters for marginalization...")
        
        self.nuisance_parameters = channel_nuisance_params
        
        for channel, params in channel_nuisance_params.items():
            print(f"   📊 {channel} channel:")
            for param_name, param_info in params.items():
                prior_mean = param_info.get('prior_mean', 0.0)
                prior_std = param_info.get('prior_std', 1.0)
                print(f"      • {param_name}: μ={prior_mean:.2f}, σ={prior_std:.2f}")
        
        print(f"   ✓ Nuisance parameter setup complete")
    
    def grb_likelihood(self, physics_params, nuisance_params, grb_data):
        """
        Compute GRB likelihood with nuisance parameters.
        
        Parameters:
        -----------
        physics_params : dict
            Physics parameters (LIV scale, coupling, etc.)
        nuisance_params : dict
            Nuisance parameters (calibration, systematics)
        grb_data : pd.DataFrame
            GRB observational data
            
        Returns:
        --------
        float : Log-likelihood value
        """
        log_like = 0.0
        
        # Extract physics parameters
        mu = 10**physics_params['log_mu']
        coupling = 10**physics_params['log_coupling']
        
        # Extract nuisance parameters
        energy_cal_factor = nuisance_params.get('energy_calibration', 1.0)
        timing_offset = nuisance_params.get('timing_offset', 0.0)
        intrinsic_scatter = nuisance_params.get('intrinsic_scatter', 1.0)
        
        # Mock GRB data for demonstration
        if grb_data is None or len(grb_data) == 0:
            # Generate mock GRB data
            n_grbs = 5
            redshifts = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
            energies = np.array([1.0, 5.0, 10.0, 20.0, 50.0])  # GeV
            observed_delays = np.array([0.1, 0.3, 0.5, 0.8, 1.2])  # seconds
            delay_errors = np.array([0.05, 0.1, 0.15, 0.2, 0.3])  # seconds
        else:
            redshifts = grb_data['redshift'].values
            energies = grb_data['energy_gev'].values
            observed_delays = grb_data['time_delay_s'].values
            delay_errors = grb_data['time_error_s'].values
        
        # Apply calibration factors
        calibrated_energies = energies * energy_cal_factor
        calibrated_delays = observed_delays - timing_offset
        
        # Compute theoretical LIV time delays
        for i in range(len(redshifts)):
            energy = calibrated_energies[i]
            redshift = redshifts[i]
            observed_delay = calibrated_delays[i]
            delay_error = delay_errors[i]
            
            # LIV time delay prediction: linear + quadratic terms
            predicted_delay = (coupling * energy / mu) * redshift + \
                            (coupling * (energy / mu)**2) * redshift
            
            # Include intrinsic scatter
            total_error = np.sqrt(delay_error**2 + intrinsic_scatter**2)
            
            # Gaussian likelihood
            chi2_contrib = ((observed_delay - predicted_delay) / total_error)**2
            log_like -= 0.5 * chi2_contrib
        
        # Add nuisance parameter priors
        log_like += self._evaluate_nuisance_priors(nuisance_params, 'grb')
        
        return log_like
    
    def uhecr_likelihood(self, physics_params, nuisance_params, uhecr_data):
        """
        Compute UHECR likelihood with nuisance parameters.
        
        Parameters:
        -----------
        physics_params : dict
            Physics parameters
        nuisance_params : dict
            Nuisance parameters
        uhecr_data : pd.DataFrame
            UHECR observational data
            
        Returns:
        --------
        float : Log-likelihood value
        """
        log_like = 0.0
        
        # Extract physics parameters
        mu = 10**physics_params['log_mu']
        coupling = 10**physics_params['log_coupling']
        
        # Extract nuisance parameters
        energy_scale = nuisance_params.get('energy_scale', 1.0)
        flux_normalization = nuisance_params.get('flux_normalization', 1.0)
        composition_factor = nuisance_params.get('composition_factor', 1.0)
        
        # Mock UHECR data for demonstration
        if uhecr_data is None or len(uhecr_data) == 0:
            energies = np.logspace(18, 20.5, 10)  # eV
            observed_fluxes = 1e-8 * (energies / 1e19)**(-2.7)  # Standard spectrum
            flux_errors = 0.2 * observed_fluxes  # 20% errors
        else:
            energies = uhecr_data['energy_ev'].values
            observed_fluxes = uhecr_data['flux'].values
            flux_errors = uhecr_data['flux_error'].values
        
        # Apply calibration factors
        calibrated_energies = energies * energy_scale
        calibrated_fluxes = observed_fluxes * flux_normalization
        
        # Compute theoretical UHECR spectrum with LIV modifications
        for i in range(len(energies)):
            energy = calibrated_energies[i]
            observed_flux = calibrated_fluxes[i]
            flux_error = flux_errors[i]
            
            # Standard GZK-modified spectrum
            base_flux = 1e-8 * (energy / 1e19)**(-2.7)
            
            # LIV modifications
            if energy > mu:
                liv_suppression = np.exp(-coupling * (energy / mu)**2)
                predicted_flux = base_flux * liv_suppression * composition_factor
            else:
                predicted_flux = base_flux * composition_factor
            
            # Log-normal likelihood for positive fluxes
            if predicted_flux > 0 and observed_flux > 0:
                log_ratio = np.log(observed_flux / predicted_flux)
                sigma_log = flux_error / observed_flux
                log_like -= 0.5 * (log_ratio / sigma_log)**2
        
        # Add nuisance parameter priors
        log_like += self._evaluate_nuisance_priors(nuisance_params, 'uhecr')
        
        return log_like
    
    def vacuum_likelihood(self, physics_params, nuisance_params, vacuum_data):
        """
        Compute vacuum instability likelihood with nuisance parameters.
        
        Parameters:
        -----------
        physics_params : dict
            Physics parameters
        nuisance_params : dict
            Nuisance parameters
        vacuum_data : pd.DataFrame
            Vacuum instability data
            
        Returns:
        --------
        float : Log-likelihood value
        """
        log_like = 0.0
        
        # Extract physics parameters
        mu = 10**physics_params['log_mu']
        coupling = 10**physics_params['log_coupling']
        
        # Extract nuisance parameters
        field_calibration = nuisance_params.get('field_calibration', 1.0)
        theory_uncertainty = nuisance_params.get('theory_uncertainty', 1.0)
        
        # Mock vacuum data for demonstration
        field_strengths = np.logspace(15, 17, 6)  # V/m
        schwinger_field = 1.32e16  # V/m
        
        # Compute theoretical vacuum pair production rates
        for field in field_strengths:
            calibrated_field = field * field_calibration
            
            # Schwinger formula with LIV modifications
            standard_rate = np.exp(-np.pi * (schwinger_field / calibrated_field))
            
            # LIV enhancement factor
            if calibrated_field > mu / 1e10:  # Field-dependent LIV scale
                liv_enhancement = 1 + coupling * (calibrated_field / mu * 1e10)**2
                predicted_rate = standard_rate * liv_enhancement * theory_uncertainty
            else:
                predicted_rate = standard_rate * theory_uncertainty
            
            # Mock observed rate (for demonstration)
            observed_rate = predicted_rate * (1 + 0.1 * np.random.randn())
            rate_error = 0.2 * predicted_rate
            
            # Gaussian likelihood in log space
            if predicted_rate > 0 and observed_rate > 0:
                log_ratio = np.log(observed_rate / predicted_rate)
                sigma_log = rate_error / predicted_rate
                log_like -= 0.5 * (log_ratio / sigma_log)**2
        
        # Add nuisance parameter priors
        log_like += self._evaluate_nuisance_priors(nuisance_params, 'vacuum')
        
        return log_like
    
    def hidden_sector_likelihood(self, physics_params, nuisance_params, hidden_data):
        """
        Compute hidden sector likelihood with nuisance parameters.
        
        Parameters:
        -----------
        physics_params : dict
            Physics parameters
        nuisance_params : dict
            Nuisance parameters
        hidden_data : pd.DataFrame
            Hidden sector search data
            
        Returns:
        --------
        float : Log-likelihood value
        """
        log_like = 0.0
        
        # Extract physics parameters
        mu = 10**physics_params['log_mu']
        coupling = 10**physics_params['log_coupling']
        
        # Extract nuisance parameters
        sensitivity_factor = nuisance_params.get('sensitivity_factor', 1.0)
        background_level = nuisance_params.get('background_level', 1.0)
        
        # Mock hidden sector data
        mass_range = np.logspace(-6, -3, 8)  # eV
        
        for mass in mass_range:
            # Theoretical dark photon mixing rate
            if mass < mu / 1e15:  # Mass-dependent LIV coupling
                mixing_rate = coupling * (mass / mu * 1e15)**0.5
                predicted_signal = mixing_rate * sensitivity_factor
            else:
                predicted_signal = 0.0
            
            # Mock exclusion limit
            observed_limit = 1e-10 * (mass / 1e-5)**(-0.5) * background_level
            limit_uncertainty = 0.3 * observed_limit
            
            # Upper limit likelihood
            if predicted_signal < observed_limit:
                log_like += 0.0  # Within allowed region
            else:
                # Penalize predictions above observed limits
                excess = (predicted_signal - observed_limit) / limit_uncertainty
                log_like -= 0.5 * excess**2
        
        # Add nuisance parameter priors
        log_like += self._evaluate_nuisance_priors(nuisance_params, 'hidden_sector')
        
        return log_like
    
    def _evaluate_nuisance_priors(self, nuisance_params, channel):
        """Evaluate prior probabilities for nuisance parameters."""
        log_prior = 0.0
        
        if channel in self.nuisance_parameters:
            channel_priors = self.nuisance_parameters[channel]
            
            for param_name, prior_info in channel_priors.items():
                if param_name in nuisance_params:
                    param_value = nuisance_params[param_name]
                    prior_mean = prior_info.get('prior_mean', 0.0)
                    prior_std = prior_info.get('prior_std', 1.0)
                    
                    # Gaussian prior
                    log_prior -= 0.5 * ((param_value - prior_mean) / prior_std)**2
        
        return log_prior
    
    def combined_likelihood(self, physics_params, all_nuisance_params, all_data):
        """
        Compute combined likelihood across all channels.
        
        Parameters:
        -----------
        physics_params : dict
            Physics parameters of interest
        all_nuisance_params : dict
            Nuisance parameters for all channels
        all_data : dict
            Observational data for all channels
            
        Returns:
        --------
        float : Combined log-likelihood
        """
        total_log_like = 0.0
        
        # GRB contribution
        if 'grb' in all_data and 'grb' in all_nuisance_params:
            grb_log_like = self.grb_likelihood(
                physics_params, all_nuisance_params['grb'], all_data['grb'])
            total_log_like += grb_log_like
        
        # UHECR contribution
        if 'uhecr' in all_data and 'uhecr' in all_nuisance_params:
            uhecr_log_like = self.uhecr_likelihood(
                physics_params, all_nuisance_params['uhecr'], all_data['uhecr'])
            total_log_like += uhecr_log_like
        
        # Vacuum contribution
        if 'vacuum' in all_data and 'vacuum' in all_nuisance_params:
            vacuum_log_like = self.vacuum_likelihood(
                physics_params, all_nuisance_params['vacuum'], all_data['vacuum'])
            total_log_like += vacuum_log_like
        
        # Hidden sector contribution
        if 'hidden_sector' in all_data and 'hidden_sector' in all_nuisance_params:
            hidden_log_like = self.hidden_sector_likelihood(
                physics_params, all_nuisance_params['hidden_sector'], all_data['hidden_sector'])
            total_log_like += hidden_log_like
        
        return total_log_like
    
    def compute_profile_likelihood(self, physics_param_values, all_data):
        """
        Compute profile likelihood by maximizing over nuisance parameters.
        
        Parameters:
        -----------
        physics_param_values : dict
            Fixed values of physics parameters
        all_data : dict
            Observational data for all channels
            
        Returns:
        --------
        float : Profile log-likelihood value
        """
        
        def objective(nuisance_params_flat):
            """Objective function for nuisance parameter optimization."""
            # Unpack flat array into structured nuisance parameters
            all_nuisance_params = self._unpack_nuisance_parameters(nuisance_params_flat)
            
            # Compute negative log-likelihood (for minimization)
            log_like = self.combined_likelihood(physics_param_values, all_nuisance_params, all_data)
            return -log_like
        
        # Initial guess for nuisance parameters (at prior means)
        initial_nuisance = self._get_initial_nuisance_parameters()
        
        # Optimize over nuisance parameters
        try:
            result = optimize.minimize(objective, initial_nuisance, method='Powell',
                                     options={'maxiter': 1000, 'disp': False})
            profile_log_like = -result.fun
        except:
            # If optimization fails, return very negative likelihood
            profile_log_like = -1e6
        
        return profile_log_like
    
    def _get_initial_nuisance_parameters(self):
        """Get initial values for nuisance parameters (prior means)."""
        initial_params = []
        
        for channel in ['grb', 'uhecr', 'vacuum', 'hidden_sector']:
            if channel in self.nuisance_parameters:
                for param_name, prior_info in self.nuisance_parameters[channel].items():
                    initial_params.append(prior_info.get('prior_mean', 0.0))
        
        return np.array(initial_params)
    
    def _unpack_nuisance_parameters(self, params_flat):
        """Unpack flat parameter array into structured dictionary."""
        all_nuisance_params = {}
        idx = 0
        
        for channel in ['grb', 'uhecr', 'vacuum', 'hidden_sector']:
            if channel in self.nuisance_parameters:
                channel_params = {}
                for param_name in self.nuisance_parameters[channel].keys():
                    channel_params[param_name] = params_flat[idx]
                    idx += 1
                all_nuisance_params[channel] = channel_params
        
        return all_nuisance_params
    
    def generate_profile_likelihood_contours(self, all_data):
        """
        Generate 2D profile likelihood contours.
        
        Parameters:
        -----------
        all_data : dict
            Observational data for all channels
            
        Returns:
        --------
        dict : Profile likelihood results and contours
        """
        print(f"🎯 Computing 2D profile likelihood contours...")
        
        if len(self.parameters_of_interest) < 2:
            raise ValueError("Need at least 2 parameters for 2D contours")
        
        param1, param2 = self.parameters_of_interest[0], self.parameters_of_interest[1]
        grid_shape = self.parameter_meshes['shape']
        
        # Initialize profile likelihood grid
        profile_likelihood_grid = np.zeros(grid_shape)
        
        # Compute profile likelihood at each grid point
        total_points = grid_shape[0] * grid_shape[1]
        point_count = 0
        
        print(f"   Computing profile likelihood at {total_points} grid points...")
        
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                # Current parameter values
                physics_params = {
                    param1: self.parameter_meshes[param1][i, j],
                    param2: self.parameter_meshes[param2][i, j]
                }
                
                # Add default values for other physics parameters
                for param_name, (default_min, default_max) in self.parameter_ranges.items():
                    if param_name not in physics_params:
                        physics_params[param_name] = (default_min + default_max) / 2
                
                # Compute profile likelihood
                profile_log_like = self.compute_profile_likelihood(physics_params, all_data)
                profile_likelihood_grid[i, j] = profile_log_like
                
                point_count += 1
                if point_count % (total_points // 10) == 0:
                    progress = point_count / total_points * 100
                    print(f"   Progress: {progress:.1f}% complete")
        
        # Convert to likelihood (not log-likelihood) for contour plotting
        max_log_like = np.max(profile_likelihood_grid)
        likelihood_grid = np.exp(profile_likelihood_grid - max_log_like)
        
        # Convert to chi-square: -2 * log(L/L_max)
        chi2_grid = -2 * (profile_likelihood_grid - max_log_like)
        
        # Store results
        self.profile_likelihoods[f"{param1}_{param2}"] = {
            'param1_name': param1,
            'param2_name': param2,
            'param1_grid': self.parameter_meshes[param1],
            'param2_grid': self.parameter_meshes[param2],
            'log_likelihood_grid': profile_likelihood_grid,
            'likelihood_grid': likelihood_grid,
            'chi2_grid': chi2_grid,
            'max_log_likelihood': max_log_like
        }
        
        print(f"   ✓ Profile likelihood computation complete")
        print(f"   📊 Maximum log-likelihood: {max_log_like:.2f}")
        
        return self.profile_likelihoods[f"{param1}_{param2}"]
    
    def identify_compatibility_regions(self, profile_results, models_to_compare):
        """
        Identify regions where different models are jointly compatible.
        
        Parameters:
        -----------
        profile_results : dict
            Profile likelihood results
        models_to_compare : list
            List of model names to compare
            
        Returns:
        --------
        dict : Compatibility analysis results
        """
        print(f"🔍 Identifying compatibility regions...")
        
        compatibility_results = {}
        
        # Extract grid information
        param1_grid = profile_results['param1_grid']
        param2_grid = profile_results['param2_grid']
        chi2_grid = profile_results['chi2_grid']
        
        # Find confidence regions for each confidence level
        for conf_level in self.confidence_levels:
            chi2_threshold = self.chi2_thresholds[conf_level]
            
            # Points within confidence region
            within_region = chi2_grid <= chi2_threshold
            
            # Extract contour coordinates
            if np.any(within_region):
                # Find contour using matplotlib
                fig, ax = plt.subplots(figsize=(1, 1))
                contour = ax.contour(param1_grid, param2_grid, chi2_grid, 
                                   levels=[chi2_threshold])
                plt.close(fig)
                  # Store contour data
                contour_data = []
                try:
                    # Try newer matplotlib interface
                    for collection in contour.collections:
                        for path in collection.get_paths():
                            vertices = path.vertices
                            contour_data.append(vertices)
                except AttributeError:
                    # Fallback for different matplotlib versions
                    if hasattr(contour, 'allsegs'):
                        for level_segs in contour.allsegs:
                            for seg in level_segs:
                                contour_data.append(seg)
                
                compatibility_results[f"{conf_level:.0%}_region"] = {
                    'chi2_threshold': chi2_threshold,
                    'grid_mask': within_region,
                    'contour_paths': contour_data,
                    'area_fraction': np.sum(within_region) / within_region.size
                }
                
                print(f"   ✓ {conf_level:.0%} confidence region covers {np.sum(within_region) / within_region.size:.1%} of parameter space")
        
        # Model comparison analysis
        model_compatibility = {}
        for model_name in models_to_compare:
            # Define approximate model prediction region (would be computed from model)
            # For demonstration, create mock model regions
            if model_name == 'string_theory':
                model_center = (17.5, -7.5)  # (log_mu, log_coupling)
                model_sigma = (0.5, 0.5)
            elif model_name == 'rainbow_gravity':
                model_center = (18.0, -7.0)
                model_sigma = (0.3, 0.4)
            elif model_name == 'polymer_quantum':
                model_center = (18.5, -6.5)
                model_sigma = (0.4, 0.6)
            else:
                model_center = (17.8, -7.2)
                model_sigma = (0.6, 0.3)
            
            # Compute model likelihood on grid
            model_likelihood = np.exp(-0.5 * (
                ((param1_grid - model_center[0]) / model_sigma[0])**2 +
                ((param2_grid - model_center[1]) / model_sigma[1])**2
            ))
            
            # Find overlap with confidence regions
            overlaps = {}
            for conf_level in self.confidence_levels:
                region_key = f"{conf_level:.0%}_region"
                if region_key in compatibility_results:
                    region_mask = compatibility_results[region_key]['grid_mask']
                    overlap_likelihood = model_likelihood * region_mask
                    total_overlap = np.sum(overlap_likelihood)
                    overlaps[conf_level] = {
                        'total_overlap': total_overlap,
                        'overlap_fraction': total_overlap / np.sum(model_likelihood),
                        'p_value': self._compute_compatibility_p_value(
                            chi2_grid, model_likelihood, conf_level)
                    }
            
            model_compatibility[model_name] = {
                'model_center': model_center,
                'model_sigma': model_sigma,
                'model_likelihood_grid': model_likelihood,
                'confidence_overlaps': overlaps
            }
        
        compatibility_results['model_compatibility'] = model_compatibility
        
        print(f"   ✓ Compatibility analysis complete for {len(models_to_compare)} models")
        
        return compatibility_results
    
    def _compute_compatibility_p_value(self, chi2_grid, model_likelihood, conf_level):
        """Compute p-value for model compatibility."""
        # Weighted chi-square statistic
        weighted_chi2 = np.sum(chi2_grid * model_likelihood) / np.sum(model_likelihood)
        
        # Convert to p-value (2 DOF)
        p_value = 1 - stats.chi2.cdf(weighted_chi2, df=2)
        
        return p_value
    
    def plot_profile_likelihood_contours(self, profile_results, compatibility_results, 
                                       save_path='results/profile_likelihood_contours.png'):
        """
        Generate comprehensive profile likelihood contour plots.
        
        Parameters:
        -----------
        profile_results : dict
            Profile likelihood computation results
        compatibility_results : dict
            Model compatibility analysis results
        save_path : str
            Path to save the plot
        """
        print(f"📈 Generating profile likelihood contour plots...")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # Extract data
        param1_name = profile_results['param1_name']
        param2_name = profile_results['param2_name']
        param1_grid = profile_results['param1_grid']
        param2_grid = profile_results['param2_grid']
        chi2_grid = profile_results['chi2_grid']
        likelihood_grid = profile_results['likelihood_grid']
        
        # 1. Main profile likelihood contours
        ax1 = plt.subplot(2, 3, 1)
          # Plot likelihood surface
        im1 = ax1.contourf(param1_grid, param2_grid, likelihood_grid, 
                          levels=50, cmap='Blues', alpha=0.8)
        
        # Plot confidence contours
        colors = ['red', 'orange', 'yellow']
        for i, conf_level in enumerate(self.confidence_levels):
            chi2_threshold = self.chi2_thresholds[conf_level]
            contour = ax1.contour(param1_grid, param2_grid, chi2_grid,
                                levels=[chi2_threshold], colors=[colors[i]], 
                                linewidths=2)
            # Skip clabel to avoid formatting issues
        
        ax1.set_xlabel(param1_name.replace('_', ' ').title())
        ax1.set_ylabel(param2_name.replace('_', ' ').title())
        ax1.set_title('Profile Likelihood Contours')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Likelihood')
        
        # 2. Chi-square surface
        ax2 = plt.subplot(2, 3, 2)
        
        # Mask very large chi-square values for better visualization
        chi2_masked = np.where(chi2_grid > 20, 20, chi2_grid)
        
        im2 = ax2.contourf(param1_grid, param2_grid, chi2_masked, 
                          levels=50, cmap='hot_r')
        
        # Add confidence contours
        for i, conf_level in enumerate(self.confidence_levels):
            chi2_threshold = self.chi2_thresholds[conf_level]
            ax2.contour(param1_grid, param2_grid, chi2_grid,
                       levels=[chi2_threshold], colors=['white'], 
                       linewidths=2, linestyles=['--'])
        
        ax2.set_xlabel(param1_name.replace('_', ' ').title())
        ax2.set_ylabel(param2_name.replace('_', ' ').title())
        ax2.set_title('Chi-Square Surface')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Δχ²')
        
        # 3. Model comparison overlays
        ax3 = plt.subplot(2, 3, 3)
        
        # Plot confidence regions as filled contours
        for i, conf_level in enumerate(self.confidence_levels):
            chi2_threshold = self.chi2_thresholds[conf_level]
            region_mask = chi2_grid <= chi2_threshold
            ax3.contourf(param1_grid, param2_grid, region_mask.astype(float),
                        levels=[0.5, 1.5], colors=[colors[i]], alpha=0.3)
        
        # Overlay model predictions
        if 'model_compatibility' in compatibility_results:
            model_colors = ['blue', 'green', 'purple', 'brown']
            for i, (model_name, model_data) in enumerate(compatibility_results['model_compatibility'].items()):
                model_center = model_data['model_center']
                model_sigma = model_data['model_sigma']
                
                # Plot model center
                ax3.scatter(model_center[0], model_center[1], 
                           color=model_colors[i % len(model_colors)], 
                           s=100, marker='*', label=model_name.replace('_', ' ').title())
                
                # Plot model uncertainty ellipse (1σ)
                ellipse = patches.Ellipse(model_center, 2*model_sigma[0], 2*model_sigma[1],
                                        facecolor=model_colors[i % len(model_colors)], 
                                        alpha=0.3, edgecolor='black')
                ax3.add_patch(ellipse)
        
        ax3.set_xlabel(param1_name.replace('_', ' ').title())
        ax3.set_ylabel(param2_name.replace('_', ' ').title())
        ax3.set_title('Model Compatibility Regions')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 1D profile likelihood slices
        ax4 = plt.subplot(2, 3, 4)
        
        # Profile over param1 (marginalize over param2)
        param1_values = param1_grid[0, :]
        param1_profile = np.max(likelihood_grid, axis=0)
        ax4.plot(param1_values, param1_profile, 'b-', linewidth=2, label=f'{param1_name} profile')
        ax4.set_xlabel(param1_name.replace('_', ' ').title())
        ax4.set_ylabel('Profile Likelihood')
        ax4.set_title('1D Profile Likelihood')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. 1D profile likelihood slices (param2)
        ax5 = plt.subplot(2, 3, 5)
        
        # Profile over param2 (marginalize over param1)
        param2_values = param2_grid[:, 0]
        param2_profile = np.max(likelihood_grid, axis=1)
        ax5.plot(param2_values, param2_profile, 'r-', linewidth=2, label=f'{param2_name} profile')
        ax5.set_xlabel(param2_name.replace('_', ' ').title())
        ax5.set_ylabel('Profile Likelihood')
        ax5.set_title('1D Profile Likelihood')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Compatibility summary
        ax6 = plt.subplot(2, 3, 6)
        
        # Create compatibility summary table
        if 'model_compatibility' in compatibility_results:
            models = list(compatibility_results['model_compatibility'].keys())
            conf_levels = [f"{cl:.0%}" for cl in self.confidence_levels]
            
            # Create p-value matrix
            p_values = np.zeros((len(models), len(self.confidence_levels)))
            for i, model_name in enumerate(models):
                model_data = compatibility_results['model_compatibility'][model_name]
                for j, conf_level in enumerate(self.confidence_levels):
                    if conf_level in model_data['confidence_overlaps']:
                        p_values[i, j] = model_data['confidence_overlaps'][conf_level]['p_value']
            
            # Plot as heatmap
            im6 = ax6.imshow(p_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax6.set_xticks(range(len(conf_levels)))
            ax6.set_xticklabels(conf_levels)
            ax6.set_yticks(range(len(models)))
            ax6.set_yticklabels([model.replace('_', ' ').title() for model in models])
            ax6.set_xlabel('Confidence Level')
            ax6.set_ylabel('Model')
            ax6.set_title('Model Compatibility p-values')
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(conf_levels)):
                    text = ax6.text(j, i, f'{p_values[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im6, ax=ax6, label='p-value')
        else:
            ax6.text(0.5, 0.5, 'Model compatibility\nanalysis not available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Model Compatibility')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        print(f"   ✓ Profile likelihood plots saved to {save_path}")
    
    def save_profile_likelihood_results(self, profile_results, compatibility_results,
                                      save_dir='results'):
        """Save profile likelihood analysis results to files."""
        print(f"💾 Saving profile likelihood analysis results...")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save profile likelihood grid data
        param1_name = profile_results['param1_name']
        param2_name = profile_results['param2_name']
        
        # Create flattened data for CSV export
        param1_flat = profile_results['param1_grid'].flatten()
        param2_flat = profile_results['param2_grid'].flatten()
        log_like_flat = profile_results['log_likelihood_grid'].flatten()
        chi2_flat = profile_results['chi2_grid'].flatten()
        
        profile_df = pd.DataFrame({
            param1_name: param1_flat,
            param2_name: param2_flat,
            'log_likelihood': log_like_flat,
            'chi2': chi2_flat,
            'likelihood': np.exp(log_like_flat - np.max(log_like_flat))
        })
        
        profile_df.to_csv(f'{save_dir}/profile_likelihood_grid.csv', index=False)
        
        # Save compatibility analysis results
        if 'model_compatibility' in compatibility_results:
            compatibility_data = []
            
            for model_name, model_data in compatibility_results['model_compatibility'].items():
                for conf_level, overlap_data in model_data['confidence_overlaps'].items():
                    compatibility_data.append({
                        'model': model_name,
                        'confidence_level': conf_level,
                        'overlap_fraction': overlap_data['overlap_fraction'],
                        'p_value': overlap_data['p_value']
                    })
            
            compatibility_df = pd.DataFrame(compatibility_data)
            compatibility_df.to_csv(f'{save_dir}/model_compatibility_analysis.csv', index=False)
        
        # Save confidence region data
        for conf_level in self.confidence_levels:
            region_key = f"{conf_level:.0%}_region"
            if region_key in compatibility_results:
                region_data = compatibility_results[region_key]
                
                # Save contour paths
                contour_data = []
                for i, path in enumerate(region_data['contour_paths']):
                    for j, point in enumerate(path):
                        contour_data.append({
                            'confidence_level': conf_level,
                            'contour_id': i,
                            'point_id': j,
                            param1_name: point[0],
                            param2_name: point[1]
                        })
                
                if contour_data:
                    contour_df = pd.DataFrame(contour_data)
                    contour_df.to_csv(f'{save_dir}/confidence_contour_{conf_level:.0%}.csv', index=False)
        
        print(f"   ✓ Results saved to {save_dir}/")

def main():
    """Demonstration of profile likelihood analysis."""
    print("🎯 PROFILE LIKELIHOOD ANALYSIS DEMONSTRATION")
    print("=" * 55)
    
    # Initialize profile likelihood analysis
    pla = ProfileLikelihoodAnalysis(confidence_levels=[0.68, 0.95, 0.99])
    
    # Set up parameter grid
    parameters_of_interest = ['log_mu', 'log_coupling']
    pla.setup_parameter_grid(parameters_of_interest, grid_resolution=25)  # Reduced for demo
    
    # Define nuisance parameters
    nuisance_params = {
        'grb': {
            'energy_calibration': {'prior_mean': 1.0, 'prior_std': 0.1},
            'timing_offset': {'prior_mean': 0.0, 'prior_std': 0.1},
            'intrinsic_scatter': {'prior_mean': 1.0, 'prior_std': 0.2}
        },
        'uhecr': {
            'energy_scale': {'prior_mean': 1.0, 'prior_std': 0.15},
            'flux_normalization': {'prior_mean': 1.0, 'prior_std': 0.2},
            'composition_factor': {'prior_mean': 1.0, 'prior_std': 0.25}
        },
        'vacuum': {
            'field_calibration': {'prior_mean': 1.0, 'prior_std': 0.05},
            'theory_uncertainty': {'prior_mean': 1.0, 'prior_std': 0.15}
        },
        'hidden_sector': {
            'sensitivity_factor': {'prior_mean': 1.0, 'prior_std': 0.2},
            'background_level': {'prior_mean': 1.0, 'prior_std': 0.15}
        }
    }
    
    pla.define_nuisance_parameters(nuisance_params)
    
    # Mock observational data
    all_data = {
        'grb': None,        # Will use mock data
        'uhecr': None,      # Will use mock data
        'vacuum': None,     # Will use mock data
        'hidden_sector': None  # Will use mock data
    }
    
    # Compute profile likelihood contours
    profile_results = pla.generate_profile_likelihood_contours(all_data)
    
    # Identify compatibility regions
    models_to_compare = ['string_theory', 'rainbow_gravity', 'polymer_quantum']
    compatibility_results = pla.identify_compatibility_regions(profile_results, models_to_compare)
    
    # Generate plots
    pla.plot_profile_likelihood_contours(profile_results, compatibility_results)
    
    # Save results
    pla.save_profile_likelihood_results(profile_results, compatibility_results)
    
    print(f"\n🎉 Profile likelihood analysis demonstration complete!")
    print(f"📊 Check results/ directory for contour plots and data files")

if __name__ == "__main__":
    main()
