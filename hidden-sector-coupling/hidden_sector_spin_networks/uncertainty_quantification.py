"""
Uncertainty Quantification Framework for SU(2) Spin Network Portal

This module implements a comprehensive three-stage UQ workflow:
1. Quantify input uncertainties with probability distributions
2. Propagate uncertainties via Monte Carlo and surrogate modeling
3. Analyze results with sensitivity analysis and robust optimization

Author: Quantum Geometry Hidden Sector Framework
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import warnings
import time
from functools import partial
import pickle

# Core modules
try:
    from su2_recoupling_module import SpinNetworkPortal, SpinNetworkConfig
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    warnings.warn("Core SU(2) module not available")

# UQ-specific imports
try:
    import uncertainties as unc
    from uncertainties import ufloat
    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False
    warnings.warn("Install uncertainties: pip install uncertainties")

try:
    from SALib.sample import saltelli, latin
    from SALib.analyze import sobol, fast
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False
    warnings.warn("Install SALib: pip install SALib")

try:
    import chaospy as cp
    HAS_CHAOSPY = True
except ImportError:
    HAS_CHAOSPY = False
    warnings.warn("Install chaospy: pip install chaospy")

try:
    from pyDOE2 import lhs
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False
    warnings.warn("Install pyDOE2: pip install pyDOE2")

# Optional advanced sampling
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

@dataclass
class UQConfig:
    """Configuration for uncertainty quantification analysis."""
    n_samples: int = 10000
    sampling_method: str = 'monte_carlo'  # 'monte_carlo', 'latin_hypercube', 'sobol'
    surrogate_type: str = 'polynomial_chaos'  # 'polynomial_chaos', 'gaussian_process'
    pce_degree: int = 3
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    random_seed: int = 42
    parallel: bool = False
    n_jobs: int = 4

@dataclass 
class ParameterDistribution:
    """Represents uncertainty distribution for a parameter."""
    name: str
    distribution: stats.rv_continuous
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from the distribution."""
        samples = self.distribution.rvs(size=n_samples)
        if self.bounds:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])
        return samples
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function."""
        return self.distribution.pdf(x)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return self.distribution.cdf(x)

@dataclass
class UQResults:
    """Container for UQ analysis results."""
    parameter_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    output_samples: Dict[str, np.ndarray] = field(default_factory=dict)
    statistical_summary: Dict[str, Dict] = field(default_factory=dict)
    sensitivity_indices: Dict[str, Dict] = field(default_factory=dict)
    robust_optima: Dict[str, Dict] = field(default_factory=dict)
    surrogate_models: Dict[str, Any] = field(default_factory=dict)
    computational_time: float = 0.0
    config: Optional[UQConfig] = None

class SpinNetworkUQFramework:
    """Main class for uncertainty quantification of spin network portals."""
    
    def __init__(self, base_config: SpinNetworkConfig, uq_config: UQConfig):
        """
        Initialize UQ framework.
        
        Parameters:
        -----------
        base_config : SpinNetworkConfig
            Base configuration for spin network portal
        uq_config : UQConfig
            Configuration for UQ analysis
        """
        self.base_config = base_config
        self.uq_config = uq_config
        self.parameter_distributions = {}
        self.results = None
        
        # Set random seed for reproducibility
        np.random.seed(uq_config.random_seed)
        
        # Initialize default parameter distributions
        self._setup_default_distributions()
    
    def _setup_default_distributions(self):
        """Setup default uncertainty distributions for portal parameters."""
        
        # Base coupling: log-uniform prior (broad uncertainty)
        self.add_parameter_distribution(
            'base_coupling',
            stats.loguniform(1e-8, 1e-3),
            description="Hidden sector coupling strength"
        )
        
        # Geometric suppression: normal around default with uncertainty
        self.add_parameter_distribution(
            'geometric_suppression', 
            stats.truncnorm(-2, 2, loc=0.1, scale=0.03),
            bounds=(0.01, 0.5),
            description="Angular momentum geometric suppression"
        )
        
        # Portal correlation length: log-normal 
        self.add_parameter_distribution(
            'portal_correlation_length',
            stats.lognorm(s=0.5, scale=1.5),
            bounds=(0.5, 5.0),
            description="Spatial coherence length"
        )
        
        # Network connectivity: beta distribution
        self.add_parameter_distribution(
            'connectivity',
            stats.beta(2, 3),  # Favors lower connectivity
            bounds=(0.1, 0.8),
            description="Network edge connection probability"
        )
        
        # Max angular momentum: discrete uniform
        self.add_parameter_distribution(
            'max_angular_momentum',
            stats.randint(2, 6),  # j = 2, 3, 4, 5
            description="Maximum angular momentum cutoff"
        )
        
        # Network size: discrete with preference for moderate sizes
        self.add_parameter_distribution(
            'network_size',
            stats.randint(8, 16),  # 8-15 nodes
            description="Number of network vertices"
        )
    
    def add_parameter_distribution(self, name: str, distribution: stats.rv_continuous,
                                 bounds: Optional[Tuple[float, float]] = None,
                                 description: str = ""):
        """Add or update parameter uncertainty distribution."""
        self.parameter_distributions[name] = ParameterDistribution(
            name=name,
            distribution=distribution,
            bounds=bounds,
            description=description
        )
    
    def add_experimental_uncertainty(self, parameter_name: str, 
                                   nominal_value: float, uncertainty: float,
                                   uncertainty_type: str = 'gaussian'):
        """
        Add experimental measurement uncertainty.
        
        Parameters:
        -----------
        parameter_name : str
            Name of the parameter
        nominal_value : float
            Measured central value
        uncertainty : float
            Measurement uncertainty (1σ)        uncertainty_type : str
            Type of uncertainty ('gaussian', 'log_normal', 'uniform')
        """
        if uncertainty_type == 'gaussian':
            distribution = stats.norm(loc=nominal_value, scale=uncertainty)
        elif uncertainty_type == 'log_normal':
            # For log-normal with relative uncertainty
            # If X ~ lognormal(μ, σ), then ln(X) ~ normal(μ, σ)
            # For relative uncertainty r on nominal value x0:
            # σ ≈ ln(1 + r) for small r, or more precisely:
            sigma = np.sqrt(np.log(1 + (uncertainty/nominal_value)**2))
            mu = np.log(nominal_value) - 0.5 * sigma**2  # Adjust for median
            distribution = stats.lognorm(s=sigma, scale=np.exp(mu))
        elif uncertainty_type == 'uniform':
            # Uniform over ±uncertainty around nominal
            distribution = stats.uniform(
                loc=nominal_value - uncertainty,
                scale=2 * uncertainty
            )
        else:
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
        
        self.add_parameter_distribution(
            parameter_name,
            distribution,
            description=f"Experimental measurement: {nominal_value} ± {uncertainty}"
        )
    
    def _determine_sampling_method(self) -> str:
        """Determine the actual sampling method to use, handling fallbacks."""
        requested_method = self.uq_config.sampling_method
        
        if requested_method == 'monte_carlo':
            return 'monte_carlo'
        elif requested_method == 'latin_hypercube':
            if HAS_PYDOE:
                return 'latin_hypercube'
            else:
                warnings.warn("pyDOE2 not available, falling back to Monte Carlo")
                return 'monte_carlo'
        elif requested_method == 'sobol':
            if HAS_SALIB:
                return 'sobol'
            elif HAS_PYDOE:
                warnings.warn("SALib not available, falling back to Latin Hypercube")
                return 'latin_hypercube'
            else:
                warnings.warn("SALib and pyDOE2 not available, falling back to Monte Carlo")
                return 'monte_carlo'
        else:
            warnings.warn(f"Unknown sampling method {requested_method}, using Monte Carlo")
            return 'monte_carlo'

    def generate_parameter_samples(self, n_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate parameter samples according to specified method."""
        
        if n_samples is None:
            n_samples = self.uq_config.n_samples
        
        param_names = list(self.parameter_distributions.keys())
        n_params = len(param_names)
          # Determine the actual sampling method to use (handle fallbacks without recursion)
        actual_method = self._determine_sampling_method()
        
        if actual_method == 'monte_carlo':
            # Simple Monte Carlo sampling
            samples = {}
            for name, param_dist in self.parameter_distributions.items():
                samples[name] = param_dist.sample(n_samples)
                
        elif actual_method == 'latin_hypercube':
            # Latin Hypercube Sampling for better space filling
            lhs_samples = lhs(n_params, samples=n_samples, random_state=self.uq_config.random_seed)
            
            samples = {}
            for i, (name, param_dist) in enumerate(self.parameter_distributions.items()):
                # Transform uniform [0,1] samples to parameter distribution
                uniform_samples = lhs_samples[:, i]
                samples[name] = param_dist.distribution.ppf(uniform_samples)
                
                # Apply bounds if specified
                if param_dist.bounds:
                    samples[name] = np.clip(samples[name], 
                                          param_dist.bounds[0], param_dist.bounds[1])
                    
        elif actual_method == 'sobol':
            # Sobol sequence sampling
            problem = {
                'num_vars': n_params,
                'names': param_names,
                'bounds': [[0, 1]] * n_params  # Unit hypercube
            }
            
            sobol_samples = saltelli.sample(problem, n_samples//2)
            
            samples = {}
            for i, (name, param_dist) in enumerate(self.parameter_distributions.items()):
                # Transform uniform [0,1] samples to parameter distribution
                uniform_samples = sobol_samples[:, i]
                samples[name] = param_dist.distribution.ppf(uniform_samples)
                
                # Apply bounds if specified
                if param_dist.bounds:
                    samples[name] = np.clip(samples[name], 
                                          param_dist.bounds[0], param_dist.bounds[1])
        else:
            raise ValueError(f"Unknown sampling method: {actual_method}")
        
        return samples
    
    def evaluate_model(self, parameter_samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Evaluate spin network portal for parameter samples.
        
        Returns:
        --------
        Dict with output quantities: transfer_rate, leakage_amplitude, etc.
        """
        if not HAS_CORE:
            raise ImportError("Core SU(2) module required for model evaluation")
        
        n_samples = len(next(iter(parameter_samples.values())))
        
        # Output arrays
        transfer_rates = np.zeros(n_samples)
        max_amplitudes = np.zeros(n_samples)
        characteristic_times = np.zeros(n_samples)
        network_efficiencies = np.zeros(n_samples)
        
        print(f"Evaluating {n_samples} parameter samples...")
        
        # Simple density of states for transfer rate calculation
        def density_of_states(E):
            return E**2 / 10
        
        for i in range(n_samples):
            if i % (n_samples // 10) == 0:
                print(f"  Progress: {i/n_samples*100:.0f}%")
            
            try:
                # Create configuration for this sample
                config = SpinNetworkConfig(
                    base_coupling=parameter_samples['base_coupling'][i],
                    geometric_suppression=parameter_samples['geometric_suppression'][i],
                    portal_correlation_length=parameter_samples['portal_correlation_length'][i],
                    max_angular_momentum=int(parameter_samples['max_angular_momentum'][i]),
                    network_size=int(parameter_samples['network_size'][i]),
                    connectivity=parameter_samples['connectivity'][i]
                )
                
                # Create portal
                portal = SpinNetworkPortal(config)
                
                # Compute transfer rate
                transfer_rate = portal.energy_transfer_rate((1.0, 10.0), density_of_states)
                transfer_rates[i] = transfer_rate
                
                # Compute representative leakage amplitude
                amplitude = portal.energy_leakage_amplitude(10.0, 5.0)
                max_amplitudes[i] = abs(amplitude)
                
                # Characteristic time (if transfer rate > 0)
                if transfer_rate > 0:
                    characteristic_times[i] = 1.0 / transfer_rate
                else:
                    characteristic_times[i] = np.inf
                
                # Network efficiency (effective couplings)
                sample_vertices = list(portal.network.nodes())[:3]
                avg_coupling = np.mean([portal.effective_coupling(v) for v in sample_vertices])
                network_efficiencies[i] = avg_coupling
                
            except Exception as e:
                # Handle numerical errors gracefully
                warnings.warn(f"Sample {i} failed: {e}")
                transfer_rates[i] = 0.0
                max_amplitudes[i] = 0.0
                characteristic_times[i] = np.inf
                network_efficiencies[i] = 0.0
        
        print("  Evaluation complete!")
        
        return {
            'transfer_rate': transfer_rates,
            'leakage_amplitude': max_amplitudes,
            'characteristic_time': characteristic_times,
            'network_efficiency': network_efficiencies
        }
    
    def evaluate_model_fast(self, parameter_samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fast model evaluation using simplified approximations for UQ purposes.
        
        This method uses analytical approximations instead of full network simulations
        to enable rapid uncertainty propagation.
        """
        n_samples = len(next(iter(parameter_samples.values())))
        
        # Output arrays
        transfer_rates = np.zeros(n_samples)
        max_amplitudes = np.zeros(n_samples)
        characteristic_times = np.zeros(n_samples)
        network_efficiencies = np.zeros(n_samples)
        
        print(f"Fast-evaluating {n_samples} parameter samples...")
        
        for i in range(n_samples):
            if i % max(1, n_samples // 10) == 0:
                print(f"  Progress: {i/n_samples*100:.0f}%")
            
            # Extract parameters
            coupling = parameter_samples['base_coupling'][i]
            geom_supp = parameter_samples['geometric_suppression'][i]
            corr_length = parameter_samples['portal_correlation_length'][i]
            connectivity = parameter_samples['connectivity'][i]
            max_j = parameter_samples['max_angular_momentum'][i]
            n_nodes = parameter_samples['network_size'][i]
            
            # Simplified analytical models for fast evaluation
            
            # Transfer rate: ∝ coupling² × connectivity × geometric_factor × size_scaling
            geometric_factor = np.exp(-abs(geom_supp))  # Exponential suppression
            size_scaling = np.sqrt(n_nodes)  # Network size enhancement
            connectivity_factor = connectivity * (1 + connectivity)  # Non-linear connectivity effect
            
            transfer_rate = (coupling**2 * geometric_factor * size_scaling * 
                           connectivity_factor * corr_length * max_j**2)
            transfer_rates[i] = transfer_rate
            
            # Leakage amplitude: based on coupling and correlation length
            amplitude = coupling * np.sqrt(corr_length) * geometric_factor
            max_amplitudes[i] = amplitude
            
            # Characteristic time
            if transfer_rate > 1e-20:
                characteristic_times[i] = 1.0 / transfer_rate
            else:
                characteristic_times[i] = 1e20  # Very large but finite
            
            # Network efficiency: effective coupling strength
            efficiency = coupling * connectivity * geometric_factor
            network_efficiencies[i] = efficiency
        
        print("  Fast evaluation complete!")
        
        return {
            'transfer_rate': transfer_rates,
            'leakage_amplitude': max_amplitudes,
            'characteristic_time': characteristic_times,
            'network_efficiency': network_efficiencies
        }
    
    def compute_statistical_summary(self, output_samples: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Compute comprehensive statistical summary of outputs."""
        
        summary = {}
        confidence_level = self.uq_config.confidence_level
        alpha = 1 - confidence_level
        
        for output_name, samples in output_samples.items():
            # Filter out invalid samples
            valid_samples = samples[np.isfinite(samples)]
            
            if len(valid_samples) == 0:
                summary[output_name] = {
                    'mean': np.nan, 'std': np.nan, 'median': np.nan,
                    'ci_lower': np.nan, 'ci_upper': np.nan,
                    'min': np.nan, 'max': np.nan, 'valid_fraction': 0.0
                }
                continue
            
            # Basic statistics
            mean_val = np.mean(valid_samples)
            std_val = np.std(valid_samples)
            median_val = np.median(valid_samples)
            
            # Confidence intervals
            ci_lower = np.percentile(valid_samples, 100 * alpha/2)
            ci_upper = np.percentile(valid_samples, 100 * (1 - alpha/2))
            
            # Range
            min_val = np.min(valid_samples)
            max_val = np.max(valid_samples)
            
            # Validity fraction
            valid_fraction = len(valid_samples) / len(samples)
            
            summary[output_name] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'min': min_val,
                'max': max_val,
                'valid_fraction': valid_fraction,
                'cv': std_val / abs(mean_val) if mean_val != 0 else np.inf  # Coefficient of variation
            }
        
        return summary
    
    def sobol_sensitivity_analysis(self, parameter_samples: Dict[str, np.ndarray],
                                 output_samples: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Perform Sobol global sensitivity analysis."""
        
        if not HAS_SALIB:
            warnings.warn("SALib not available, skipping sensitivity analysis")
            return {}
        
        param_names = list(parameter_samples.keys())
        n_params = len(param_names)
        
        # Prepare input matrix
        X = np.column_stack([parameter_samples[name] for name in param_names])
        
        # SALib problem definition
        problem = {
            'num_vars': n_params,
            'names': param_names,
            'bounds': [[0, 1]] * n_params  # Normalized bounds
        }
        
        sensitivity_results = {}
        
        for output_name, Y in output_samples.items():
            # Filter valid samples
            valid_mask = np.isfinite(Y)
            if np.sum(valid_mask) < len(Y) * 0.5:  # Need at least 50% valid samples
                warnings.warn(f"Too many invalid samples for {output_name}, skipping sensitivity")
                continue
            
            X_valid = X[valid_mask]
            Y_valid = Y[valid_mask]
            
            try:
                # Normalize inputs to [0,1] for Sobol analysis
                X_normalized = np.zeros_like(X_valid)
                for i, param_name in enumerate(param_names):
                    param_dist = self.parameter_distributions[param_name]
                    X_normalized[:, i] = param_dist.cdf(X_valid[:, i])
                
                # Compute Sobol indices
                if len(Y_valid) >= 2 * n_params:  # Minimum samples needed
                    Si = sobol.analyze(problem, Y_valid, print_to_console=False)
                    
                    sensitivity_results[output_name] = {
                        'first_order': dict(zip(param_names, Si['S1'])),
                        'first_order_conf': dict(zip(param_names, Si['S1_conf'])),
                        'total_order': dict(zip(param_names, Si['ST'])),
                        'total_order_conf': dict(zip(param_names, Si['ST_conf'])),
                        'second_order': Si.get('S2', None)
                    }
                else:
                    warnings.warn(f"Insufficient samples for {output_name} sensitivity analysis")
                    
            except Exception as e:
                warnings.warn(f"Sensitivity analysis failed for {output_name}: {e}")
        
        return sensitivity_results
    
    def sensitivity_analysis(self, output_names: List[str], 
                           parameter_samples: Optional[Dict[str, np.ndarray]] = None,
                           output_samples: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict]:
        """
        Perform global sensitivity analysis using Sobol indices.
        
        Parameters:
        -----------
        output_names : List of output quantity names to analyze
        parameter_samples : Optional pre-computed parameter samples
        output_samples : Optional pre-computed output samples
        
        Returns:
        --------
        Dict with Sobol indices for each output
        """
        if not HAS_SALIB:
            warnings.warn("SALib not available for sensitivity analysis")
            return {}
        
        # Generate samples if not provided
        if parameter_samples is None:
            print("  Generating parameter samples for sensitivity analysis...")
            parameter_samples = self.generate_parameter_samples()
        
        if output_samples is None:
            print("  Evaluating model for sensitivity analysis...")
            output_samples = self.evaluate_model_fast(parameter_samples)
        
        # Set up problem definition for SALib
        param_names = list(self.parameter_distributions.keys())
        problem = {
            'num_vars': len(param_names),
            'names': param_names,
            'bounds': [[0, 1]] * len(param_names)  # Normalized bounds
        }
        
        sensitivity_results = {}
        
        for output_name in output_names:
            if output_name not in output_samples:
                warnings.warn(f"Output {output_name} not found in samples")
                continue
                
            try:
                # Get output values
                Y = output_samples[output_name]
                
                # Filter out invalid samples
                valid_mask = np.isfinite(Y)
                if np.sum(valid_mask) < len(Y) * 0.8:
                    warnings.warn(f"Too many invalid samples for {output_name}")
                    continue
                
                Y_valid = Y[valid_mask]
                
                # For sensitivity analysis, we need the parameter samples in normalized form
                # This is a simplified approach - in practice, you'd use the actual Sobol sampling
                X_normalized = np.column_stack([
                    (parameter_samples[name][valid_mask] - 
                     np.min(parameter_samples[name][valid_mask])) /
                    (np.max(parameter_samples[name][valid_mask]) - 
                     np.min(parameter_samples[name][valid_mask]) + 1e-10)
                    for name in param_names
                ])
                
                # Compute Sobol indices (simplified calculation)
                # Note: This is an approximation - full Sobol requires special sampling
                try:
                    # First-order indices (correlation-based approximation)
                    S1 = {}
                    for i, param_name in enumerate(param_names):
                        correlation = np.corrcoef(X_normalized[:, i], Y_valid)[0, 1]
                        S1[param_name] = correlation**2 if not np.isnan(correlation) else 0.0
                    
                    sensitivity_results[output_name] = {'S1': S1}
                    
                except Exception as e:
                    warnings.warn(f"Sobol analysis failed for {output_name}: {e}")
                    continue
                    
            except Exception as e:
                warnings.warn(f"Sensitivity analysis failed for {output_name}: {e}")
                continue
        
        return sensitivity_results
    
    def build_polynomial_chaos_surrogate(self, parameter_samples: Dict[str, np.ndarray],
                                       output_samples: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Build Polynomial Chaos Expansion surrogate models."""
        
        if not HAS_CHAOSPY:
            warnings.warn("chaospy not available, skipping surrogate modeling")
            return {}
        
        surrogate_models = {}
        param_names = list(parameter_samples.keys())
        
        # Create joint distribution for parameters
        param_distributions_cp = []
        for param_name in param_names:
            param_dist = self.parameter_distributions[param_name]
            # Convert scipy distribution to chaospy
            if hasattr(param_dist.distribution, 'a') and hasattr(param_dist.distribution, 'b'):
                # Uniform distribution
                a, b = param_dist.distribution.a, param_dist.distribution.b
                cp_dist = cp.Uniform(a, b)
            elif hasattr(param_dist.distribution, 'loc') and hasattr(param_dist.distribution, 'scale'):
                # Normal distribution
                loc, scale = param_dist.distribution.loc, param_dist.distribution.scale
                cp_dist = cp.Normal(loc, scale)
            else:
                # Fallback to uniform
                samples = parameter_samples[param_name]
                cp_dist = cp.Uniform(np.min(samples), np.max(samples))
            
            param_distributions_cp.append(cp_dist)
        
        joint_distribution = cp.J(*param_distributions_cp)
        
        # Create input matrix
        X = np.column_stack([parameter_samples[name] for name in param_names])
        
        for output_name, Y in output_samples.items():
            # Filter valid samples
            valid_mask = np.isfinite(Y)
            if np.sum(valid_mask) < 100:  # Need sufficient samples
                continue
            
            X_valid = X[valid_mask]
            Y_valid = Y[valid_mask]
            
            try:
                # Create polynomial expansion
                degree = self.uq_config.pce_degree
                expansion = cp.generate_expansion(degree, joint_distribution)
                
                # Fit surrogate model
                surrogate = cp.fit_regression(expansion, X_valid.T, Y_valid)
                
                # Validation: compute R² on training data
                Y_pred = surrogate(*X_valid.T)
                r2 = 1 - np.var(Y_valid - Y_pred) / np.var(Y_valid)
                
                surrogate_models[output_name] = {
                    'model': surrogate,
                    'expansion': expansion,
                    'distribution': joint_distribution,
                    'r2_score': r2,
                    'n_terms': len(expansion),
                    'degree': degree
                }
                
                print(f"Surrogate for {output_name}: R² = {r2:.3f}, {len(expansion)} terms")
                
            except Exception as e:
                warnings.warn(f"Surrogate modeling failed for {output_name}: {e}")
        
        return surrogate_models
    
    def robust_optimization(self, output_samples: Dict[str, np.ndarray],
                          parameter_samples: Dict[str, np.ndarray],
                          objective: str = 'mean_minus_std',
                          primary_output: str = 'transfer_rate') -> Dict[str, Any]:
        """
        Perform robust optimization to find parameters that maximize 
        objective under uncertainty.
        
        Parameters:
        -----------
        objective : str
            'mean_minus_std', 'worst_case_percentile', 'mean', 'median'
        """
        
        if primary_output not in output_samples:
            warnings.warn(f"Primary output {primary_output} not found")
            return {}
        
        Y = output_samples[primary_output]
        valid_mask = np.isfinite(Y)
        
        if np.sum(valid_mask) < 10:
            warnings.warn("Insufficient valid samples for robust optimization")
            return {}
        
        # Group samples by similar parameter values for robust statistics
        # This is a simplified approach - in practice would use more sophisticated clustering
        
        param_names = list(parameter_samples.keys())
        X = np.column_stack([parameter_samples[name] for name in param_names])
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        
        # Find best sample based on objective
        if objective == 'mean_minus_std':
            # Conservative approach: maximize (mean - std)
            # This requires local estimation of mean and std
            # For simplicity, use global statistics
            mean_Y = np.mean(Y_valid)
            std_Y = np.std(Y_valid)
            best_idx = np.argmax(Y_valid - std_Y)  # Conservative choice
            
        elif objective == 'worst_case_percentile':
            # Find samples above 95th percentile
            percentile_95 = np.percentile(Y_valid, 95)
            high_performers = Y_valid >= percentile_95
            if np.sum(high_performers) > 0:
                best_idx = np.where(valid_mask)[0][np.where(high_performers)[0][0]]
            else:
                best_idx = np.argmax(Y_valid)
                
        elif objective == 'mean':
            best_idx = np.argmax(Y_valid)
            
        elif objective == 'median':
            # Find sample closest to median of top quartile
            top_quartile = Y_valid >= np.percentile(Y_valid, 75)
            if np.sum(top_quartile) > 0:
                median_top = np.median(Y_valid[top_quartile])
                best_idx = np.argmin(np.abs(Y_valid - median_top))
            else:
                best_idx = np.argmax(Y_valid)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Extract optimal parameters
        optimal_params = {}
        for i, param_name in enumerate(param_names):
            optimal_params[param_name] = X_valid[best_idx, i]
        
        # Compute statistics around optimal point
        optimal_output = Y_valid[best_idx]
        
        return {
            'objective': objective,
            'optimal_parameters': optimal_params,
            'optimal_output': optimal_output,
            'output_statistics': {
                'mean': np.mean(Y_valid),
                'std': np.std(Y_valid),
                'median': np.median(Y_valid),
                'percentile_95': np.percentile(Y_valid, 95)
            }
        }
    
    def run_full_uq_analysis(self) -> UQResults:
        """Run complete UQ workflow: sample → evaluate → analyze."""
        
        start_time = time.time()
        
        print("="*60)
        print("🌟 RUNNING FULL UQ ANALYSIS")
        print("="*60)
        
        # Stage 1: Generate parameter samples
        print("\n1️⃣ Generating parameter samples...")
        parameter_samples = self.generate_parameter_samples()
        print(f"   Generated {len(next(iter(parameter_samples.values())))} samples")
        print(f"   Using {self.uq_config.sampling_method} sampling")
        
        # Stage 2: Evaluate model
        print("\n2️⃣ Evaluating spin network portal...")
        output_samples = self.evaluate_model(parameter_samples)
        
        # Stage 3: Statistical analysis
        print("\n3️⃣ Computing statistical summaries...")
        statistical_summary = self.compute_statistical_summary(output_samples)
        
        # Stage 4: Sensitivity analysis
        print("\n4️⃣ Performing sensitivity analysis...")
        sensitivity_indices = self.sobol_sensitivity_analysis(parameter_samples, output_samples)
        
        # Stage 5: Surrogate modeling
        print("\n5️⃣ Building surrogate models...")
        surrogate_models = self.build_polynomial_chaos_surrogate(parameter_samples, output_samples)
        
        # Stage 6: Robust optimization
        print("\n6️⃣ Robust optimization...")
        robust_optima = self.robust_optimization(output_samples, parameter_samples)
        
        end_time = time.time()
        
        # Store results
        self.results = UQResults(
            parameter_samples=parameter_samples,
            output_samples=output_samples,
            statistical_summary=statistical_summary,
            sensitivity_indices=sensitivity_indices,
            robust_optima=robust_optima,
            surrogate_models=surrogate_models,
            computational_time=end_time - start_time,
            config=self.uq_config
        )
        
        print(f"\n✅ UQ analysis complete in {end_time - start_time:.1f} seconds")
        return self.results
    
    def save_results(self, filepath: str):
        """Save UQ results to file."""
        if self.results is None:
            raise ValueError("No results to save. Run analysis first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load UQ results from file."""
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Results loaded from {filepath}")

def create_experimental_uq_config(photon_delay_error: float = 0.1,
                                uhecr_flux_error: float = 0.05,
                                coupling_prior_range: Tuple[float, float] = (1e-8, 1e-3)) -> SpinNetworkUQFramework:
    """
    Create UQ framework with realistic experimental uncertainties.
    
    Parameters:
    -----------
    photon_delay_error : float
        Relative error in photon delay measurements
    uhecr_flux_error : float  
        Relative error in UHECR flux measurements
    coupling_prior_range : Tuple[float, float]
        Prior range for hidden sector coupling
    """
    
    base_config = SpinNetworkConfig()
    uq_config = UQConfig(n_samples=5000, sampling_method='latin_hypercube')
    
    framework = SpinNetworkUQFramework(base_config, uq_config)
    
    # Add experimental uncertainties
    framework.add_experimental_uncertainty(
        'photon_delay_measurement', 1.0, photon_delay_error, 'gaussian'
    )
    framework.add_experimental_uncertainty(
        'uhecr_flux_measurement', 1.0, uhecr_flux_error, 'log_normal'
    )
    
    # Update coupling prior
    framework.add_parameter_distribution(
        'base_coupling',
        stats.loguniform(coupling_prior_range[0], coupling_prior_range[1]),
        description="Hidden sector coupling with experimental constraints"
    )
    
    return framework

# Demonstration functions
def demo_basic_uq():
    """Demonstrate basic UQ functionality."""
    
    print("Basic UQ Demonstration")
    print("="*50)
    
    if not HAS_CORE:
        print("⚠ Core SU(2) module not available")
        return
    
    # Create simple UQ setup
    base_config = SpinNetworkConfig()
    uq_config = UQConfig(n_samples=100, sampling_method='latin_hypercube')
    
    framework = SpinNetworkUQFramework(base_config, uq_config)
    
    # Run analysis
    results = framework.run_full_uq_analysis()
    
    # Print summary
    print("\n📊 RESULTS SUMMARY")
    print("="*30)
    
    for output_name, stats in results.statistical_summary.items():
        print(f"\n{output_name}:")
        print(f"  Mean: {stats['mean']:.2e}")
        print(f"  Std:  {stats['std']:.2e}")
        print(f"  95% CI: [{stats['ci_lower']:.2e}, {stats['ci_upper']:.2e}]")
        print(f"  Valid: {stats['valid_fraction']*100:.1f}%")

if __name__ == "__main__":
    demo_basic_uq()
