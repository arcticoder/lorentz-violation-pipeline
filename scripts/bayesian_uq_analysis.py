#!/usr/bin/env python3
"""
Bayesian Uncertainty Quantification for Multi-Channel LIV Analysis

This module implements a comprehensive Bayesian inference framework for
combining constraints from:
1. GRB time delays
2. UHECR propagation 
3. Vacuum instability predictions
4. Hidden sector signatures

Key features:
- Joint posterior sampling via MCMC
- Correlated priors across observational channels
- Systematic uncertainty propagation
- Model comparison via Bayesian evidence
- Confidence region estimation for parameter space
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.special import logsumexp
import emcee
import corner
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

class BayesianLIVAnalysis:
    """
    Comprehensive Bayesian uncertainty quantification for LIV constraints.
    
    This class implements joint posterior sampling across multiple observational
    channels with proper uncertainty propagation and model comparison.
    """
    
    def __init__(self, data_files=None, theoretical_models=None):
        """
        Initialize the Bayesian LIV analysis framework.
        
        Parameters:
        -----------
        data_files : dict
            Dictionary containing paths to observational data
        theoretical_models : list
            List of theoretical frameworks to analyze
        """
        self.data_files = data_files or {}
        self.theoretical_models = theoretical_models or [
            'polymer_quantum', 'rainbow_gravity', 'string_theory', 'axion_like'
        ]
        
        # Physical constants and scales
        self.PLANCK_ENERGY = 1.22e19  # GeV
        self.SCHWINGER_FIELD = 1.32e16  # V/m
        
        # Observational data containers
        self.grb_data = None
        self.uhecr_data = None
        self.lab_constraints = None
        
        # MCMC setup
        self.sampler = None
        self.samples = None
        
        # Model comparison
        self.evidence_dict = {}
        
    def load_observational_data(self):
        """Load and prepare observational data for all channels."""
        print("Loading observational data...")
        
        # Load GRB time delay data
        try:
            grb_file = "results/grb_polynomial_analysis.csv"
            self.grb_data = pd.read_csv(grb_file)
            print(f"✓ Loaded GRB data: {len(self.grb_data)} events")
        except FileNotFoundError:
            print("⚠ GRB data not found, using mock data")
            self.grb_data = self._generate_mock_grb_data()
        
        # Load UHECR spectrum data
        try:
            uhecr_file = "results/uhecr_enhanced_exclusion.csv"
            self.uhecr_data = pd.read_csv(uhecr_file)
            print(f"✓ Loaded UHECR data: {len(self.uhecr_data)} energy bins")
        except FileNotFoundError:
            print("⚠ UHECR data not found, using mock data")
            self.uhecr_data = self._generate_mock_uhecr_data()
        
        # Load laboratory constraint data
        try:
            lab_file = "results/unified_liv_framework_results.csv"
            self.lab_constraints = pd.read_csv(lab_file)
            print(f"✓ Loaded lab data: {len(self.lab_constraints)} model points")
        except FileNotFoundError:
            print("⚠ Lab data not found, using mock data")
            self.lab_constraints = self._generate_mock_lab_data()
    
    def _generate_mock_grb_data(self):
        """Generate mock GRB data for testing."""
        np.random.seed(42)
        n_grbs = 15
        
        data = []
        for i in range(n_grbs):
            redshift = np.random.uniform(0.1, 3.0)
            n_photons = np.random.randint(20, 100)
            
            for j in range(n_photons):
                energy = np.random.lognormal(0, 1.5)  # GeV
                # Mock time delay with LIV signature + noise
                time_delay = (1e-17 * energy + 1e-33 * energy**2) * redshift + \
                           np.random.normal(0, 0.1)  # seconds
                
                data.append({
                    'grb_id': f'GRB_{i:03d}',
                    'redshift': redshift,
                    'energy_gev': energy,
                    'time_delay_s': time_delay,
                    'time_error_s': np.random.uniform(0.01, 0.1)
                })
        
        return pd.DataFrame(data)
    
    def _generate_mock_uhecr_data(self):
        """Generate mock UHECR data for testing."""
        energies = np.logspace(18, 20.5, 25)  # eV
        
        data = []
        for energy in energies:
            # Mock flux with LIV modification + uncertainties
            flux_base = 1e-8 * (energy / 1e19)**(-2.7)
            liv_modification = 1 + 1e-2 * (energy / self.PLANCK_ENERGY)**2
            flux = flux_base * liv_modification
            flux_error = 0.2 * flux  # 20% uncertainty
            
            data.append({
                'energy_ev': energy,
                'flux': flux,
                'flux_error': flux_error
            })
        
        return pd.DataFrame(data)
    
    def _generate_mock_lab_data(self):
        """Generate mock laboratory constraint data."""
        mu_values = np.logspace(14, 20, 8)
        coupling_values = np.logspace(-12, -4, 5)
        
        data = []
        for model in self.theoretical_models:
            for mu in mu_values:
                for coupling in coupling_values:
                    # Mock enhancement factors and detectability
                    enhancement = 1 + 10 * coupling * (1e15 / mu)**2
                    detectable = enhancement > 1.01
                    
                    data.append({
                        'model_type': model,
                        'mu_GeV': mu,
                        'coupling': coupling,
                        'max_vacuum_enhancement': enhancement,
                        'vacuum_detectable': detectable
                    })
        
        return pd.DataFrame(data)
    
    def define_priors(self, model_type):
        """
        Define correlated priors for LIV parameters.
        
        Parameters:
        -----------
        model_type : str
            Theoretical framework ('polymer_quantum', 'string_theory', etc.)
            
        Returns:
        --------
        dict : Prior distribution parameters
        """
        priors = {}
        
        if model_type == 'polymer_quantum':
            # Polymer scale should be near Planck scale with correlation
            priors['log_mu'] = {'type': 'normal', 'loc': 18.0, 'scale': 1.0}
            priors['log_coupling'] = {'type': 'uniform', 'low': -12, 'high': -4}
            # Correlation: stronger coupling requires higher mu
            priors['correlation'] = {'mu_coupling': 0.3}
            
        elif model_type == 'rainbow_gravity':
            priors['log_mu'] = {'type': 'normal', 'loc': 17.5, 'scale': 1.2}
            priors['log_coupling'] = {'type': 'uniform', 'low': -12, 'high': -4}
            priors['correlation'] = {'mu_coupling': 0.2}
            
        elif model_type == 'string_theory':
            # String scale can vary more broadly
            priors['log_mu'] = {'type': 'uniform', 'low': 15, 'high': 19}
            priors['log_coupling'] = {'type': 'uniform', 'low': -12, 'high': -4}
            priors['correlation'] = {'mu_coupling': 0.1}
            
        elif model_type == 'axion_like':
            priors['log_mu'] = {'type': 'uniform', 'low': 14, 'high': 20}
            priors['log_coupling'] = {'type': 'uniform', 'low': -12, 'high': -4}
            priors['correlation'] = {'mu_coupling': -0.1}  # Weak anti-correlation
            
        return priors
    
    def log_prior(self, params, model_type):
        """
        Calculate log prior probability for parameter set.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Log prior probability
        """
        log_mu, log_coupling = params
        priors = self.define_priors(model_type)
        
        log_p = 0.0
        
        # Individual parameter priors
        if priors['log_mu']['type'] == 'normal':
            log_p += stats.norm.logpdf(log_mu, 
                                     priors['log_mu']['loc'], 
                                     priors['log_mu']['scale'])
        elif priors['log_mu']['type'] == 'uniform':
            if priors['log_mu']['low'] <= log_mu <= priors['log_mu']['high']:
                log_p += 0  # Uniform contribution
            else:
                return -np.inf
        
        if priors['log_coupling']['type'] == 'uniform':
            if priors['log_coupling']['low'] <= log_coupling <= priors['log_coupling']['high']:
                log_p += 0
            else:
                return -np.inf
        
        # Add correlation term (bivariate normal correction)
        if 'correlation' in priors and 'mu_coupling' in priors['correlation']:
            rho = priors['correlation']['mu_coupling']
            # Simplified correlation term
            log_p += -0.5 * rho * (log_mu - 17) * (log_coupling + 8)
        
        return log_p
    
    def grb_likelihood(self, params, model_type):
        """
        Calculate likelihood for GRB time delay data.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Log likelihood
        """
        log_mu, log_coupling = params
        mu = 10**log_mu  # GeV
        coupling = 10**log_coupling
        
        log_like = 0.0
        
        for _, grb in self.grb_data.iterrows():
            # Predicted time delay from LIV
            energy = grb['energy_gev']
            redshift = grb['redshift']
            
            # Model-dependent dispersion relation
            if model_type == 'polymer_quantum':
                predicted_delay = coupling * (energy / mu) * redshift * 1e-15
            elif model_type == 'rainbow_gravity':
                predicted_delay = coupling * (energy / mu)**1.2 * redshift * 1e-15
            elif model_type == 'string_theory':
                predicted_delay = coupling * (energy / mu)**2 * redshift * 1e-15
            elif model_type == 'axion_like':
                predicted_delay = coupling * np.sin(energy / mu) * redshift * 1e-16
            
            # Likelihood (assuming Gaussian errors)
            observed_delay = grb['time_delay_s']
            error = grb['time_error_s']
            
            log_like += stats.norm.logpdf(observed_delay, predicted_delay, error)
        
        return log_like
    
    def uhecr_likelihood(self, params, model_type):
        """
        Calculate likelihood for UHECR propagation data.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Log likelihood
        """
        log_mu, log_coupling = params
        mu = 10**log_mu * 1e9  # Convert to eV
        coupling = 10**log_coupling
        
        log_like = 0.0
        
        for _, point in self.uhecr_data.iterrows():
            energy = point['energy_ev']
            observed_flux = point['flux']
            flux_error = point['flux_error']
            
            # Model-dependent flux modification
            if model_type in ['polymer_quantum', 'rainbow_gravity']:
                # Suppression at high energies
                modification = 1 - coupling * (energy / mu)**2
            elif model_type == 'string_theory':
                # Enhanced propagation
                modification = 1 + coupling * (energy / mu)
            elif model_type == 'axion_like':
                # Oscillatory behavior
                modification = 1 + coupling * np.cos(energy / mu)
            
            # Ensure physical flux (positive)
            if modification <= 0:
                return -np.inf
            
            # Standard model flux (power law)
            sm_flux = 1e-8 * (energy / 1e19)**(-2.7)
            predicted_flux = sm_flux * modification
            
            log_like += stats.norm.logpdf(observed_flux, predicted_flux, flux_error)
        
        return log_like
    
    def vacuum_likelihood(self, params, model_type):
        """
        Calculate likelihood for vacuum instability predictions.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Log likelihood
        """
        log_mu, log_coupling = params
        mu = 10**log_mu  # GeV
        coupling = 10**log_coupling
        
        # Laboratory field strength
        field_strength = 1e15  # V/m
        
        # Model-dependent enhancement factor
        if model_type == 'polymer_quantum':
            enhancement = 1 + coupling * (field_strength / (mu * 1e9))**2
        elif model_type == 'rainbow_gravity':
            enhancement = 1 + coupling * (field_strength / (mu * 1e9))**1.5
        elif model_type == 'string_theory':
            enhancement = 1 + coupling * np.exp(-mu * 1e9 / field_strength)
        elif model_type == 'axion_like':
            enhancement = 1 + coupling * (field_strength / (mu * 1e9))
        
        # Likelihood based on detectability
        # Assume enhancement > 1.01 is detectable with 90% confidence
        if enhancement > 1.01:
            detection_prob = 0.9
        else:
            detection_prob = 0.1
        
        # All viable models should be detectable
        log_like = np.log(detection_prob)
        
        return log_like
    
    def hidden_sector_likelihood(self, params, model_type):
        """
        Calculate likelihood for hidden sector signatures.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Log likelihood
        """
        log_mu, log_coupling = params
        mu = 10**log_mu  # GeV
        coupling = 10**log_coupling
        
        # Only axion-like models have significant hidden sector signatures
        if model_type == 'axion_like':
            conversion_rate = coupling**2 * (1e-3 / mu)**2  # Hz
            
            # Likelihood based on laboratory accessibility
            if conversion_rate > 1e-8:  # Detectable threshold
                log_like = np.log(0.8)  # 80% detection confidence
            else:
                log_like = np.log(0.2)
        else:
            # Other models have minimal hidden sector signatures
            log_like = np.log(0.95)  # High confidence of no detection
        
        return log_like
    
    def log_likelihood(self, params, model_type):
        """
        Calculate total log likelihood across all observational channels.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Total log likelihood
        """
        # Individual channel likelihoods
        grb_like = self.grb_likelihood(params, model_type)
        uhecr_like = self.uhecr_likelihood(params, model_type)
        vacuum_like = self.vacuum_likelihood(params, model_type)
        hidden_like = self.hidden_sector_likelihood(params, model_type)
        
        # Check for invalid likelihood
        if not np.isfinite(grb_like + uhecr_like + vacuum_like + hidden_like):
            return -np.inf
        
        return grb_like + uhecr_like + vacuum_like + hidden_like
    
    def log_posterior(self, params, model_type):
        """
        Calculate log posterior probability.
        
        Parameters:
        -----------
        params : array
            [log_mu, log_coupling]
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        float : Log posterior probability
        """
        log_p = self.log_prior(params, model_type)
        
        if not np.isfinite(log_p):
            return -np.inf
        
        log_l = self.log_likelihood(params, model_type)
        
        return log_p + log_l
    
    def run_mcmc(self, model_type, n_walkers=32, n_steps=5000, n_burn=1000):
        """
        Run MCMC sampling for parameter estimation.
        
        Parameters:
        -----------
        model_type : str
            Theoretical framework to analyze
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of steps per walker
        n_burn : int
            Number of burn-in steps
            
        Returns:
        --------
        tuple : (samples, log_prob_samples)
        """
        print(f"Running MCMC for {model_type} model...")
        
        # Parameter space: [log_mu, log_coupling]
        ndim = 2
        
        # Initialize walkers around reasonable starting points
        priors = self.define_priors(model_type)
        
        if priors['log_mu']['type'] == 'normal':
            mu_init = priors['log_mu']['loc']
        else:
            mu_init = (priors['log_mu']['low'] + priors['log_mu']['high']) / 2
        
        coupling_init = (priors['log_coupling']['low'] + priors['log_coupling']['high']) / 2
        
        # Random initialization around starting point
        pos = np.random.normal([mu_init, coupling_init], [0.1, 0.1], (n_walkers, ndim))
        
        # Set up sampler
        self.sampler = emcee.EnsembleSampler(
            n_walkers, ndim, self.log_posterior, args=[model_type]
        )
        
        # Run burn-in
        print(f"Running burn-in ({n_burn} steps)...")
        pos, _, _ = self.sampler.run_mcmc(pos, n_burn, progress=True)
        self.sampler.reset()
        
        # Run production chain
        print(f"Running production chain ({n_steps} steps)...")
        self.sampler.run_mcmc(pos, n_steps, progress=True)
        
        # Extract samples
        samples = self.sampler.get_chain(flat=True)
        log_prob_samples = self.sampler.get_log_prob(flat=True)
        
        print(f"✓ MCMC completed. Acceptance fraction: {np.mean(self.sampler.acceptance_fraction):.3f}")
        
        return samples, log_prob_samples
    
    def calculate_model_evidence(self, model_type, samples, log_prob_samples):
        """
        Calculate Bayesian evidence for model comparison.
        
        Parameters:
        -----------
        model_type : str
            Theoretical framework
        samples : array
            MCMC samples
        log_prob_samples : array
            Log posterior values
            
        Returns:
        --------
        float : Log evidence estimate
        """
        # Use thermodynamic integration or harmonic mean estimator
        # For simplicity, using maximum likelihood approximation
        max_log_prob = np.max(log_prob_samples)
        
        # Simple Gaussian approximation around maximum
        max_idx = np.argmax(log_prob_samples)
        best_params = samples[max_idx]
        
        # Estimate covariance from samples
        cov = np.cov(samples.T)
        
        # Gaussian evidence approximation
        log_evidence = max_log_prob + 0.5 * np.log(2 * np.pi * np.linalg.det(cov))
        
        self.evidence_dict[model_type] = log_evidence
        
        return log_evidence
    
    def analyze_parameter_correlations(self, samples, model_type):
        """
        Analyze parameter correlations and uncertainties.
        
        Parameters:
        -----------
        samples : array
            MCMC samples
        model_type : str
            Theoretical framework
            
        Returns:
        --------
        dict : Correlation analysis results
        """
        log_mu_samples = samples[:, 0]
        log_coupling_samples = samples[:, 1]
        
        results = {
            'model_type': model_type,
            'n_samples': len(samples),
            'log_mu_mean': np.mean(log_mu_samples),
            'log_mu_std': np.std(log_mu_samples),
            'log_mu_credible': np.percentile(log_mu_samples, [16, 84]),
            'log_coupling_mean': np.mean(log_coupling_samples),
            'log_coupling_std': np.std(log_coupling_samples),
            'log_coupling_credible': np.percentile(log_coupling_samples, [16, 84]),
            'correlation_coefficient': np.corrcoef(log_mu_samples, log_coupling_samples)[0, 1]
        }
        
        return results
    
    def generate_corner_plot(self, samples, model_type, save_path=None):
        """
        Generate corner plot for parameter posterior.
        
        Parameters:
        -----------
        samples : array
            MCMC samples
        model_type : str
            Theoretical framework
        save_path : str, optional
            Path to save plot
        """
        labels = [r'$\log_{10}(\mu/\mathrm{GeV})$', r'$\log_{10}(g)$']
        
        fig = corner.corner(
            samples, 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        
        fig.suptitle(f'Parameter Posterior: {model_type.replace("_", " ").title()}', 
                    fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Corner plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_models(self):
        """
        Compare models using Bayesian evidence.
        
        Returns:
        --------
        pd.DataFrame : Model comparison results
        """
        if not self.evidence_dict:
            print("⚠ No evidence calculations found. Run MCMC first.")
            return None
        
        # Calculate Bayes factors relative to best model
        max_evidence = max(self.evidence_dict.values())
        
        results = []
        for model, evidence in self.evidence_dict.items():
            bayes_factor = evidence - max_evidence
            results.append({
                'model': model,
                'log_evidence': evidence,
                'bayes_factor': bayes_factor,
                'relative_probability': np.exp(bayes_factor)
            })
        
        df = pd.DataFrame(results).sort_values('log_evidence', ascending=False)
        
        print("\n" + "="*60)
        print("BAYESIAN MODEL COMPARISON")
        print("="*60)
        print(df.to_string(index=False, float_format='%.2f'))
        print("="*60)
        
        return df
    
    def run_full_analysis(self, models_to_analyze=None, save_results=True):
        """
        Run complete Bayesian analysis for all models.
        
        Parameters:
        -----------
        models_to_analyze : list, optional
            Models to analyze (default: all)
        save_results : bool
            Whether to save results to files
            
        Returns:
        --------
        dict : Complete analysis results
        """
        if models_to_analyze is None:
            models_to_analyze = self.theoretical_models
        
        print("\n" + "="*80)
        print("BAYESIAN UNCERTAINTY QUANTIFICATION FOR LIV CONSTRAINTS")
        print("="*80)
        
        # Load data
        self.load_observational_data()
        
        all_results = {}
        
        for model in models_to_analyze:
            print(f"\n📊 Analyzing {model.replace('_', ' ').title()} Model")
            print("-" * 50)
            
            # Run MCMC
            samples, log_prob = self.run_mcmc(model)
            
            # Calculate evidence
            evidence = self.calculate_model_evidence(model, samples, log_prob)
            
            # Parameter analysis
            param_results = self.analyze_parameter_correlations(samples, model)
            
            # Generate plots
            if save_results:
                plot_path = f"results/bayesian_{model}_corner.png"
                self.generate_corner_plot(samples, model, plot_path)
            
            # Store results
            all_results[model] = {
                'samples': samples,
                'log_prob': log_prob,
                'evidence': evidence,
                'parameters': param_results
            }
            
            print(f"✓ {model} analysis complete")
        
        # Model comparison
        comparison_df = self.compare_models()
        
        if save_results:
            # Save comparison results
            comparison_df.to_csv("results/bayesian_model_comparison.csv", index=False)
            
            # Save parameter summaries
            param_summary = pd.DataFrame([
                all_results[model]['parameters'] for model in models_to_analyze
            ])
            param_summary.to_csv("results/bayesian_parameter_summary.csv", index=False)
            
            print(f"\n✓ Results saved to results/bayesian_*.csv")
        
        return all_results

def main():
    """Run Bayesian UQ analysis."""
    # Initialize analysis
    analyzer = BayesianLIVAnalysis()
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    print("\n🎉 Bayesian analysis complete!")
    print("📊 Check results/bayesian_*.csv for detailed outputs")
    print("📈 Corner plots saved as results/bayesian_*_corner.png")

if __name__ == "__main__":
    main()
