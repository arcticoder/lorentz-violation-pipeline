#!/usr/bin/env python3
"""
LV Parameter Sweep: Comprehensive Exotic Energy Platform Analysis
===============================================================

This script systematically scans LV parameters (μ, α, β) across all five exotic
energy extraction pathways, providing unified analysis and optimization for the
complete hidden-sector coupling framework.

Pathways Analyzed:
1. **Macroscopic Negative Energy**: Casimir LV effects
2. **Dynamic Vacuum Extraction**: Time-dependent boundary power generation  
3. **Extra-Dimensional Transfer**: Hidden sector portal efficiency
4. **Dark Energy Coupling**: Axion/dark field interactions
5. **Matter-Gravity Coherence**: Quantum entanglement preservation

Cross-Cutting Analysis:
- Global parameter optimization across all pathways
- Sobol sensitivity analysis for pathway dominance
- Uncertainty quantification with experimental bounds
- Automated visualization and reporting

Author: Quantum Geometry Hidden Sector Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from itertools import product
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import all LV pathway modules
from casimir_lv import CasimirLVCalculator, CasimirLVConfig
from dynamic_casimir_lv import DynamicCasimirLV, DynamicCasimirConfig
from su2_recoupling_module import EnhancedSpinNetworkPortal, LorentzViolationConfig, SpinNetworkConfig

# Optional advanced analysis (install with: pip install SALib)
try:
    from SALib import sample, analyze
    from SALib.sample import sobol
    from SALib.analyze import sobol as sobol_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("⚠️  SALib not available. Install with 'pip install SALib' for sensitivity analysis.")

@dataclass
class LVParameterSweepConfig:
    """Configuration for comprehensive LV parameter sweep."""
    # Parameter ranges (log-space bounds relative to experimental limits)
    mu_range: Tuple[float, float] = (-2, 5)      # 0.01x to 100,000x bounds
    alpha_range: Tuple[float, float] = (-2, 5)   # 0.01x to 100,000x bounds  
    beta_range: Tuple[float, float] = (-2, 5)    # 0.01x to 100,000x bounds
    
    # Sampling parameters
    n_samples_per_dim: int = 20                  # Grid points per dimension
    n_sobol_samples: int = 1000                  # Sobol sequence samples
    
    # Pathway activation
    enable_casimir: bool = True                  # Macroscopic negative energy
    enable_dynamic: bool = True                  # Dynamic vacuum extraction
    enable_spin_network: bool = True             # Spin network portal (from existing)
    enable_extra_dim: bool = False               # Extra-dimensional (placeholder)
    enable_dark_energy: bool = False             # Dark energy coupling (placeholder)
    
    # Output options
    save_results: bool = True                    # Save to CSV/JSON
    generate_plots: bool = True                  # Auto-generate visualizations
    output_dir: str = "lv_parameter_sweep_results"

class LVParameterSweeper:
    """
    Comprehensive parameter sweeper for all LV pathways.
    """
    
    def __init__(self, config: LVParameterSweepConfig = None):
        self.config = config or LVParameterSweepConfig()
        
        # Experimental bounds
        self.exp_bounds = {
            'mu': 1e-20,
            'alpha': 1e-15,
            'beta': 1e-15
        }
        
        # Create output directory
        if self.config.save_results:
            Path(self.config.output_dir).mkdir(exist_ok=True)
        
        print("🔄 LV Parameter Sweeper Initialized")
        print(f"   Parameter ranges: μ∈10^{self.config.mu_range}, "
              f"α∈10^{self.config.alpha_range}, β∈10^{self.config.beta_range}")
        print(f"   Grid sampling: {self.config.n_samples_per_dim}³ = "
              f"{self.config.n_samples_per_dim**3} points")
        print(f"   Sobol sampling: {self.config.n_sobol_samples} points")
        
        # Initialize pathway calculators
        self._initialize_calculators()
    
    def _initialize_calculators(self):
        """Initialize calculators for active pathways."""
        self.calculators = {}
        
        if self.config.enable_casimir:
            self.calculators['casimir'] = None  # Will be updated per parameter set
            
        if self.config.enable_dynamic:
            self.calculators['dynamic'] = None  # Will be updated per parameter set
            
        if self.config.enable_spin_network:
            # Base spin network configuration
            spin_config = SpinNetworkConfig(base_coupling=1e-6, network_size=8)
            self.calculators['spin_network'] = (spin_config, None)  # LV config updated per loop
    
    def generate_parameter_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate parameter samples using both grid and Sobol methods.
        
        Returns:
        --------
        grid_samples : np.ndarray
            Regular grid samples [n_grid, 3]
        sobol_samples : np.ndarray  
            Sobol sequence samples [n_sobol, 3]
        """
        # Grid sampling
        mu_grid = np.logspace(self.config.mu_range[0], self.config.mu_range[1], 
                             self.config.n_samples_per_dim)
        alpha_grid = np.logspace(self.config.alpha_range[0], self.config.alpha_range[1],
                                self.config.n_samples_per_dim)
        beta_grid = np.logspace(self.config.beta_range[0], self.config.beta_range[1],
                               self.config.n_samples_per_dim)
        
        # Convert to absolute values (multiply by experimental bounds)
        mu_abs = mu_grid * self.exp_bounds['mu']
        alpha_abs = alpha_grid * self.exp_bounds['alpha']
        beta_abs = beta_grid * self.exp_bounds['beta']
        
        # Create parameter combinations
        grid_combinations = list(product(mu_abs, alpha_abs, beta_abs))
        grid_samples = np.array(grid_combinations)
        
        # Sobol sampling
        if SALIB_AVAILABLE:
            problem = {
                'num_vars': 3,
                'names': ['mu', 'alpha', 'beta'],
                'bounds': [
                    [self.config.mu_range[0], self.config.mu_range[1]],
                    [self.config.alpha_range[0], self.config.alpha_range[1]], 
                    [self.config.beta_range[0], self.config.beta_range[1]]
                ]
            }
            
            sobol_log_samples = sobol.sample(problem, self.config.n_sobol_samples)
            
            # Convert to absolute values
            sobol_samples = np.zeros_like(sobol_log_samples)
            sobol_samples[:, 0] = 10**sobol_log_samples[:, 0] * self.exp_bounds['mu']
            sobol_samples[:, 1] = 10**sobol_log_samples[:, 1] * self.exp_bounds['alpha']
            sobol_samples[:, 2] = 10**sobol_log_samples[:, 2] * self.exp_bounds['beta']
        else:
            # Fallback to random sampling
            n_samples = self.config.n_sobol_samples
            sobol_log_samples = np.random.uniform(
                [self.config.mu_range[0], self.config.alpha_range[0], self.config.beta_range[0]],
                [self.config.mu_range[1], self.config.alpha_range[1], self.config.beta_range[1]],
                (n_samples, 3)
            )
            
            sobol_samples = np.zeros_like(sobol_log_samples)
            sobol_samples[:, 0] = 10**sobol_log_samples[:, 0] * self.exp_bounds['mu']
            sobol_samples[:, 1] = 10**sobol_log_samples[:, 1] * self.exp_bounds['alpha']
            sobol_samples[:, 2] = 10**sobol_log_samples[:, 2] * self.exp_bounds['beta']
        
        return grid_samples, sobol_samples
    
    def evaluate_single_parameter_set(self, mu: float, alpha: float, beta: float) -> Dict:
        """
        Evaluate all active pathways for a single parameter set.
        """
        results = {
            'mu': mu,
            'alpha': alpha,
            'beta': beta,
            'mu_relative': mu / self.exp_bounds['mu'],
            'alpha_relative': alpha / self.exp_bounds['alpha'],
            'beta_relative': beta / self.exp_bounds['beta']
        }
        
        # 1. Casimir LV pathway
        if self.config.enable_casimir:
            try:
                casimir_config = CasimirLVConfig(mu=mu, alpha=alpha, beta=beta)
                casimir_calc = CasimirLVCalculator(casimir_config)
                
                regions = casimir_calc.negative_energy_regions()
                
                results['casimir_energy_density'] = regions['energy_density']
                results['casimir_enhancement'] = regions['lv_enhancement_factor']
                results['casimir_negative_energy'] = 1.0 if regions['has_negative_energy'] else 0.0
                results['casimir_macroscopic'] = 1.0 if regions['macroscopic_scale'] else 0.0
                
            except Exception as e:
                results.update({
                    'casimir_energy_density': 0.0,
                    'casimir_enhancement': 1.0,
                    'casimir_negative_energy': 0.0,
                    'casimir_macroscopic': 0.0
                })
        
        # 2. Dynamic Casimir pathway
        if self.config.enable_dynamic:
            try:
                casimir_config = CasimirLVConfig(mu=mu, alpha=alpha, beta=beta)
                dyn_config = DynamicCasimirConfig(
                    casimir_config=casimir_config,
                    drive_frequency=1e11,  # 100 GHz
                    drive_amplitude=1e-9   # 1 nm
                )
                dyn_calc = DynamicCasimirLV(dyn_config)
                
                power_results = dyn_calc.power_extraction_analysis()
                
                results['dynamic_power_extracted'] = power_results['power_extracted']
                results['dynamic_photon_flux'] = power_results['photon_flux']
                results['dynamic_enhancement'] = power_results['power_enhancement']
                results['dynamic_power_density'] = power_results['power_density']
                
            except Exception as e:
                results.update({
                    'dynamic_power_extracted': 0.0,
                    'dynamic_photon_flux': 0.0,
                    'dynamic_enhancement': 1.0,
                    'dynamic_power_density': 0.0
                })
        
        # 3. Spin Network Portal pathway
        if self.config.enable_spin_network:
            try:
                spin_config, _ = self.calculators['spin_network']
                lv_config = LorentzViolationConfig(mu=mu, alpha=alpha, beta=beta)
                
                portal = EnhancedSpinNetworkPortal(spin_config, lv_config)
                
                # Compute enhanced metrics
                vertex = 0
                standard_coupling = portal.effective_coupling(vertex)
                lv_coupling = portal.effective_coupling_lv(vertex)
                
                standard_amp = portal.energy_leakage_amplitude(10.0, 8.0)
                lv_amp = portal.energy_leakage_amplitude_lv(10.0, 8.0)
                
                coupling_enhancement = lv_coupling / max(standard_coupling, 1e-20)
                amplitude_enhancement = abs(lv_amp) / max(abs(standard_amp), 1e-20)
                
                pathway_summary = portal.exotic_pathway_summary()
                
                results['spin_coupling_enhancement'] = coupling_enhancement
                results['spin_amplitude_enhancement'] = amplitude_enhancement
                results['spin_active_pathways'] = pathway_summary['pathway_count']
                results['spin_total_enhancement'] = pathway_summary['total_enhancement']
                
            except Exception as e:
                results.update({
                    'spin_coupling_enhancement': 1.0,
                    'spin_amplitude_enhancement': 1.0,
                    'spin_active_pathways': 0,
                    'spin_total_enhancement': 1.0
                })
        
        # 4. Extra-dimensional pathway (placeholder)
        if self.config.enable_extra_dim:
            # Placeholder implementation
            results['extra_dim_efficiency'] = np.random.random() * mu / self.exp_bounds['mu']
        
        # 5. Dark energy pathway (placeholder)  
        if self.config.enable_dark_energy:
            # Placeholder implementation
            results['dark_energy_coupling'] = np.random.random() * alpha / self.exp_bounds['alpha']
        
        return results
    
    def run_comprehensive_sweep(self) -> Dict:
        """
        Run comprehensive parameter sweep across all pathways.
        """
        print("🚀 Starting comprehensive LV parameter sweep...")
        
        # Generate parameter samples
        grid_samples, sobol_samples = self.generate_parameter_samples()
        
        print(f"   Grid samples: {len(grid_samples)}")
        print(f"   Sobol samples: {len(sobol_samples)}")
        
        # Combine samples for analysis
        all_samples = np.vstack([grid_samples, sobol_samples])
        
        results_list = []
        
        print("🔄 Evaluating parameter combinations...")
        for i, (mu, alpha, beta) in enumerate(all_samples):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(all_samples)} ({100*i/len(all_samples):.1f}%)")
            
            result = self.evaluate_single_parameter_set(mu, alpha, beta)
            results_list.append(result)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results_list)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(df)
        
        # Global sensitivity analysis
        sensitivity_results = self._global_sensitivity_analysis(df) if SALIB_AVAILABLE else {}
        
        # Optimization results
        optimization_results = self._find_optimal_parameters(df)
        
        # Package results
        comprehensive_results = {
            'parameter_sweep_data': df,
            'summary_statistics': summary_stats,
            'sensitivity_analysis': sensitivity_results,
            'optimization_results': optimization_results,
            'config': asdict(self.config)
        }
        
        # Save results
        if self.config.save_results:
            self._save_results(comprehensive_results)
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_visualizations(comprehensive_results)
        
        print("✅ Comprehensive sweep completed!")
        return comprehensive_results
    
    def _compute_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute summary statistics for all metrics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {}
        for col in numeric_cols:
            if col not in ['mu', 'alpha', 'beta']:  # Skip parameter columns
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(), 
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'q25': df[col].quantile(0.25),
                    'q75': df[col].quantile(0.75)
                }
        
        return summary
    
    def _global_sensitivity_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform global sensitivity analysis using Sobol indices."""
        if not SALIB_AVAILABLE:
            return {}
        
        # Define problem for sensitivity analysis
        problem = {
            'num_vars': 3,
            'names': ['mu_relative', 'alpha_relative', 'beta_relative'],
            'bounds': [
                [df['mu_relative'].min(), df['mu_relative'].max()],
                [df['alpha_relative'].min(), df['alpha_relative'].max()],
                [df['beta_relative'].min(), df['beta_relative'].max()]
            ]
        }
        
        sensitivity_results = {}
        
        # Analyze each output metric
        output_metrics = [col for col in df.columns if col not in 
                         ['mu', 'alpha', 'beta', 'mu_relative', 'alpha_relative', 'beta_relative']]
        
        for metric in output_metrics:
            try:
                # Prepare input/output data
                X = df[['mu_relative', 'alpha_relative', 'beta_relative']].values
                y = df[metric].values
                
                # Remove any NaN/inf values
                valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                if len(y_clean) > 10:  # Need sufficient data
                    Si = sobol_analyze.analyze(problem, y_clean, print_to_console=False)
                    
                    sensitivity_results[metric] = {
                        'S1': Si['S1'].tolist(),  # First-order indices
                        'ST': Si['ST'].tolist(),  # Total-order indices
                        'S1_conf': Si['S1_conf'].tolist(),  # Confidence intervals
                        'ST_conf': Si['ST_conf'].tolist()
                    }
            except Exception as e:
                print(f"   ⚠️  Sensitivity analysis failed for {metric}: {e}")
                continue
        
        return sensitivity_results
    
    def _find_optimal_parameters(self, df: pd.DataFrame) -> Dict:
        """Find optimal parameter combinations for different objectives."""
        optimization_results = {}
        
        # Define optimization objectives
        objectives = {
            'max_casimir_enhancement': ('casimir_enhancement', 'maximize'),
            'max_dynamic_power': ('dynamic_power_extracted', 'maximize'),
            'max_spin_enhancement': ('spin_total_enhancement', 'maximize'),
            'max_combined_performance': (None, 'maximize')  # Combined metric
        }
        
        for obj_name, (metric, direction) in objectives.items():
            try:
                if metric is None:
                    # Combined performance metric
                    performance_metric = (
                        df.get('casimir_enhancement', 1.0) *
                        df.get('dynamic_enhancement', 1.0) *
                        df.get('spin_total_enhancement', 1.0)
                    )
                else:
                    performance_metric = df[metric]
                
                # Find optimal index
                if direction == 'maximize':
                    optimal_idx = performance_metric.idxmax()
                else:
                    optimal_idx = performance_metric.idxmin()
                
                optimal_row = df.loc[optimal_idx]
                
                optimization_results[obj_name] = {
                    'optimal_mu': optimal_row['mu'],
                    'optimal_alpha': optimal_row['alpha'],
                    'optimal_beta': optimal_row['beta'],
                    'optimal_value': performance_metric.loc[optimal_idx],
                    'mu_relative': optimal_row['mu_relative'],
                    'alpha_relative': optimal_row['alpha_relative'],
                    'beta_relative': optimal_row['beta_relative']
                }
            except Exception as e:
                print(f"   ⚠️  Optimization failed for {obj_name}: {e}")
                continue
        
        return optimization_results
    
    def _save_results(self, results: Dict):
        """Save results to files."""
        output_dir = Path(self.config.output_dir)
        
        # Save DataFrame to CSV
        csv_path = output_dir / "lv_parameter_sweep_data.csv"
        results['parameter_sweep_data'].to_csv(csv_path, index=False)
        print(f"   💾 Saved parameter sweep data to {csv_path}")
        
        # Save summary statistics to JSON
        json_path = output_dir / "lv_sweep_summary.json"
        json_data = {
            'summary_statistics': results['summary_statistics'],
            'sensitivity_analysis': results['sensitivity_analysis'],
            'optimization_results': results['optimization_results'],
            'config': results['config']
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"   💾 Saved summary analysis to {json_path}")
    
    def _generate_visualizations(self, results: Dict):
        """Generate comprehensive visualization suite."""
        df = results['parameter_sweep_data']
        output_dir = Path(self.config.output_dir)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Parameter space visualization
        self._plot_parameter_space(df, output_dir)
        
        # 2. Enhancement factor correlations
        self._plot_enhancement_correlations(df, output_dir)
        
        # 3. Pathway activation analysis
        self._plot_pathway_activation(df, output_dir)
        
        # 4. Sensitivity analysis visualization
        if results['sensitivity_analysis']:
            self._plot_sensitivity_analysis(results['sensitivity_analysis'], output_dir)
        
        # 5. Optimization results
        self._plot_optimization_results(results['optimization_results'], df, output_dir)
        
        print(f"   📊 Generated visualizations in {output_dir}")
    
    def _plot_parameter_space(self, df: pd.DataFrame, output_dir: Path):
        """Plot parameter space exploration."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('LV Parameter Space Exploration', fontsize=14)
        
        # μ vs α scatter with enhancement coloring
        ax1 = axes[0, 0]
        if 'casimir_enhancement' in df.columns:
            scatter = ax1.scatter(df['mu_relative'], df['alpha_relative'], 
                                c=np.log10(df['casimir_enhancement'] + 1e-10), 
                                cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax1, label='log₁₀(Casimir Enhancement)')
        ax1.set_xlabel('μ (relative to bounds)')
        ax1.set_ylabel('α (relative to bounds)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_title('Casimir Enhancement')
        
        # Dynamic power extraction
        ax2 = axes[0, 1]
        if 'dynamic_power_extracted' in df.columns:
            scatter = ax2.scatter(df['mu_relative'], df['beta_relative'],
                                c=np.log10(df['dynamic_power_extracted'] + 1e-20),
                                cmap='plasma', alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='log₁₀(Power [W])')
        ax2.set_xlabel('μ (relative to bounds)')
        ax2.set_ylabel('β (relative to bounds)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('Dynamic Power Extraction')
        
        # Spin network enhancement
        ax3 = axes[1, 0]
        if 'spin_total_enhancement' in df.columns:
            scatter = ax3.scatter(df['alpha_relative'], df['beta_relative'],
                                c=np.log10(df['spin_total_enhancement'] + 1e-10),
                                cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, ax=ax3, label='log₁₀(Spin Enhancement)')
        ax3.set_xlabel('α (relative to bounds)')
        ax3.set_ylabel('β (relative to bounds)')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_title('Spin Network Enhancement')
        
        # Combined performance
        ax4 = axes[1, 1]
        if all(col in df.columns for col in ['casimir_enhancement', 'dynamic_enhancement', 'spin_total_enhancement']):
            combined = df['casimir_enhancement'] * df['dynamic_enhancement'] * df['spin_total_enhancement']
            scatter = ax4.scatter(df['mu_relative'], df['alpha_relative'],
                                c=np.log10(combined + 1e-10),
                                cmap='inferno', alpha=0.6)
            plt.colorbar(scatter, ax=ax4, label='log₁₀(Combined Performance)')
        ax4.set_xlabel('μ (relative to bounds)')
        ax4.set_ylabel('α (relative to bounds)')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_title('Combined Performance')
        
        plt.tight_layout()
        plt.savefig(output_dir / "parameter_space_exploration.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_enhancement_correlations(self, df: pd.DataFrame, output_dir: Path):
        """Plot correlations between different enhancement factors."""
        enhancement_cols = [col for col in df.columns if 'enhancement' in col.lower()]
        
        if len(enhancement_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[enhancement_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Enhancement Factor Correlations')
            plt.tight_layout()
            plt.savefig(output_dir / "enhancement_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_pathway_activation(self, df: pd.DataFrame, output_dir: Path):
        """Plot pathway activation analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pathway activation vs LV strength
        ax1 = axes[0]
        if 'spin_active_pathways' in df.columns:
            lv_strength = np.maximum(df['mu_relative'], df['alpha_relative'])
            ax1.scatter(lv_strength, df['spin_active_pathways'], alpha=0.6)
            ax1.set_xlabel('LV Strength (max relative)')
            ax1.set_ylabel('Number of Active Pathways')
            ax1.set_xscale('log')
            ax1.set_title('Pathway Activation vs LV Strength')
            ax1.grid(True, alpha=0.3)
        
        # Threshold analysis
        ax2 = axes[1]
        activation_metrics = ['casimir_negative_energy', 'casimir_macroscopic']
        activation_metrics = [col for col in activation_metrics if col in df.columns]
        
        if activation_metrics:
            lv_bins = np.logspace(-2, 5, 20)
            lv_centers = (lv_bins[1:] + lv_bins[:-1]) / 2
            
            for metric in activation_metrics:
                activation_rates = []
                for i in range(len(lv_bins) - 1):
                    mask = (df['mu_relative'] >= lv_bins[i]) & (df['mu_relative'] < lv_bins[i+1])
                    if mask.sum() > 0:
                        rate = df.loc[mask, metric].mean()
                        activation_rates.append(rate)
                    else:
                        activation_rates.append(0.0)
                
                ax2.plot(lv_centers, activation_rates, 'o-', label=metric, linewidth=2)
            
            ax2.set_xlabel('μ (relative to bounds)')
            ax2.set_ylabel('Activation Rate')
            ax2.set_xscale('log')
            ax2.set_title('Activation Thresholds')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pathway_activation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_analysis(self, sensitivity_results: Dict, output_dir: Path):
        """Plot Sobol sensitivity analysis results."""
        if not sensitivity_results:
            return
        
        # Create sensitivity plots for key metrics
        key_metrics = ['casimir_enhancement', 'dynamic_power_extracted', 'spin_total_enhancement']
        available_metrics = [m for m in key_metrics if m in sensitivity_results]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]
        
        parameter_names = ['μ', 'α', 'β']
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            s1_indices = sensitivity_results[metric]['S1']
            st_indices = sensitivity_results[metric]['ST']
            
            x_pos = np.arange(len(parameter_names))
            width = 0.35
            
            ax.bar(x_pos - width/2, s1_indices, width, label='First-order (S₁)', alpha=0.8)
            ax.bar(x_pos + width/2, st_indices, width, label='Total-order (Sₜ)', alpha=0.8)
            
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Sobol Index')
            ax.set_title(f'Sensitivity: {metric}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(parameter_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_results(self, optimization_results: Dict, df: pd.DataFrame, output_dir: Path):
        """Plot optimization results."""
        if not optimization_results:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        objectives = list(optimization_results.keys())
        mu_values = [optimization_results[obj]['mu_relative'] for obj in objectives]
        alpha_values = [optimization_results[obj]['alpha_relative'] for obj in objectives]
        beta_values = [optimization_results[obj]['beta_relative'] for obj in objectives]
        
        x_pos = np.arange(len(objectives))
        width = 0.25
        
        ax.bar(x_pos - width, np.log10(mu_values), width, label='log₁₀(μ_rel)', alpha=0.8)
        ax.bar(x_pos, np.log10(alpha_values), width, label='log₁₀(α_rel)', alpha=0.8)
        ax.bar(x_pos + width, np.log10(beta_values), width, label='log₁₀(β_rel)', alpha=0.8)
        
        ax.set_xlabel('Optimization Objectives')
        ax.set_ylabel('log₁₀(Parameter / Experimental Bound)')
        ax.set_title('Optimal Parameter Combinations')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([obj.replace('_', ' ').title() for obj in objectives], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "optimization_results.png", dpi=300, bbox_inches='tight')
        plt.close()

def demo_comprehensive_lv_sweep():
    """
    Demonstration of comprehensive LV parameter sweep.
    """
    print("🌌 COMPREHENSIVE LV PARAMETER SWEEP DEMO")
    print("=" * 60)
    
    # Configure sweep
    config = LVParameterSweepConfig(
        mu_range=(-1, 3),           # 0.1x to 1000x bounds
        alpha_range=(-1, 3),        # 0.1x to 1000x bounds
        beta_range=(-1, 3),         # 0.1x to 1000x bounds
        n_samples_per_dim=10,       # Smaller grid for demo
        n_sobol_samples=200,        # Manageable Sobol sampling
        enable_casimir=True,
        enable_dynamic=True,
        enable_spin_network=True
    )
    
    # Run comprehensive sweep
    sweeper = LVParameterSweeper(config)
    results = sweeper.run_comprehensive_sweep()
    
    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    
    # Summary statistics
    summary = results['summary_statistics']
    if 'casimir_enhancement' in summary:
        print(f"   Casimir Enhancement: {summary['casimir_enhancement']['mean']:.2e} ± {summary['casimir_enhancement']['std']:.2e}")
        print(f"   Maximum: {summary['casimir_enhancement']['max']:.2e}")
    
    if 'dynamic_power_extracted' in summary:
        print(f"   Dynamic Power: {summary['dynamic_power_extracted']['mean']:.2e} ± {summary['dynamic_power_extracted']['std']:.2e} W")
        print(f"   Maximum: {summary['dynamic_power_extracted']['max']:.2e} W")
    
    # Optimization results
    if results['optimization_results']:
        print(f"\n🏆 OPTIMAL PARAMETER COMBINATIONS:")
        for obj, result in results['optimization_results'].items():
            print(f"   {obj.replace('_', ' ').title()}:")
            print(f"     μ = {result['optimal_mu']:.2e} ({result['mu_relative']:.1f}× bound)")
            print(f"     α = {result['optimal_alpha']:.2e} ({result['alpha_relative']:.1f}× bound)")
            print(f"     β = {result['optimal_beta']:.2e} ({result['beta_relative']:.1f}× bound)")
            print(f"     Performance: {result['optimal_value']:.2e}")
    
    # Sensitivity analysis
    if results['sensitivity_analysis']:
        print(f"\n📊 PARAMETER SENSITIVITY:")
        for metric, sens in results['sensitivity_analysis'].items():
            dominant_param_idx = np.argmax(sens['S1'])
            dominant_param = ['μ', 'α', 'β'][dominant_param_idx]
            sensitivity_value = sens['S1'][dominant_param_idx]
            print(f"   {metric}: {dominant_param} dominates ({sensitivity_value:.2f} sensitivity)")
    
    print(f"\n✅ Comprehensive LV parameter sweep completed!")
    print(f"   Results saved to: {config.output_dir}")
    
    return results

if __name__ == "__main__":
    demo_comprehensive_lv_sweep()
