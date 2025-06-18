#!/usr/bin/env python3
"""
Comprehensive LV Parameter Sweep: All Exotic Energy Pathways
============================================================

This script performs a comprehensive parameter sweep across all five exotic energy
pathways in the LV-powered hidden sector framework. It analyzes the scaling behavior,
optimization landscape, and sensitivity to LV parameters across all pathways.

Pathways Analyzed:
1. Casimir LV (Negative Energy)
2. Dynamic Casimir LV (Vacuum Energy Extraction)
3. Hidden Sector Portal (Extra-Dimensional)
4. Axion Coupling LV (Dark Energy)
5. Matter-Gravity Coherence (Quantum Entanglement)

Author: Quantum Geometry Hidden Sector Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import asdict

# Import all pathway modules
from casimir_lv import CasimirLVCalculator, CasimirLVConfig
from dynamic_casimir_lv import DynamicCasimirLV, DynamicCasimirConfig
from hidden_sector_portal import HiddenSectorPortal, HiddenSectorConfig
from axion_coupling_lv import AxionCouplingLV, AxionCouplingConfig
from matter_gravity_coherence import MatterGravityCoherence, MatterGravityConfig

class ComprehensiveLVSweep:
    """
    Comprehensive parameter sweep across all LV pathways.
    """
    
    def __init__(self):
        self.experimental_bounds = {
            'mu_lv': 1e-19,      # Current CPT violation bounds
            'alpha_lv': 1e-16,   # Lorentz violation bounds
            'beta_lv': 1e-13     # Various LV coupling bounds
        }
        
        # Parameter ranges for sweep
        self.parameter_ranges = {
            'mu_lv': np.logspace(-20, -16, 20),
            'alpha_lv': np.logspace(-17, -13, 20),
            'beta_lv': np.logspace(-14, -10, 20)
        }
        
        # Initialize pathway calculators
        self.initialize_pathways()
        
    def initialize_pathways(self):
        """Initialize all pathway calculators with baseline configurations."""
        
        # Casimir LV
        self.casimir_config = CasimirLVConfig(
            plate_separation=1e-6,
            plate_area=1e-4,
            dielectric_1=1.0,
            dielectric_2=1.0,
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        self.casimir_calc = CasimirLVCalculator(self.casimir_config)
        
        # Dynamic Casimir LV
        self.dynamic_casimir_config = DynamicCasimirConfig(
            cavity_length=0.01,
            cavity_width=0.01,
            cavity_height=0.01,
            modulation_frequency=1e9,
            modulation_amplitude=0.1,
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        self.dynamic_casimir_calc = DynamicCasimirLV(self.dynamic_casimir_config)
        
        # Hidden Sector Portal
        self.hidden_sector_config = HiddenSectorConfig(
            n_extra_dims=2,
            compactification_radius=1e-3,
            kinetic_mixing=1e-3,
            coupling_strength=1e-2,
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        self.hidden_sector_calc = HiddenSectorPortal(self.hidden_sector_config)
        
        # Axion Coupling LV
        self.axion_config = AxionCouplingConfig(
            axion_mass=1e-5,
            axion_decay_constant=1e16,
            photon_coupling=1e-10,
            oscillation_frequency=1e6,
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        self.axion_calc = AxionCouplingLV(self.axion_config)
        
        # Matter-Gravity Coherence
        self.coherence_config = MatterGravityConfig(
            particle_mass=1e-26,
            coherence_length=1e-6,
            coherence_time=1e-3,
            entanglement_depth=10,
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        self.coherence_calc = MatterGravityCoherence(self.coherence_config)
        
    def update_lv_parameters(self, mu_lv: float, alpha_lv: float, beta_lv: float):
        """Update LV parameters across all pathway calculators."""
        
        # Update Casimir LV
        self.casimir_config.mu_lv = mu_lv
        self.casimir_config.alpha_lv = alpha_lv
        self.casimir_config.beta_lv = beta_lv
        
        # Update Dynamic Casimir LV
        self.dynamic_casimir_config.mu_lv = mu_lv
        self.dynamic_casimir_config.alpha_lv = alpha_lv
        self.dynamic_casimir_config.beta_lv = beta_lv
        
        # Update Hidden Sector Portal
        self.hidden_sector_config.mu_lv = mu_lv
        self.hidden_sector_config.alpha_lv = alpha_lv
        self.hidden_sector_config.beta_lv = beta_lv
        
        # Update Axion Coupling LV
        self.axion_config.mu_lv = mu_lv
        self.axion_config.alpha_lv = alpha_lv
        self.axion_config.beta_lv = beta_lv
        
        # Update Matter-Gravity Coherence
        self.coherence_config.mu_lv = mu_lv
        self.coherence_config.alpha_lv = alpha_lv
        self.coherence_config.beta_lv = beta_lv
        
    def calculate_pathway_metrics(self, mu_lv: float, alpha_lv: float, beta_lv: float) -> Dict[str, float]:
        """Calculate metrics for all pathways at given LV parameters."""
        
        # Update parameters
        self.update_lv_parameters(mu_lv, alpha_lv, beta_lv)
        
        metrics = {}
        
        try:
            # Casimir LV metrics
            casimir_energy = self.casimir_calc.total_casimir_energy()
            casimir_power = abs(casimir_energy) * 1e6  # Convert to μW equivalent
            metrics['casimir_energy'] = casimir_energy
            metrics['casimir_power'] = casimir_power
            metrics['casimir_active'] = self.casimir_calc.is_pathway_active()
            
        except Exception as e:
            print(f"Casimir calculation error: {e}")
            metrics['casimir_energy'] = 0.0
            metrics['casimir_power'] = 0.0
            metrics['casimir_active'] = False
            
        try:
            # Dynamic Casimir LV metrics
            dynamic_power = self.dynamic_casimir_calc.total_power_output()
            dynamic_photons = self.dynamic_casimir_calc.photon_production_rate()
            metrics['dynamic_casimir_power'] = dynamic_power
            metrics['dynamic_casimir_photons'] = dynamic_photons
            metrics['dynamic_casimir_active'] = self.dynamic_casimir_calc.is_pathway_active()
            
        except Exception as e:
            print(f"Dynamic Casimir calculation error: {e}")
            metrics['dynamic_casimir_power'] = 0.0
            metrics['dynamic_casimir_photons'] = 0.0
            metrics['dynamic_casimir_active'] = False
            
        try:
            # Hidden Sector Portal metrics
            hidden_power = self.hidden_sector_calc.total_power_extraction()
            hidden_transfer_rate = self.hidden_sector_calc.energy_transfer_rate(1.0)
            metrics['hidden_sector_power'] = hidden_power
            metrics['hidden_sector_transfer_rate'] = hidden_transfer_rate
            metrics['hidden_sector_active'] = self.hidden_sector_calc.is_pathway_active()
            
        except Exception as e:
            print(f"Hidden sector calculation error: {e}")
            metrics['hidden_sector_power'] = 0.0
            metrics['hidden_sector_transfer_rate'] = 0.0
            metrics['hidden_sector_active'] = False
            
        try:
            # Axion Coupling LV metrics
            axion_osc_power = self.axion_calc.coherent_oscillation_power()
            axion_de_power = self.axion_calc.dark_energy_extraction_rate()
            axion_photon_rate = self.axion_calc.photon_production_rate()
            metrics['axion_oscillation_power'] = axion_osc_power
            metrics['axion_dark_energy_power'] = axion_de_power
            metrics['axion_photon_rate'] = axion_photon_rate
            metrics['axion_active'] = self.axion_calc.is_pathway_active()
            
        except Exception as e:
            print(f"Axion calculation error: {e}")
            metrics['axion_oscillation_power'] = 0.0
            metrics['axion_dark_energy_power'] = 0.0
            metrics['axion_photon_rate'] = 0.0
            metrics['axion_active'] = False
            
        try:
            # Matter-Gravity Coherence metrics
            coherence_power = self.coherence_calc.total_extractable_power()
            coherence_fidelity = self.coherence_calc.entanglement_fidelity_evolution(1.0)
            coherence_fisher = self.coherence_calc.quantum_fisher_information(1.0)
            metrics['coherence_power'] = coherence_power
            metrics['coherence_fidelity'] = coherence_fidelity
            metrics['coherence_fisher'] = coherence_fisher
            metrics['coherence_active'] = self.coherence_calc.is_pathway_active()
            
        except Exception as e:
            print(f"Coherence calculation error: {e}")
            metrics['coherence_power'] = 0.0
            metrics['coherence_fidelity'] = 0.0
            metrics['coherence_fisher'] = 0.0
            metrics['coherence_active'] = False
        
        # Total power across all pathways
        total_power = (metrics.get('casimir_power', 0) + 
                      metrics.get('dynamic_casimir_power', 0) + 
                      metrics.get('hidden_sector_power', 0) + 
                      metrics.get('axion_oscillation_power', 0) + 
                      metrics.get('axion_dark_energy_power', 0) + 
                      metrics.get('coherence_power', 0))
        
        metrics['total_power'] = total_power
        metrics['active_pathways'] = sum([
            metrics.get('casimir_active', False),
            metrics.get('dynamic_casimir_active', False),
            metrics.get('hidden_sector_active', False),
            metrics.get('axion_active', False),
            metrics.get('coherence_active', False)
        ])
        
        # LV parameters
        metrics['mu_lv'] = mu_lv
        metrics['alpha_lv'] = alpha_lv
        metrics['beta_lv'] = beta_lv
        
        return metrics
    
    def parameter_sweep_1d(self, parameter: str, n_points: int = 20) -> pd.DataFrame:
        """Perform 1D parameter sweep varying one LV parameter."""
        
        results = []
        param_values = self.parameter_ranges[parameter]
        
        # Base LV parameters
        base_params = {
            'mu_lv': 1e-18,
            'alpha_lv': 1e-15,
            'beta_lv': 1e-12
        }
        
        print(f"Starting 1D sweep of {parameter}...")
        
        for i, param_value in enumerate(param_values):
            print(f"Progress: {i+1}/{len(param_values)} ({param_value:.2e})")
            
            # Update the varying parameter
            current_params = base_params.copy()
            current_params[parameter] = param_value
            
            # Calculate metrics
            metrics = self.calculate_pathway_metrics(
                current_params['mu_lv'],
                current_params['alpha_lv'],
                current_params['beta_lv']
            )
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def parameter_sweep_2d(self, param1: str, param2: str, n_points: int = 15) -> pd.DataFrame:
        """Perform 2D parameter sweep varying two LV parameters."""
        
        results = []
        param1_values = self.parameter_ranges[param1][:n_points]
        param2_values = self.parameter_ranges[param2][:n_points]
        
        # Base LV parameters
        base_params = {
            'mu_lv': 1e-18,
            'alpha_lv': 1e-15,
            'beta_lv': 1e-12
        }
        
        print(f"Starting 2D sweep of {param1} vs {param2}...")
        
        total_combinations = len(param1_values) * len(param2_values)
        combination_count = 0
        
        for p1_val in param1_values:
            for p2_val in param2_values:
                combination_count += 1
                print(f"Progress: {combination_count}/{total_combinations}")
                
                # Update varying parameters
                current_params = base_params.copy()
                current_params[param1] = p1_val
                current_params[param2] = p2_val
                
                # Calculate metrics
                metrics = self.calculate_pathway_metrics(
                    current_params['mu_lv'],
                    current_params['alpha_lv'],
                    current_params['beta_lv']
                )
                
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def parameter_sweep_3d(self, n_points: int = 10) -> pd.DataFrame:
        """Perform 3D parameter sweep varying all LV parameters."""
        
        results = []
        mu_values = self.parameter_ranges['mu_lv'][:n_points]
        alpha_values = self.parameter_ranges['alpha_lv'][:n_points]
        beta_values = self.parameter_ranges['beta_lv'][:n_points]
        
        print("Starting 3D sweep of all LV parameters...")
        
        total_combinations = len(mu_values) * len(alpha_values) * len(beta_values)
        combination_count = 0
        
        # Use parallel processing for 3D sweep
        def calculate_single_point(params):
            mu, alpha, beta = params
            return self.calculate_pathway_metrics(mu, alpha, beta)
        
        # Generate all parameter combinations
        param_combinations = []
        for mu in mu_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    param_combinations.append((mu, alpha, beta))
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_params = {executor.submit(calculate_single_point, params): params 
                               for params in param_combinations}
            
            for future in as_completed(future_to_params):
                combination_count += 1
                print(f"Progress: {combination_count}/{total_combinations}")
                
                try:
                    metrics = future.result()
                    results.append(metrics)
                except Exception as e:
                    print(f"Error in parallel calculation: {e}")
        
        return pd.DataFrame(results)
    
    def visualize_1d_sweep(self, df: pd.DataFrame, parameter: str, save_path: Optional[str] = None):
        """Visualize 1D parameter sweep results."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        param_values = df[parameter].values
        
        # Total power across all pathways
        ax1.loglog(param_values, df['total_power'], 'k-', linewidth=3, label='Total Power')
        ax1.loglog(param_values, df['casimir_power'], 'b--', label='Casimir LV')
        ax1.loglog(param_values, df['dynamic_casimir_power'], 'r--', label='Dynamic Casimir')
        ax1.loglog(param_values, df['hidden_sector_power'], 'g--', label='Hidden Sector')
        ax1.loglog(param_values, df['axion_oscillation_power'], 'm--', label='Axion Oscillation')
        ax1.loglog(param_values, df['coherence_power'], 'c--', label='Matter-Gravity')
        ax1.axvline(self.experimental_bounds[parameter], color='red', linestyle=':', 
                   label='Experimental Bound')
        ax1.set_xlabel(f'{parameter}')
        ax1.set_ylabel('Power (W)')
        ax1.set_title('Power vs LV Parameter')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Number of active pathways
        ax2.plot(param_values, df['active_pathways'], 'ko-', linewidth=2, markersize=4)
        ax2.axvline(self.experimental_bounds[parameter], color='red', linestyle=':', 
                   label='Experimental Bound')
        ax2.set_xlabel(f'{parameter}')
        ax2.set_ylabel('Number of Active Pathways')
        ax2.set_title('Pathway Activation')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Specific pathway comparison
        ax3.loglog(param_values, df['axion_photon_rate'], 'b-', label='Axion Photon Rate')
        ax3.loglog(param_values, df['dynamic_casimir_photons'], 'r-', label='Dynamic Casimir Photons')
        ax3.loglog(param_values, df['coherence_fisher'], 'g-', label='Quantum Fisher Info')
        ax3.axvline(self.experimental_bounds[parameter], color='red', linestyle=':', 
                   label='Experimental Bound')
        ax3.set_xlabel(f'{parameter}')
        ax3.set_ylabel('Rate/Information')
        ax3.set_title('Secondary Metrics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Efficiency metrics
        efficiency_casimir = df['casimir_power'] / df['total_power'].replace(0, np.nan)
        efficiency_dynamic = df['dynamic_casimir_power'] / df['total_power'].replace(0, np.nan)
        efficiency_hidden = df['hidden_sector_power'] / df['total_power'].replace(0, np.nan)
        
        ax4.semilogx(param_values, efficiency_casimir, 'b-', label='Casimir Fraction')
        ax4.semilogx(param_values, efficiency_dynamic, 'r-', label='Dynamic Casimir Fraction')
        ax4.semilogx(param_values, efficiency_hidden, 'g-', label='Hidden Sector Fraction')
        ax4.axvline(self.experimental_bounds[parameter], color='red', linestyle=':', 
                   label='Experimental Bound')
        ax4.set_xlabel(f'{parameter}')
        ax4.set_ylabel('Power Fraction')
        ax4.set_title('Pathway Power Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_2d_sweep(self, df: pd.DataFrame, param1: str, param2: str, 
                          save_path: Optional[str] = None):
        """Visualize 2D parameter sweep results."""
        
        # Create parameter grids
        param1_unique = sorted(df[param1].unique())
        param2_unique = sorted(df[param2].unique())
        
        # Create meshgrids
        P1, P2 = np.meshgrid(param1_unique, param2_unique)
        
        # Reshape data for contour plots
        total_power = np.zeros_like(P1)
        active_pathways = np.zeros_like(P1)
        
        for i, p1 in enumerate(param1_unique):
            for j, p2 in enumerate(param2_unique):
                mask = (df[param1] == p1) & (df[param2] == p2)
                if mask.any():
                    total_power[j, i] = df[mask]['total_power'].iloc[0]
                    active_pathways[j, i] = df[mask]['active_pathways'].iloc[0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total power contour
        contour1 = ax1.contourf(np.log10(P1), np.log10(P2), np.log10(total_power + 1e-20), 
                               levels=20, cmap='viridis')
        ax1.set_xlabel(f'log10({param1})')
        ax1.set_ylabel(f'log10({param2})')
        ax1.set_title('Log10(Total Power)')
        plt.colorbar(contour1, ax=ax1)
        
        # Active pathways contour
        contour2 = ax2.contourf(np.log10(P1), np.log10(P2), active_pathways, 
                               levels=[0, 1, 2, 3, 4, 5], cmap='plasma')
        ax2.set_xlabel(f'log10({param1})')
        ax2.set_ylabel(f'log10({param2})')
        ax2.set_title('Number of Active Pathways')
        plt.colorbar(contour2, ax=ax2)
        
        # Power efficiency landscape
        casimir_power = np.zeros_like(P1)
        for i, p1 in enumerate(param1_unique):
            for j, p2 in enumerate(param2_unique):
                mask = (df[param1] == p1) & (df[param2] == p2)
                if mask.any():
                    casimir_power[j, i] = df[mask]['casimir_power'].iloc[0]
        
        contour3 = ax3.contourf(np.log10(P1), np.log10(P2), np.log10(casimir_power + 1e-20), 
                               levels=20, cmap='coolwarm')
        ax3.set_xlabel(f'log10({param1})')
        ax3.set_ylabel(f'log10({param2})')
        ax3.set_title('Log10(Casimir Power)')
        plt.colorbar(contour3, ax=ax3)
        
        # Combined pathway effectiveness
        effectiveness = total_power * active_pathways
        contour4 = ax4.contourf(np.log10(P1), np.log10(P2), np.log10(effectiveness + 1e-20), 
                               levels=20, cmap='magma')
        ax4.set_xlabel(f'log10({param1})')
        ax4.set_ylabel(f'log10({param2})')
        ax4.set_title('Log10(Power × Active Pathways)')
        plt.colorbar(contour4, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, results_1d: Dict[str, pd.DataFrame],
                                    results_2d: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive analysis report."""
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experimental_bounds': self.experimental_bounds,
            'pathways_analyzed': [
                'Casimir LV (Negative Energy)',
                'Dynamic Casimir LV (Vacuum Extraction)',
                'Hidden Sector Portal (Extra-Dimensional)',
                'Axion Coupling LV (Dark Energy)',
                'Matter-Gravity Coherence (Quantum Entanglement)'
            ],
            'sweep_summary': {}
        }
        
        # Analyze 1D sweeps
        for param, df in results_1d.items():
            # Find optimal parameter value
            optimal_idx = df['total_power'].idxmax()
            optimal_value = df.loc[optimal_idx, param]
            max_power = df.loc[optimal_idx, 'total_power']
            
            # Find activation threshold
            active_mask = df['active_pathways'] > 0
            if active_mask.any():
                activation_threshold = df.loc[active_mask, param].min()
            else:
                activation_threshold = None
            
            # Scaling analysis
            above_bound = df[df[param] > self.experimental_bounds[param]]
            if len(above_bound) > 1:
                scaling_exponent = np.polyfit(
                    np.log10(above_bound[param]), 
                    np.log10(above_bound['total_power'] + 1e-20), 
                    1
                )[0]
            else:
                scaling_exponent = None
            
            report['sweep_summary'][param] = {
                'optimal_value': optimal_value,
                'max_power': max_power,
                'activation_threshold': activation_threshold,
                'scaling_exponent': scaling_exponent,
                'experimental_bound': self.experimental_bounds[param],
                'enhancement_factor': max_power / (1e-20 if max_power == 0 else 
                                                  df['total_power'].iloc[0])
            }
        
        return report
    
    def save_results(self, results: Dict, filename: str):
        """Save sweep results to file."""
        
        # Convert DataFrames to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")

def run_comprehensive_sweep():
    """Run comprehensive parameter sweep across all pathways."""
    
    print("=== Comprehensive LV Parameter Sweep ===")
    print("Analyzing all five exotic energy pathways...")
    
    # Initialize sweep
    sweep = ComprehensiveLVSweep()
    
    # 1D sweeps for each parameter
    print("\n1. Running 1D parameter sweeps...")
    results_1d = {}
    
    for param in ['mu_lv', 'alpha_lv', 'beta_lv']:
        print(f"\nSweeping {param}...")
        df_1d = sweep.parameter_sweep_1d(param, n_points=20)
        results_1d[param] = df_1d
        
        # Visualize
        sweep.visualize_1d_sweep(df_1d, param, f'lv_sweep_1d_{param}.png')
    
    # 2D sweep (mu vs alpha)
    print("\n2. Running 2D parameter sweep (mu vs alpha)...")
    df_2d = sweep.parameter_sweep_2d('mu_lv', 'alpha_lv', n_points=15)
    sweep.visualize_2d_sweep(df_2d, 'mu_lv', 'alpha_lv', 'lv_sweep_2d_mu_alpha.png')
    
    # Generate comprehensive report
    print("\n3. Generating comprehensive report...")
    report = sweep.generate_comprehensive_report(results_1d, df_2d)
    
    # Save all results
    all_results = {
        'report': report,
        'results_1d': results_1d,
        'results_2d': df_2d
    }
    
    sweep.save_results(all_results, 'comprehensive_lv_sweep_results.json')
    
    # Print summary
    print("\n=== COMPREHENSIVE SWEEP SUMMARY ===")
    for param, summary in report['sweep_summary'].items():
        print(f"\n{param.upper()}:")
        print(f"  Optimal Value: {summary['optimal_value']:.2e}")
        print(f"  Max Power: {summary['max_power']:.2e} W")
        print(f"  Experimental Bound: {summary['experimental_bound']:.2e}")
        print(f"  Enhancement Factor: {summary['enhancement_factor']:.1f}")
        if summary['scaling_exponent']:
            print(f"  Scaling Exponent: {summary['scaling_exponent']:.2f}")
    
    print(f"\nPathways Analyzed: {len(report['pathways_analyzed'])}")
    print("All results saved to comprehensive_lv_sweep_results.json")
    
    return sweep, all_results

if __name__ == "__main__":
    sweep, results = run_comprehensive_sweep()
