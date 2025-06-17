"""
GUT-Enhanced Hidden Sector Coupling Module

This module integrates Grand Unified Theory (GUT) polymerization with hidden sector
energy extraction mechanisms, providing GUT-scale symmetry breaking, parameter
scanning, and enhanced dispersion relations for energy extraction beyond E=mc².

Adapted from the GUT Unified Polymerization Framework for hidden sector applications.
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy.special import hyp2f1
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class GUTHiddenConfig:
    """Configuration for GUT-enhanced hidden sector coupling."""
    
    # GUT parameters
    gut_group: str = "SU(5)"  # Options: "SU(5)", "SO(10)", "E6"
    unification_scale: float = 2e16  # GeV
    polymer_scale: float = 1e-18  # GeV^-1 (Planck scale vicinity)
    
    # Hidden sector parameters
    hidden_coupling: float = 1e-5  # Portal coupling strength
    hidden_mass_scale: float = 1e3  # GeV (hidden gauge boson masses)
    portal_lambda: float = 1e-6  # Scalar portal coupling
    
    # LV parameters
    c_tensor_magnitude: float = 1e-16  # LV tensor coefficient magnitude
    dispersion_order: int = 3  # Leading LV dispersion order
    
    # Laboratory parameters
    lab_energy: float = 1.0  # GeV
    lab_magnetic_field: float = 10.0  # Tesla
    
    # Numerical parameters
    n_scan_points: int = 100
    optimization_method: str = "differential_evolution"

class GUTHiddenSectorCoupling:
    """
    Main class implementing GUT-enhanced hidden sector energy extraction.
    
    This class combines:
    - GUT symmetry breaking for hidden gauge boson production
    - Polymerized propagators with LV modifications
    - Parameter space scanning and optimization
    - Constraint satisfaction analysis
    """
    
    def __init__(self, config: GUTHiddenConfig):
        """Initialize with specified configuration."""
        self.config = config
        self._setup_gut_data()
        self._setup_hidden_sector()
        self._setup_constraints()
        
        # Results storage
        self.scan_results = {}
        self.optimal_parameters = {}
        
    def _setup_gut_data(self):
        """Initialize GUT group theory data."""
        gut_data = {
            "SU(5)": {
                "dimension": 24,
                "rank": 4,
                "hidden_bosons": 12,
                "casimir": 5,
                "beta_coefficient": 16.17,
                "enhancement_factor": 1.0
            },
            "SO(10)": {
                "dimension": 45,
                "rank": 5,
                "hidden_bosons": 33,
                "casimir": 8,
                "beta_coefficient": 27.17,
                "enhancement_factor": 2.75
            },
            "E6": {
                "dimension": 78,
                "rank": 6,
                "hidden_bosons": 65,
                "casimir": 12,
                "beta_coefficient": 41.83,
                "enhancement_factor": 6.5
            }
        }
        
        if self.config.gut_group not in gut_data:
            raise ValueError(f"Unsupported GUT group: {self.config.gut_group}")
            
        self.gut_data = gut_data[self.config.gut_group]
        
    def _setup_hidden_sector(self):
        """Initialize hidden sector parameters."""
        self.hidden_sector = {
            "n_hidden_bosons": self.gut_data["hidden_bosons"],
            "portal_couplings": np.random.uniform(
                self.config.hidden_coupling * 0.1,
                self.config.hidden_coupling * 10,
                self.gut_data["hidden_bosons"]
            ),
            "hidden_masses": np.logspace(
                np.log10(self.config.hidden_mass_scale * 0.1),
                np.log10(self.config.hidden_mass_scale * 10),
                self.gut_data["hidden_bosons"]
            )
        }
        
    def _setup_constraints(self):
        """Initialize experimental and theoretical constraints."""
        self.constraints = {
            # LIV bounds from SME (Standard Model Extension)
            "lv_bounds": {
                "c_00": 4e-8,    # Clock comparison
                "c_11": 2e-16,   # Michelson-Morley
                "c_12": 3e-11    # Hughes-Drever
            },
            
            # GUT phenomenology bounds
            "gut_bounds": {
                "proton_lifetime": 1.6e34,  # years
                "gauge_unification": 0.0003,  # relative precision
                "neutrino_mass_sum": 0.12   # eV
            },
            
            # Laboratory constraints
            "lab_bounds": {
                "max_energy": 1e3,    # GeV (achievable)
                "max_field": 100,     # Tesla
                "min_detection": 1e-12  # Hz (rate sensitivity)
            }
        }
        
    def polymerized_propagator(self, momentum: float, mass: float = 0.0) -> complex:
        """
        Calculate polymerized GUT gauge boson propagator.
        
        Args:
            momentum: Four-momentum magnitude (GeV)
            mass: Gauge boson mass (GeV)
            
        Returns:
            Complex propagator value with polymer corrections
        """
        k_squared = momentum**2 + mass**2
        
        # Polymer modification
        mu_arg = self.config.polymer_scale * np.sqrt(k_squared)
        
        if mu_arg == 0:
            sinc_factor = 1.0
        else:
            sinc_factor = np.sin(mu_arg) / mu_arg
            
        # Full propagator with gauge structure
        propagator = sinc_factor**2 / (k_squared + 1e-12)  # Small regularization
        
        return propagator
        
    def vertex_form_factor(self, momenta: List[float]) -> complex:
        """
        Calculate polymerized vertex form factor for n-point interaction.
        
        Args:
            momenta: List of external momentum magnitudes
            
        Returns:
            Complex form factor with polymer corrections
        """
        form_factor = 1.0
        
        for p in momenta:
            mu_arg = self.config.polymer_scale * p
            if mu_arg == 0:
                sinc_val = 1.0
            else:
                sinc_val = np.sin(mu_arg) / mu_arg
            form_factor *= sinc_val
            
        return form_factor
        
    def lv_dispersion_relation(self, energy: float, momentum: float) -> float:
        """
        Calculate Lorentz-violating dispersion relation.
        
        Args:
            energy: Particle energy (GeV)
            momentum: Particle momentum (GeV)
            
        Returns:
            Modified energy with LV corrections
        """
        # Standard dispersion
        e_standard = np.sqrt(momentum**2)  # Assuming massless for simplicity
        
        # LV corrections
        lv_correction = 0.0
        
        if self.config.dispersion_order >= 3:
            lv_correction += self.config.c_tensor_magnitude * energy**3 / self.config.unification_scale
            
        if self.config.dispersion_order >= 4:
            lv_correction += self.config.c_tensor_magnitude * energy**4 / self.config.unification_scale**2
            
        # Polymer modification
        mu_arg = self.config.polymer_scale * energy
        if mu_arg != 0:
            polymer_factor = (np.sin(mu_arg) / mu_arg)**2
            lv_correction *= polymer_factor
            
        return e_standard + lv_correction
        
    def extraction_rate(self, energy: float, magnetic_field: float = 0.0) -> float:
        """
        Calculate energy extraction rate from hidden sector.
        
        Args:
            energy: Laboratory energy scale (GeV)
            magnetic_field: Applied magnetic field (Tesla)
            
        Returns:
            Extraction rate (Hz)
        """
        total_rate = 0.0
        
        # Sum over all hidden gauge bosons
        for i in range(self.hidden_sector["n_hidden_bosons"]):
            g_h = self.hidden_sector["portal_couplings"][i]
            m_h = self.hidden_sector["hidden_masses"][i]
            
            # Basic rate formula
            if m_h > 0:
                rate_i = (g_h**2 / (16 * np.pi)) * (energy**2 / m_h**2)
            else:
                rate_i = 0.0
                
            # Polymer enhancement
            propagator = self.polymerized_propagator(energy, m_h)
            rate_i *= abs(propagator)**2
            
            # Vertex corrections (4-point interaction)
            vertex_factor = abs(self.vertex_form_factor([energy] * 4))**2
            rate_i *= vertex_factor
            
            # Magnetic field enhancement (if applicable)
            if magnetic_field > 0:
                # Simple B² enhancement for electromagnetic portals
                b_enhancement = 1 + (magnetic_field / 1e3)**2
                rate_i *= b_enhancement
                
            total_rate += rate_i
            
        # GUT enhancement factor
        total_rate *= self.gut_data["enhancement_factor"]
        
        return total_rate
        
    def constraint_satisfaction(self, parameters: Dict[str, float]) -> float:
        """
        Evaluate constraint satisfaction for parameter set.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Constraint satisfaction score (0 = violated, 1 = satisfied)
        """
        score = 1.0
        
        # LIV constraints
        c_mag = parameters.get("c_tensor", self.config.c_tensor_magnitude)
        if c_mag > self.constraints["lv_bounds"]["c_11"]:
            score *= 0.1  # Strong penalty for LIV violation
            
        # Energy conservation
        extraction_rate = parameters.get("extraction_rate", 0.0)
        if extraction_rate > 1e6:  # Unreasonably high rate
            score *= 0.01
            
        # Laboratory feasibility
        lab_energy = parameters.get("lab_energy", self.config.lab_energy)
        if lab_energy > self.constraints["lab_bounds"]["max_energy"]:
            score *= 0.5
            
        # GUT scale consistency
        gut_scale = parameters.get("gut_scale", self.config.unification_scale)
        if gut_scale < 1e15 or gut_scale > 1e18:
            score *= 0.3
            
        return score
        
    def figure_of_merit(self, parameters: Dict[str, float]) -> float:
        """
        Calculate figure of merit for parameter optimization.
        
        Args:
            parameters: Parameter values to evaluate
            
        Returns:
            Figure of merit (higher = better)
        """
        # Extract energy if provided, otherwise use config
        energy = parameters.get("energy", self.config.lab_energy)
        field = parameters.get("magnetic_field", self.config.lab_magnetic_field)
        
        # Calculate extraction rate
        rate = self.extraction_rate(energy, field)
        
        # Constraint satisfaction
        constraint_score = self.constraint_satisfaction(parameters)
        
        # Detection probability (simplified)
        detection_prob = min(1.0, rate / self.constraints["lab_bounds"]["min_detection"])
        
        # Overall figure of merit
        fom = rate * constraint_score * detection_prob
        
        # Normalization (optional)
        fom = fom / 1e-6  # Scale to reasonable range
        
        return fom
        
    def parameter_scan_2d(self, param1: str, param2: str, 
                         ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Perform 2D parameter space scan.
        
        Args:
            param1, param2: Parameter names to scan
            ranges: Dictionary of parameter ranges
            
        Returns:
            Dictionary containing scan results
        """
        n_points = self.config.n_scan_points
        
        # Create parameter grids
        p1_vals = np.logspace(
            np.log10(ranges[param1][0]),
            np.log10(ranges[param1][1]),
            n_points
        )
        p2_vals = np.logspace(
            np.log10(ranges[param2][0]),
            np.log10(ranges[param2][1]),
            n_points
        )
        
        P1, P2 = np.meshgrid(p1_vals, p2_vals)
        FOM = np.zeros_like(P1)
        
        # Evaluate figure of merit at each grid point
        for i in range(n_points):
            for j in range(n_points):
                params = {param1: P1[i, j], param2: P2[i, j]}
                
                # Add other parameters at default values
                if param1 != "energy":
                    params["energy"] = self.config.lab_energy
                if param2 != "energy":
                    params["energy"] = self.config.lab_energy
                if param1 != "magnetic_field":
                    params["magnetic_field"] = self.config.lab_magnetic_field
                if param2 != "magnetic_field":
                    params["magnetic_field"] = self.config.lab_magnetic_field
                    
                FOM[i, j] = self.figure_of_merit(params)
                
        # Store results
        results = {
            "param1": param1,
            "param2": param2,
            "param1_values": p1_vals,
            "param2_values": p2_vals,
            "grid_param1": P1,
            "grid_param2": P2,
            "figure_of_merit": FOM,
            "max_fom": np.max(FOM),
            "optimal_indices": np.unravel_index(np.argmax(FOM), FOM.shape),
            "optimal_param1": P1[np.unravel_index(np.argmax(FOM), FOM.shape)],
            "optimal_param2": P2[np.unravel_index(np.argmax(FOM), FOM.shape)]
        }
        
        return results
        
    def optimize_parameters(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Optimize parameters for maximum figure of merit.
        
        Args:
            parameter_bounds: Dictionary of parameter bounds for optimization
            
        Returns:
            Optimization results
        """
        def objective(x):
            # Convert parameter array to dictionary
            param_names = list(parameter_bounds.keys())
            params = {name: val for name, val in zip(param_names, x)}
            
            # Return negative FOM for minimization
            return -self.figure_of_merit(params)
            
        # Bounds for optimization
        bounds = [parameter_bounds[name] for name in parameter_bounds.keys()]
        
        if self.config.optimization_method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=300,
                popsize=15
            )
        else:
            # Use scipy.optimize.minimize as fallback
            x0 = [(b[0] + b[1]) / 2 for b in bounds]  # Mid-point initial guess
            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds
            )
            
        # Extract optimal parameters
        param_names = list(parameter_bounds.keys())
        optimal_params = {name: val for name, val in zip(param_names, result.x)}
        
        optimization_results = {
            "success": result.success,
            "optimal_parameters": optimal_params,
            "optimal_fom": -result.fun,  # Convert back from negative
            "n_evaluations": result.nfev if hasattr(result, 'nfev') else None,
            "optimization_result": result
        }
        
        return optimization_results
        
    def gut_enhanced_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive GUT-enhanced hidden sector analysis.
        
        Returns:
            Complete analysis results
        """
        results = {
            "gut_group": self.config.gut_group,
            "gut_data": self.gut_data,
            "config": self.config.__dict__
        }
        
        # Energy range analysis
        energies = np.logspace(-3, 3, 100)  # 1 meV to 1 TeV
        rates = [self.extraction_rate(e) for e in energies]
        
        results["energy_analysis"] = {
            "energies": energies,
            "extraction_rates": rates,
            "optimal_energy": energies[np.argmax(rates)],
            "max_rate": np.max(rates)
        }
        
        # Parameter sensitivity analysis
        base_params = {
            "energy": self.config.lab_energy,
            "magnetic_field": self.config.lab_magnetic_field
        }
        
        sensitivity = {}
        for param in ["energy", "magnetic_field"]:
            base_fom = self.figure_of_merit(base_params)
            
            # 10% parameter variation
            params_plus = base_params.copy()
            params_plus[param] *= 1.1
            fom_plus = self.figure_of_merit(params_plus)
            
            params_minus = base_params.copy()
            params_minus[param] *= 0.9
            fom_minus = self.figure_of_merit(params_minus)
            
            # Numerical derivative
            sensitivity[param] = (fom_plus - fom_minus) / (0.2 * base_params[param])
            
        results["sensitivity_analysis"] = sensitivity
        
        # Constraint analysis
        constraint_score = self.constraint_satisfaction(base_params)
        results["constraint_satisfaction"] = constraint_score
        
        return results
        
    def plot_parameter_scan(self, scan_results: Dict[str, Any], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D parameter scan results.
        
        Args:
            scan_results: Results from parameter_scan_2d
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create contour plot
        contour = ax.contourf(
            scan_results["grid_param1"],
            scan_results["grid_param2"],
            scan_results["figure_of_merit"],
            levels=20,
            cmap="viridis"
        )
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Figure of Merit", fontsize=12)
        
        # Mark optimal point
        ax.plot(
            scan_results["optimal_param1"],
            scan_results["optimal_param2"],
            "r*",
            markersize=15,
            label=f"Optimal: ({scan_results['optimal_param1']:.2e}, {scan_results['optimal_param2']:.2e})"
        )
        
        # Labels and formatting
        ax.set_xlabel(scan_results["param1"], fontsize=12)
        ax.set_ylabel(scan_results["param2"], fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        title = f"Parameter Scan: {self.config.gut_group} Hidden Sector\n"
        title += f"Max FOM: {scan_results['max_fom']:.2e}"
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
        return fig

def demonstrate_gut_hidden_coupling():
    """Demonstrate GUT-enhanced hidden sector coupling capabilities."""
    print("=== GUT-Enhanced Hidden Sector Coupling Demonstration ===\n")
    
    # Test different GUT groups
    gut_groups = ["SU(5)", "SO(10)", "E6"]
    
    for gut_group in gut_groups:
        print(f"--- {gut_group} Analysis ---")
        
        # Configuration
        config = GUTHiddenConfig(
            gut_group=gut_group,
            polymer_scale=1e-18,
            hidden_coupling=1e-5,
            lab_energy=1.0,
            lab_magnetic_field=10.0
        )
        
        # Initialize coupling system
        gut_coupling = GUTHiddenSectorCoupling(config)
        
        # Comprehensive analysis
        analysis = gut_coupling.gut_enhanced_analysis()
        
        print(f"  Hidden gauge bosons: {analysis['gut_data']['hidden_bosons']}")
        print(f"  Enhancement factor: {analysis['gut_data']['enhancement_factor']:.2f}")
        print(f"  Optimal energy: {analysis['energy_analysis']['optimal_energy']:.3f} GeV")
        print(f"  Max extraction rate: {analysis['energy_analysis']['max_rate']:.2e} Hz")
        print(f"  Constraint satisfaction: {analysis['constraint_satisfaction']:.3f}")
        print()
        
    # Parameter optimization example
    print("--- Parameter Optimization Example ---")
    config = GUTHiddenConfig(gut_group="E6", n_scan_points=50)
    gut_coupling = GUTHiddenSectorCoupling(config)
    
    # Define parameter bounds
    bounds = {
        "energy": (0.1, 100.0),      # GeV
        "magnetic_field": (1.0, 50.0) # Tesla
    }
    
    # Optimize
    opt_results = gut_coupling.optimize_parameters(bounds)
    
    if opt_results["success"]:
        print(f"Optimization successful!")
        print(f"Optimal energy: {opt_results['optimal_parameters']['energy']:.3f} GeV")
        print(f"Optimal field: {opt_results['optimal_parameters']['magnetic_field']:.1f} T")
        print(f"Optimal FOM: {opt_results['optimal_fom']:.2e}")
    else:
        print("Optimization failed")
        
    print("\n=== Demonstration Complete ===")

if __name__ == "__main__":
    demonstrate_gut_hidden_coupling()
