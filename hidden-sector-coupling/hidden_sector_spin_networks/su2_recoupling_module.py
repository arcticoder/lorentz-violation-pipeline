"""
SU(2) Recoupling Module for Hidden-Sector Energy Transfer

This module provides high-performance computation of SU(2) recoupling coefficients
and their application to spin-network-mediated energy transfer between visible
and hidden sectors.

Author: Quantum Geometry Hidden Sector Framework
"""

import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import networkx as nx
from functools import lru_cache
import warnings

@dataclass
class SpinNetworkConfig:
    """Configuration for spin network portal parameters."""
    base_coupling: float = 1e-6  # g_0 base coupling strength
    geometric_suppression: float = 0.1  # α_geom geometric suppression scale
    portal_correlation_length: float = 1.0  # σ_portal correlation length
    max_angular_momentum: int = 5  # j_max cutoff
    network_size: int = 10  # Number of network vertices
    connectivity: float = 0.3  # Network connectivity fraction

class SU2RecouplingCalculator:
    """High-performance calculator for SU(2) recoupling coefficients."""
    
    def __init__(self, max_j: int = 10):
        """
        Initialize the calculator with maximum angular momentum.
        
        Parameters:
        -----------
        max_j : int
            Maximum angular momentum for precomputed coefficients
        """
        self.max_j = max_j
        self._cache_factorials()
    
    def _cache_factorials(self):
        """Pre-compute factorials for efficiency."""
        max_fact = 4 * self.max_j + 10
        self._log_factorials = np.array([sp.loggamma(i + 1) for i in range(max_fact)])
    
    def _log_factorial(self, n: int) -> float:
        """Fast log factorial using pre-computed values."""
        if n < len(self._log_factorials):
            return self._log_factorials[n]
        return sp.loggamma(n + 1)
    
    @lru_cache(maxsize=10000)
    def triangle_coefficient(self, j1: float, j2: float, j3: float) -> float:
        """
        Compute triangle coefficient Δ(j1,j2,j3).
        
        Parameters:
        -----------
        j1, j2, j3 : float
            Angular momentum quantum numbers
            
        Returns:
        --------
        float
            Triangle coefficient value
        """
        if not self._triangle_condition(j1, j2, j3):
            return 0.0
        
        # Convert to integers (assuming half-integer inputs are doubled)
        j1, j2, j3 = int(2*j1), int(2*j2), int(2*j3)
        
        log_delta = (self._log_factorial((j1+j2-j3)//2) +
                     self._log_factorial((j1-j2+j3)//2) +
                     self._log_factorial((-j1+j2+j3)//2) -
                     self._log_factorial((j1+j2+j3)//2 + 1))
        
        return np.exp(log_delta)
    
    def _triangle_condition(self, j1: float, j2: float, j3: float) -> bool:
        """Check triangle inequality for angular momenta."""
        return (abs(j1-j2) <= j3 <= j1+j2 and
                abs(j1-j3) <= j2 <= j1+j3 and
                abs(j2-j3) <= j1 <= j2+j3)
    
    @lru_cache(maxsize=50000)
    def wigner_3j(self, j1: float, j2: float, j3: float, 
                  m1: float, m2: float, m3: float) -> float:
        """
        Compute Wigner 3j symbol using optimized algorithm.
        
        Parameters:
        -----------
        j1, j2, j3 : float
            Angular momentum quantum numbers
        m1, m2, m3 : float
            Magnetic quantum numbers
            
        Returns:
        --------
        float
            Wigner 3j symbol value
        """
        # Check selection rules
        if not self._triangle_condition(j1, j2, j3):
            return 0.0
        if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
            return 0.0
        if abs(m1 + m2 + m3) > 1e-10:  # m1 + m2 + m3 = 0
            return 0.0
        
        # Convert to integers (double for half-integers)
        j1, j2, j3 = int(2*j1), int(2*j2), int(2*j3)
        m1, m2, m3 = int(2*m1), int(2*m2), int(2*m3)
        
        # Phase factor
        phase = (-1)**(j1 - j2 - m3) / 2
        
        # Normalization
        norm_log = (0.5 * (self._log_factorial((j1+m1)//2) +
                          self._log_factorial((j1-m1)//2) +
                          self._log_factorial((j2+m2)//2) +
                          self._log_factorial((j2-m2)//2) +
                          self._log_factorial((j3+m3)//2) +
                          self._log_factorial((j3-m3)//2)) +
                   np.log(self.triangle_coefficient(j1/2, j2/2, j3/2)))
        
        # Sum over k
        k_min = max(0, (j2-j3+m1)//2, (j1-j3-m2)//2)
        k_max = min((j1+j2-j3)//2, (j1-m1)//2, (j2+m2)//2)
        
        if k_min > k_max:
            return 0.0
        
        sum_k = 0.0
        for k in range(k_min, k_max + 1):
            try:
                log_term = (self._log_factorial(k) +
                           self._log_factorial((j1+j2-j3)//2 - k) +
                           self._log_factorial((j1-m1)//2 - k) +
                           self._log_factorial((j2+m2)//2 - k) +
                           self._log_factorial((j3-j2+m1)//2 + k) +
                           self._log_factorial((j3-j1-m2)//2 + k))
                sum_k += (-1)**k * np.exp(-log_term)
            except (ValueError, OverflowError):
                continue
        
        return phase * np.exp(norm_log) * sum_k
    
    @lru_cache(maxsize=10000)
    def wigner_6j(self, j1: float, j2: float, j3: float,
                  j4: float, j5: float, j6: float) -> float:
        """
        Compute Wigner 6j symbol {j1 j2 j3; j4 j5 j6}.
        
        Parameters:
        -----------
        j1, j2, j3, j4, j5, j6 : float
            Angular momentum quantum numbers
            
        Returns:
        --------
        float
            Wigner 6j symbol value
        """
        # Check triangle conditions
        triangles = [(j1,j2,j3), (j1,j5,j6), (j4,j2,j6), (j4,j5,j3)]
        if not all(self._triangle_condition(*tri) for tri in triangles):
            return 0.0
        
        # Convert to Racah W-coefficient
        # {j1 j2 j3; j4 j5 j6} = (-1)^(j1+j2+j4+j5) W(j1,j2,j5,j4;j3,j6)
        phase = (-1)**(j1 + j2 + j4 + j5)
        
        # Use sum formula for Racah coefficient
        # Implementation based on Varshalovich et al.
        result = self._racah_w(j1, j2, j5, j4, j3, j6)
        return phase * result
    
    def _racah_w(self, a: float, b: float, c: float, d: float, 
                 e: float, f: float) -> float:
        """Compute Racah W-coefficient."""
        # Simplified implementation - full version would use optimized summation
        # For now, use relationship to 3j symbols
        
        # This is a placeholder for the full Racah coefficient calculation
        # In practice, would implement the full sum or use library
        try:
            # Use scipy if available for cross-validation
            from sympy.physics.wigner import wigner_6j
            return float(wigner_6j(a, b, e, d, c, f))
        except ImportError:
            # Fallback to approximate calculation
            warnings.warn("Using approximate 6j calculation. Install sympy for exact values.")
            return np.exp(-(a+b+c+d+e+f)/10)  # Rough approximation

class SpinNetworkPortal:
    """Main class for modeling spin-network-mediated hidden sector energy transfer."""
    
    def __init__(self, config: SpinNetworkConfig):
        """
        Initialize spin network portal.
        
        Parameters:
        -----------
        config : SpinNetworkConfig
            Configuration parameters for the portal
        """
        self.config = config
        self.calculator = SU2RecouplingCalculator(config.max_angular_momentum)
        self.network = None
        self._generate_network()
    
    def _generate_network(self):
        """Generate random spin network topology."""
        # Create random graph
        n_nodes = self.config.network_size
        p_connect = self.config.connectivity
        
        self.network = nx.erdos_renyi_graph(n_nodes, p_connect)
        
        # Assign random angular momenta to edges
        for edge in self.network.edges():
            j = np.random.uniform(0.5, self.config.max_angular_momentum)
            self.network.edges[edge]['angular_momentum'] = j
            self.network.edges[edge]['magnetic_quantum'] = np.random.uniform(-j, j)
        
        # Assign vertex properties
        for node in self.network.nodes():
            degree = self.network.degree[node]
            self.network.nodes[node]['vertex_weight'] = np.sqrt(degree + 1)
    
    def effective_coupling(self, vertex_id: int) -> float:
        """
        Compute effective coupling constant for a vertex.
        
        Parameters:
        -----------
        vertex_id : int
            Vertex identifier
            
        Returns:
        --------
        float
            Effective coupling strength
        """
        if vertex_id not in self.network:
            return 0.0
        
        # Get adjacent edges
        edges = list(self.network.edges(vertex_id, data=True))
        if len(edges) < 3:
            return 0.0  # Need at least 3 edges for non-trivial recoupling
        
        # Calculate recoupling amplitude
        total_amplitude = 0.0
        
        # Consider all possible 3-edge combinations at this vertex
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                for k in range(j+1, len(edges)):
                    edge1, edge2, edge3 = edges[i], edges[j], edges[k]
                    
                    j1 = edge1[2]['angular_momentum']
                    j2 = edge2[2]['angular_momentum']
                    j3 = edge3[2]['angular_momentum']
                    
                    m1 = edge1[2]['magnetic_quantum']
                    m2 = edge2[2]['magnetic_quantum']
                    m3 = edge3[2]['magnetic_quantum']
                    
                    # Compute 3j symbol
                    wigner_3j = self.calculator.wigner_3j(j1, j2, j3, m1, m2, m3)
                    
                    # Add geometric suppression
                    geom_factor = np.exp(-self.config.geometric_suppression * 
                                       (j1 + j2 + j3))
                    
                    total_amplitude += wigner_3j * geom_factor
        
        return self.config.base_coupling * total_amplitude
    
    def energy_leakage_amplitude(self, initial_energy: float, 
                               final_energy: float) -> complex:
        """
        Compute energy leakage amplitude from visible to hidden sector.
        
        Parameters:
        -----------
        initial_energy : float
            Initial energy in visible sector
        final_energy : float
            Final energy in hidden sector
            
        Returns:
        --------
        complex
            Leakage amplitude
        """
        total_amplitude = 0.0 + 0j
        
        # Sum over all network paths
        for path in self._generate_paths():
            path_amplitude = 1.0
            path_phase = 0.0
            
            # Product over vertices in path
            for vertex in path:
                vertex_coupling = self.effective_coupling(vertex)
                vertex_weight = self.network.nodes[vertex]['vertex_weight']
                path_amplitude *= vertex_coupling * vertex_weight
            
            # Add path length suppression
            path_length = len(path) - 1
            path_suppression = np.exp(-path_length**2 / 
                                    (2 * self.config.portal_correlation_length**2))
            
            # Energy conservation phase
            energy_diff = initial_energy - final_energy
            path_phase += energy_diff * path_length / 100  # Arbitrary scale
            
            total_amplitude += path_amplitude * path_suppression * np.exp(1j * path_phase)
        
        return total_amplitude
    
    def _generate_paths(self, max_paths: int = 100) -> List[List[int]]:
        """Generate sample paths through the network."""
        paths = []
        nodes = list(self.network.nodes())
        
        for _ in range(min(max_paths, len(nodes)**2)):
            # Random walk path
            start = np.random.choice(nodes)
            current = start
            path = [current]
            
            for step in range(np.random.randint(2, 6)):  # 2-5 step paths
                neighbors = list(self.network.neighbors(current))
                if not neighbors:
                    break
                current = np.random.choice(neighbors)
                path.append(current)
            
            if len(path) > 1:
                paths.append(path)
        
        return paths
    
    def energy_transfer_rate(self, energy_range: Tuple[float, float],
                           hidden_density_of_states: callable) -> float:
        """
        Compute energy transfer rate to hidden sector.
        
        Parameters:
        -----------
        energy_range : Tuple[float, float]
            (min_energy, max_energy) range for integration
        hidden_density_of_states : callable
            Function ρ(E) giving density of states in hidden sector
            
        Returns:
        --------
        float
            Energy transfer rate Γ
        """
        e_min, e_max = energy_range
        n_points = 100
        energies = np.linspace(e_min, e_max, n_points)
        
        total_rate = 0.0
        
        for E_initial in energies:
            for E_final in energies:
                if E_final < E_initial:  # Energy can only decrease
                    amplitude = self.energy_leakage_amplitude(E_initial, E_final)
                    rate_contribution = (2 * np.pi / 1.0) * abs(amplitude)**2 * \
                                      hidden_density_of_states(E_final)
                    total_rate += rate_contribution
        
        # Normalize by integration step
        de = (e_max - e_min) / n_points
        return total_rate * de**2
    
    def parameter_sweep(self, param_ranges: Dict[str, Tuple[float, float]],
                       n_samples: int = 50) -> Dict[str, np.ndarray]:
        """
        Perform parameter sweep to optimize energy transfer.
        
        Parameters:
        -----------
        param_ranges : Dict[str, Tuple[float, float]]
            Parameter ranges to sweep
        n_samples : int
            Number of samples per parameter
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Results dictionary with parameter values and transfer rates
        """
        results = {param: [] for param in param_ranges.keys()}
        results['transfer_rate'] = []
        
        # Simple density of states (constant)
        def rho_hidden(E):
            return 1.0
        
        for _ in range(n_samples):
            # Sample random parameters
            for param, (min_val, max_val) in param_ranges.items():
                value = np.random.uniform(min_val, max_val)
                results[param].append(value)
                setattr(self.config, param, value)
            
            # Regenerate network with new parameters
            self._generate_network()
            
            # Compute transfer rate
            rate = self.energy_transfer_rate((0.1, 10.0), rho_hidden)
            results['transfer_rate'].append(rate)
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def visualize_network(self, save_path: Optional[str] = None):
        """Visualize the spin network topology."""
        plt.figure(figsize=(10, 8))
        
        # Position nodes
        pos = nx.spring_layout(self.network)
        
        # Draw network
        nx.draw_networkx_nodes(self.network, pos, 
                              node_color='lightblue',
                              node_size=500)
        
        # Draw edges with thickness proportional to angular momentum
        edges = self.network.edges(data=True)
        for (u, v, data) in edges:
            j = data['angular_momentum']
            nx.draw_networkx_edges(self.network, pos, 
                                 edgelist=[(u, v)],
                                 width=j/2,
                                 alpha=0.7)
        
        # Add labels        nx.draw_networkx_labels(self.network, pos)
        
        plt.title("Spin Network Portal Topology")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

@dataclass 
class LorentzViolationConfig:
    """Configuration for Lorentz-violating parameters from warp-bubble QFT."""
    mu: float = 0.0  # Polymer discretization parameter
    alpha: float = 0.0  # Einstein tensor coupling  
    beta: float = 0.0  # Ricci tensor coupling
    experimental_bounds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.experimental_bounds is None:
            self.experimental_bounds = {
                'mu': 1e-20,      # Polymer bound from LQG constraints
                'alpha': 1e-15,   # Ghost scalar coupling bound
                'beta': 1e-15     # Curvature coupling bound  
            }
    
    def pathways_active(self) -> List[str]:
        """Determine which exotic pathways are active based on LV parameter values."""
        active = []
        
        if self.mu > self.experimental_bounds['mu']:
            active.extend(['negative_energy', 'extra_dimensional', 'coherent_vacuum'])
            
        if (self.alpha > self.experimental_bounds['alpha'] or 
            self.beta > self.experimental_bounds['beta']):
            active.extend(['vacuum_harvesting', 'kinetic_suppression'])
            
        return active
    
    def lv_enhancement_factor(self) -> float:
        """Compute overall LV enhancement factor for portal coupling."""
        mu_factor = 1.0 if self.mu <= self.experimental_bounds['mu'] else (
            self.mu / self.experimental_bounds['mu']
        )
        
        alpha_factor = 1.0 if self.alpha <= self.experimental_bounds['alpha'] else (
            self.alpha / self.experimental_bounds['alpha'] 
        )
        
        beta_factor = 1.0 if self.beta <= self.experimental_bounds['beta'] else (
            self.beta / self.experimental_bounds['beta']
        )
        
        # Synergistic enhancement when multiple pathways active
        return mu_factor * alpha_factor * beta_factor

class EnhancedSpinNetworkPortal(SpinNetworkPortal):
    """Spin network portal enhanced with Lorentz violation capabilities."""
    
    def __init__(self, config: SpinNetworkConfig, lv_config: LorentzViolationConfig = None):
        """
        Initialize enhanced portal with LV capabilities.
        
        Parameters:
        -----------
        config : SpinNetworkConfig
            Basic spin network configuration
        lv_config : LorentzViolationConfig, optional
            Lorentz violation parameters. If None, uses default (no LV).
        """
        super().__init__(config)
        self.lv_config = lv_config or LorentzViolationConfig()
        
    def effective_coupling_lv(self, vertex: int) -> float:
        """Compute LV-enhanced effective coupling at a vertex."""
        base_coupling = self.effective_coupling(vertex)
        lv_enhancement = self.lv_config.lv_enhancement_factor()
        
        # Additional pathway-specific enhancements
        pathways = self.lv_config.pathways_active()
        pathway_boost = 1.0
        
        if 'negative_energy' in pathways:
            # ANEC violation enhances coupling strength
            pathway_boost *= (1 + abs(self.lv_config.mu) * 1e18)
            
        if 'vacuum_harvesting' in pathways:
            # Ghost scalar provides additional channels
            pathway_boost *= (1 + self.lv_config.alpha * 1e13 + self.lv_config.beta * 1e13)
            
        return base_coupling * lv_enhancement * pathway_boost
    
    def energy_leakage_amplitude_lv(self, E_initial: float, E_final: float) -> complex:
        """Compute LV-enhanced energy leakage amplitude."""
        base_amplitude = self.energy_leakage_amplitude(E_initial, E_final)
        
        pathways = self.lv_config.pathways_active()
        lv_amplitude = 0.0
        
        # Contribution from each active pathway
        if 'negative_energy' in pathways:
            # Direct negative energy extraction
            neg_energy_amp = self.lv_config.mu * (E_initial - E_final) * 1e15
            lv_amplitude += neg_energy_amp * np.exp(1j * np.pi/4)
            
        if 'vacuum_harvesting' in pathways:
            # Ghost scalar mediated extraction
            ghost_amp = (self.lv_config.alpha + self.lv_config.beta) * np.sqrt(E_initial * E_final) * 1e12
            lv_amplitude += ghost_amp * np.exp(1j * np.pi/3)
            
        if 'extra_dimensional' in pathways:
            # Extra-dimensional polymer channels
            polymer_amp = self.lv_config.mu * np.log(E_initial/E_final) * 1e14
            lv_amplitude += polymer_amp * np.exp(1j * np.pi/6)
            
        if 'coherent_vacuum' in pathways:
            # Modified quantum inequality effects
            sinc_factor = np.sinc(np.pi * self.lv_config.mu * 1e20)
            coherent_amp = (1.0 / sinc_factor - 1.0) * np.sqrt(E_initial) * 1e10
            lv_amplitude += coherent_amp * np.exp(1j * np.pi/2)
            
        if 'kinetic_suppression' in pathways:
            # Reduced kinetic energy costs
            suppression_factor = 1.0 / (1 + (self.lv_config.alpha + self.lv_config.beta) * 1e15)
            kinetic_amp = base_amplitude * (1 - suppression_factor)
            lv_amplitude += kinetic_amp
        
        return base_amplitude + lv_amplitude
    
    def exotic_pathway_summary(self) -> Dict[str, float]:
        """Generate summary of exotic pathway activations and strengths."""
        pathways = self.lv_config.pathways_active()
        
        summary = {
            'active_pathways': pathways,
            'total_enhancement': self.lv_config.lv_enhancement_factor(),
            'pathway_count': len(pathways)
        }
        
        # Individual pathway strengths
        if 'negative_energy' in pathways:
            summary['negative_energy_strength'] = self.lv_config.mu / self.lv_config.experimental_bounds['mu']
            
        if 'vacuum_harvesting' in pathways:
            summary['vacuum_harvest_strength'] = max(
                self.lv_config.alpha / self.lv_config.experimental_bounds['alpha'],
                self.lv_config.beta / self.lv_config.experimental_bounds['beta']
            )
            
        return summary

def demo_lorentz_violation():
    """Demonstrate Lorentz violation enhanced portal capabilities."""
    print("=== Lorentz Violation Enhanced Portal Demo ===")
    
    # Standard configuration
    config = SpinNetworkConfig(
        base_coupling=1e-6,
        geometric_suppression=0.1,
        network_size=8
    )
    
    # LV configuration with parameters beyond bounds
    lv_config = LorentzViolationConfig(
        mu=1e-18,      # 100x beyond bound
        alpha=1e-13,   # 100x beyond bound  
        beta=5e-14,    # 50x beyond bound
    )
    
    # Create enhanced portal
    enhanced_portal = EnhancedSpinNetworkPortal(config, lv_config)
    
    print(f"Active pathways: {lv_config.pathways_active()}")
    print(f"LV enhancement factor: {lv_config.lv_enhancement_factor():.2e}")
    
    # Compare standard vs LV-enhanced coupling
    vertex = 0
    standard_coupling = enhanced_portal.effective_coupling(vertex)
    lv_coupling = enhanced_portal.effective_coupling_lv(vertex)
    
    print(f"\nCoupling at vertex {vertex}:")
    print(f"  Standard: {standard_coupling:.2e}")
    print(f"  LV-enhanced: {lv_coupling:.2e}")
    print(f"  Enhancement ratio: {lv_coupling/standard_coupling:.2e}")
    
    # Compare energy leakage amplitudes
    standard_amp = enhanced_portal.energy_leakage_amplitude(10.0, 8.0)
    lv_amp = enhanced_portal.energy_leakage_amplitude_lv(10.0, 8.0)
    
    print(f"\nEnergy leakage amplitude (10→8):")
    print(f"  Standard: {abs(standard_amp):.2e}")
    print(f"  LV-enhanced: {abs(lv_amp):.2e}")
    print(f"  Enhancement ratio: {abs(lv_amp)/abs(standard_amp):.2e}")
    
    # Pathway summary
    summary = enhanced_portal.exotic_pathway_summary()
    print(f"\nExotic pathway summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

def demo_su2_recoupling():
    """Demonstration of SU(2) recoupling calculations."""
    print("SU(2) Recoupling Coefficient Demo")
    print("="*40)
    
    calc = SU2RecouplingCalculator()
    
    # Test 3j symbols
    print("\nWigner 3j symbols:")
    test_cases = [
        (1, 1, 0, 1, -1, 0),
        (1, 1, 1, 1, 0, -1),
        (2, 1, 1, 0, 1, -1)
    ]
    
    for j1, j2, j3, m1, m2, m3 in test_cases:
        wigner_3j = calc.wigner_3j(j1, j2, j3, m1, m2, m3)
        print(f"({j1} {j2} {j3}; {m1} {m2} {m3}) = {wigner_3j:.6f}")
    
    # Test 6j symbols
    print("\nWigner 6j symbols:")
    test_6j = [
        (1, 1, 1, 1, 1, 1),
        (2, 1, 1, 1, 2, 1)
    ]
    
    for j1, j2, j3, j4, j5, j6 in test_6j:
        wigner_6j = calc.wigner_6j(j1, j2, j3, j4, j5, j6)
        print(f"{{{j1} {j2} {j3}; {j4} {j5} {j6}}} = {wigner_6j:.6f}")

def demo_energy_transfer():
    """Demonstration of energy transfer calculations."""
    print("\nEnergy Transfer Portal Demo")
    print("="*40)
    
    # Create configuration
    config = SpinNetworkConfig(
        base_coupling=1e-5,
        geometric_suppression=0.05,
        portal_correlation_length=2.0,
        max_angular_momentum=3,
        network_size=8
    )
    
    # Initialize portal
    portal = SpinNetworkPortal(config)
    
    # Test effective coupling
    print(f"\nNetwork has {portal.network.number_of_nodes()} nodes and "
          f"{portal.network.number_of_edges()} edges")
    
    for vertex in list(portal.network.nodes())[:3]:
        coupling = portal.effective_coupling(vertex)
        print(f"Effective coupling at vertex {vertex}: {coupling:.2e}")
    
    # Test energy leakage
    amplitude = portal.energy_leakage_amplitude(10.0, 8.0)
    print(f"\nEnergy leakage amplitude (10→8): {abs(amplitude):.2e}")
    
    # Simple density of states
    def rho_hidden(E):
        return E**2 / 100  # Quadratic density
    
    # Compute transfer rate
    rate = portal.energy_transfer_rate((1.0, 5.0), rho_hidden)
    print(f"Energy transfer rate: {rate:.2e}")

if __name__ == "__main__":
    demo_su2_recoupling()
    demo_energy_transfer()
    demo_lorentz_violation()
