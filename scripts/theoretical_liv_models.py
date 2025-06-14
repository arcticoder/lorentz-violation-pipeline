#!/usr/bin/env python3
"""
Advanced Theoretical LIV Models

This module implements sophisticated Lorentz Invariance Violation models including:
- Polymer-quantized QED with modified dispersion relations
- Gravity-rainbow models with f(k/E_Pl) factors
- Vacuum instability calculations (Schwinger-like rates)
- Hidden sector energy leakage mechanisms
- Polynomial dispersion expansions

Key dispersion relations implemented:
1. Standard: ω² = k² + m²
2. Polymer-QED: ω² = k²[1 + α(k/E_Pl) + β(k/E_Pl)²] + m²
3. Gravity-Rainbow: ω² = k²f(k/E_Pl) + m²g(k/E_Pl)
4. Doubly Special Relativity: ω² = k²h(k,E_Pl) + m²

References:
- Amelino-Camelia, Living Rev. Rel. 16, 5 (2013)
- Gambini & Pullin, Phys. Rev. D 59, 124021 (1999)
- Magueijo & Albrecht, Phys. Rev. Lett. 86, 2464 (2001)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import factorial
import matplotlib.pyplot as plt

# Physical constants
E_PLANCK = 1.22e19  # GeV (Planck energy)
ALPHA_EM = 1/137.036  # Fine structure constant
ELECTRON_MASS = 0.511e-3  # GeV
LIGHT_SPEED = 2.998e8  # m/s

class DispersionRelation:
    """Base class for modified dispersion relations."""
    
    def __init__(self, name="Standard"):
        self.name = name
        self.parameters = {}
    
    def omega_squared(self, k, m=0):
        """Return ω² for given momentum k and mass m."""
        raise NotImplementedError
    
    def group_velocity(self, E, m=0):
        """Calculate group velocity dω/dk."""
        raise NotImplementedError
    
    def time_delay(self, E1, E2, distance, m=0):
        """Calculate time delay between photons of energies E1, E2."""
        v1 = self.group_velocity(E1, m)
        v2 = self.group_velocity(E2, m)
        return distance * (1/v1 - 1/v2)

class StandardDispersion(DispersionRelation):
    """Standard relativistic dispersion: ω² = k² + m²."""
    
    def __init__(self):
        super().__init__("Standard")
    
    def omega_squared(self, k, m=0):
        return k**2 + m**2
    
    def group_velocity(self, E, m=0):
        return LIGHT_SPEED * E / np.sqrt(E**2 + m**2)

class PolymerQEDDispersion(DispersionRelation):
    """
    Polymer-quantized QED dispersion relation.
    
    ω² = k²[1 + α₁(k/E_Pl) + α₂(k/E_Pl)² + ...] + m²
    
    This arises from loop quantum gravity and polymer field theory
    where space-time has a discrete structure at the Planck scale.
    """
    
    def __init__(self, alpha1=0, alpha2=0, alpha3=0):
        super().__init__("Polymer-QED")
        self.parameters = {
            'alpha1': alpha1,  # Linear correction
            'alpha2': alpha2,  # Quadratic correction  
            'alpha3': alpha3   # Cubic correction
        }
    
    def polymer_factor(self, k):
        """Calculate the polymer correction factor f(k/E_Pl)."""
        x = k / E_PLANCK
        return (1 + self.parameters['alpha1'] * x + 
                self.parameters['alpha2'] * x**2 + 
                self.parameters['alpha3'] * x**3)
    
    def omega_squared(self, k, m=0):
        return k**2 * self.polymer_factor(k) + m**2
    
    def group_velocity(self, E, m=0):
        """Approximate group velocity for massless case."""
        if m == 0:
            k = E  # For photons: E = k
            polymer_f = self.polymer_factor(k)
            # v_g ≈ c[1 + (1/2)α₁(E/E_Pl) + ...]
            correction = (0.5 * self.parameters['alpha1'] * E / E_PLANCK +
                         self.parameters['alpha2'] * (E / E_PLANCK)**2)
            return LIGHT_SPEED * (1 + correction)
        else:
            # Full calculation for massive particles
            k = np.sqrt(E**2 - m**2)
            omega = np.sqrt(self.omega_squared(k, m))
            # Numerical derivative dω/dk
            dk = k * 1e-6
            omega_plus = np.sqrt(self.omega_squared(k + dk, m))
            return LIGHT_SPEED * (omega_plus - omega) / dk

class GravityRainbowDispersion(DispersionRelation):
    """
    Gravity-rainbow dispersion relation.
    
    ω² = k²f(k/E_Pl) + m²g(k/E_Pl)
    
    Different particle species see different geometries.
    Common forms:
    - f(x) = (1 - x)^n
    - g(x) = (1 - x)^m
    """
    
    def __init__(self, n=1, m=1, eta=1.0):
        super().__init__("Gravity-Rainbow")
        self.parameters = {
            'n': n,      # Momentum function exponent
            'm': m,      # Mass function exponent
            'eta': eta   # Overall scale factor
        }
    
    def f_function(self, k):
        """Rainbow function for kinetic term."""
        x = self.parameters['eta'] * k / E_PLANCK
        return (1 - x)**self.parameters['n']
    
    def g_function(self, k):
        """Rainbow function for mass term."""
        x = self.parameters['eta'] * k / E_PLANCK
        return (1 - x)**self.parameters['m']
    
    def omega_squared(self, k, m=0):
        return k**2 * self.f_function(k) + m**2 * self.g_function(k)
    
    def group_velocity(self, E, m=0):
        """Group velocity with rainbow corrections."""
        if m == 0:
            # For photons
            x = self.parameters['eta'] * E / E_PLANCK
            f_val = (1 - x)**self.parameters['n']
            # v_g = c * sqrt(f) * [1 + (n/2)x/(1-x)]
            velocity_factor = np.sqrt(f_val) * (1 + self.parameters['n'] * x / (2 * (1 - x)))
            return LIGHT_SPEED * velocity_factor
        else:
            # Numerical calculation for massive particles
            k = np.sqrt(E**2 - m**2)
            omega = np.sqrt(self.omega_squared(k, m))
            dk = k * 1e-6
            omega_plus = np.sqrt(self.omega_squared(k + dk, m))
            return LIGHT_SPEED * (omega_plus - omega) / dk

class VacuumInstability:
    """
    Calculate vacuum instability rates in modified QED.
    
    In LIV theories, the vacuum can become unstable to pair production
    even in the absence of external fields due to modified dispersion relations.
    """
    
    def __init__(self, dispersion_model):
        self.dispersion = dispersion_model
    
    def schwinger_rate(self, field_strength, particle_mass=ELECTRON_MASS):
        """
        Calculate Schwinger pair production rate with LIV corrections.
        
        Standard rate: Γ ~ (eE)²/(4π³) * exp(-πm²/eE)
        LIV modifications can dramatically alter the exponential factor.
        """
        if field_strength <= 0:
            return 0.0
        
        # Critical field strength
        E_crit = particle_mass**2 / ALPHA_EM
        
        # Standard Schwinger formula
        prefactor = (ALPHA_EM * field_strength**2) / (4 * np.pi**3)
        exponential = np.exp(-np.pi * particle_mass**2 / (ALPHA_EM * field_strength))
        
        # LIV correction factor (model-dependent)
        if isinstance(self.dispersion, PolymerQEDDispersion):
            # Polymer corrections modify the effective mass
            alpha1 = self.dispersion.parameters['alpha1']
            correction = 1 + alpha1 * particle_mass / E_PLANCK
            exponential *= np.exp(-np.pi * correction)
        
        return prefactor * exponential
    
    def vacuum_decay_rate(self, energy_density):
        """
        Calculate vacuum decay rate in high-energy environments.
        
        LIV can trigger vacuum instabilities when energy densities
        approach the Planck scale.
        """
        if energy_density < E_PLANCK**4:
            return 0.0
        
        # Dimensional analysis: Γ ~ (ρ/E_Pl⁴)^(3/2) * E_Pl
        return (energy_density / E_PLANCK**4)**(3/2) * E_PLANCK

class HiddenSectorInteraction:
    """
    Model energy leakage into hidden sectors via LIV interactions.
    
    In some LIV models, high-energy particles can transfer energy
    to hidden sector fields, affecting cosmic ray propagation.
    """
    
    def __init__(self, coupling_strength=1e-6, hidden_mass=1e-3):
        self.g_coupling = coupling_strength  # Coupling to hidden sector
        self.m_hidden = hidden_mass         # Hidden particle mass (GeV)
    
    def energy_loss_rate(self, particle_energy, particle_mass=0):
        """
        Calculate energy loss rate due to hidden sector interactions.
        
        dE/dt ~ g² * E² / (16π) for E >> m_hidden
        """
        if particle_energy < self.m_hidden:
            return 0.0
        
        # Dimensional analysis for energy loss
        loss_rate = (self.g_coupling**2 * particle_energy**2) / (16 * np.pi)
        
        # Kinematic suppression for massive particles
        if particle_mass > 0:
            velocity = particle_energy / np.sqrt(particle_energy**2 + particle_mass**2)
            loss_rate *= velocity
        
        return loss_rate
    
    def propagation_length(self, initial_energy, final_energy, particle_mass=0):
        """
        Calculate propagation distance before energy drops to final_energy.
        """
        if initial_energy <= final_energy:
            return float('inf')
        
        # Integrate dE/dx = -dE/dt / c
        # For simplicity, assume constant loss rate
        avg_energy = (initial_energy + final_energy) / 2
        avg_loss_rate = self.energy_loss_rate(avg_energy, particle_mass)
        
        if avg_loss_rate == 0:
            return float('inf')
        
        energy_lost = initial_energy - final_energy
        time_taken = energy_lost / avg_loss_rate
        
        return LIGHT_SPEED * time_taken

def create_theoretical_models():
    """Create a suite of theoretical LIV models for testing."""
    
    models = {
        'standard': StandardDispersion(),
        
        'polymer_linear': PolymerQEDDispersion(alpha1=1.0, alpha2=0, alpha3=0),
        'polymer_quadratic': PolymerQEDDispersion(alpha1=0, alpha2=1.0, alpha3=0),
        'polymer_full': PolymerQEDDispersion(alpha1=0.5, alpha2=0.1, alpha3=0.01),
        
        'rainbow_linear': GravityRainbowDispersion(n=1, m=1, eta=1.0),
        'rainbow_quadratic': GravityRainbowDispersion(n=2, m=2, eta=1.0),
        'rainbow_asymmetric': GravityRainbowDispersion(n=1, m=2, eta=0.5),
    }
    
    return models

def test_dispersion_models():
    """Test various dispersion models with sample parameters."""
    
    print("Testing Theoretical LIV Dispersion Models")
    print("=" * 50)
    
    models = create_theoretical_models()
    test_energies = np.logspace(15, 20, 6)  # 1 PeV to 100 EeV
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Model: {model.name}")
        
        if hasattr(model, 'parameters'):
            print(f"  Parameters: {model.parameters}")
        
        # Calculate group velocities
        velocities = []
        for E in test_energies:
            try:
                v = model.group_velocity(E) / LIGHT_SPEED
                velocities.append(v)
                print(f"  E = {E:.1e} GeV: v/c = {v:.6f}")
            except:
                print(f"  E = {E:.1e} GeV: calculation failed")
                velocities.append(1.0)
        
        # Check for superluminal propagation
        if any(v > 1.0 for v in velocities):
            print(f"  WARNING: Superluminal propagation detected!")

if __name__ == "__main__":
    test_dispersion_models()
