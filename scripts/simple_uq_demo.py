#!/usr/bin/env python3
"""
Simple UQ Results Demo

This script demonstrates the uncertainty quantification results
without complex plotting to avoid any syntax issues.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def simple_uq_demo():
    """Run a simple demonstration of uncertainty quantification."""
    print("🔬 UNCERTAINTY QUANTIFICATION RESULTS DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 1000
    
    print(f"Using {n_samples} Monte Carlo samples for demonstration\n")
    
    # 1. GRB Time Delay Uncertainties
    print("🌟 1. GRB TIME DELAY UNCERTAINTIES")
    print("-" * 40)
    
    # Mock GRB constraint with realistic uncertainties
    grb_base_constraint = 2.5
    
    # Uncertainty sources
    redshift_uncertainty = 0.05 * grb_base_constraint  # 5% redshift calibration
    energy_cal_uncertainty = 0.10 * grb_base_constraint  # 10% energy calibration
    timing_uncertainty = 0.08 * grb_base_constraint     # 8% timing systematics
    intrinsic_uncertainty = 0.12 * grb_base_constraint  # 12% intrinsic variations
    
    # Total GRB uncertainty (add in quadrature)
    grb_total_uncertainty = np.sqrt(
        redshift_uncertainty**2 + 
        energy_cal_uncertainty**2 + 
        timing_uncertainty**2 + 
        intrinsic_uncertainty**2
    )
    
    # Generate constraint samples
    grb_samples = np.random.normal(grb_base_constraint, grb_total_uncertainty, n_samples)
    grb_mean = np.mean(grb_samples)
    grb_std = np.std(grb_samples)
    
    print(f"   ✓ Redshift calibration: {redshift_uncertainty/grb_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Energy calibration: {energy_cal_uncertainty/grb_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Timing systematics: {timing_uncertainty/grb_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Intrinsic variations: {intrinsic_uncertainty/grb_base_constraint*100:.1f}% contribution")
    print(f"   🎯 GRB constraint: {grb_mean:.3f} ± {grb_std:.3f}")
    
    # 2. UHECR Propagation Uncertainties
    print("\n⚡ 2. UHECR PROPAGATION UNCERTAINTIES")
    print("-" * 40)
    
    # Mock UHECR constraint with realistic uncertainties
    uhecr_base_constraint = 1.8
    
    # Uncertainty sources
    energy_recon_uncertainty = 0.15 * uhecr_base_constraint  # 15% energy reconstruction
    stochastic_loss_uncertainty = 0.20 * uhecr_base_constraint  # 20% stochastic losses
    atmospheric_uncertainty = 0.08 * uhecr_base_constraint   # 8% atmospheric modeling
    detector_uncertainty = 0.05 * uhecr_base_constraint      # 5% detector acceptance
    
    # Total UHECR uncertainty
    uhecr_total_uncertainty = np.sqrt(
        energy_recon_uncertainty**2 + 
        stochastic_loss_uncertainty**2 + 
        atmospheric_uncertainty**2 + 
        detector_uncertainty**2
    )
    
    # Generate constraint samples
    uhecr_samples = np.random.normal(uhecr_base_constraint, uhecr_total_uncertainty, n_samples)
    uhecr_mean = np.mean(uhecr_samples)
    uhecr_std = np.std(uhecr_samples)
    
    print(f"   ✓ Energy reconstruction: {energy_recon_uncertainty/uhecr_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Stochastic losses: {stochastic_loss_uncertainty/uhecr_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Atmospheric modeling: {atmospheric_uncertainty/uhecr_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Detector acceptance: {detector_uncertainty/uhecr_base_constraint*100:.1f}% contribution")
    print(f"   🎯 UHECR constraint: {uhecr_mean:.3f} ± {uhecr_std:.3f}")
    
    # 3. Vacuum Instability Uncertainties
    print("\n⚛️  3. VACUUM INSTABILITY UNCERTAINTIES")
    print("-" * 40)
    
    # Mock vacuum constraint with realistic uncertainties
    vacuum_base_constraint = 0.8
    
    # Uncertainty sources
    field_cal_uncertainty = 0.05 * vacuum_base_constraint    # 5% field calibration
    eft_param_uncertainty = 0.15 * vacuum_base_constraint    # 15% EFT parameters
    quantum_uncertainty = 0.08 * vacuum_base_constraint      # 8% quantum corrections
    finite_size_uncertainty = 0.03 * vacuum_base_constraint  # 3% finite size effects
    
    # Total vacuum uncertainty
    vacuum_total_uncertainty = np.sqrt(
        field_cal_uncertainty**2 + 
        eft_param_uncertainty**2 + 
        quantum_uncertainty**2 + 
        finite_size_uncertainty**2
    )
    
    # Generate constraint samples
    vacuum_samples = np.random.normal(vacuum_base_constraint, vacuum_total_uncertainty, n_samples)
    vacuum_mean = np.mean(vacuum_samples)
    vacuum_std = np.std(vacuum_samples)
    
    print(f"   ✓ Field calibration: {field_cal_uncertainty/vacuum_base_constraint*100:.1f}% contribution")
    print(f"   ✓ EFT parameters: {eft_param_uncertainty/vacuum_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Quantum corrections: {quantum_uncertainty/vacuum_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Finite size effects: {finite_size_uncertainty/vacuum_base_constraint*100:.1f}% contribution")
    print(f"   🎯 Vacuum constraint: {vacuum_mean:.3f} ± {vacuum_std:.3f}")
    
    # 4. Hidden Sector Uncertainties
    print("\n🔍 4. HIDDEN SECTOR UNCERTAINTIES")
    print("-" * 40)
    
    # Mock hidden sector constraint with realistic uncertainties
    hidden_base_constraint = 1.2
    
    # Uncertainty sources
    sensitivity_uncertainty = 0.20 * hidden_base_constraint  # 20% sensitivity calibration
    background_uncertainty = 0.15 * hidden_base_constraint   # 15% background modeling
    conversion_uncertainty = 0.10 * hidden_base_constraint   # 10% conversion efficiency
    theory_uncertainty = 0.25 * hidden_base_constraint       # 25% theoretical coupling
    
    # Total hidden sector uncertainty
    hidden_total_uncertainty = np.sqrt(
        sensitivity_uncertainty**2 + 
        background_uncertainty**2 + 
        conversion_uncertainty**2 + 
        theory_uncertainty**2
    )
    
    # Generate constraint samples
    hidden_samples = np.random.normal(hidden_base_constraint, hidden_total_uncertainty, n_samples)
    hidden_mean = np.mean(hidden_samples)
    hidden_std = np.std(hidden_samples)
    
    print(f"   ✓ Sensitivity calibration: {sensitivity_uncertainty/hidden_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Background modeling: {background_uncertainty/hidden_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Conversion efficiency: {conversion_uncertainty/hidden_base_constraint*100:.1f}% contribution")
    print(f"   ✓ Theoretical coupling: {theory_uncertainty/hidden_base_constraint*100:.1f}% contribution")
    print(f"   🎯 Hidden sector constraint: {hidden_mean:.3f} ± {hidden_std:.3f}")
    
    # 5. Multi-Channel Combination
    print("\n🔗 5. MULTI-CHANNEL COMBINATION")
    print("-" * 40)
    
    # Combine constraints with optimal weighting
    constraints = [grb_mean, uhecr_mean, vacuum_mean, hidden_mean]
    uncertainties = [grb_std, uhecr_std, vacuum_std, hidden_std]
    channel_names = ['GRB', 'UHECR', 'Vacuum', 'Hidden']
    
    # Weights are inverse variances
    weights = [1.0 / (unc**2) for unc in uncertainties]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Combined constraint
    combined_constraint = sum(w * c for w, c in zip(normalized_weights, constraints))
    combined_uncertainty = 1.0 / np.sqrt(total_weight)
    
    print("   Channel weights:")
    for name, weight in zip(channel_names, normalized_weights):
        print(f"   • {name}: {weight:.1%}")
    
    print(f"\n   🎯 Combined constraint: {combined_constraint:.3f} ± {combined_uncertainty:.3f}")
    
    # 6. Summary Statistics
    print("\n📊 6. UNCERTAINTY QUANTIFICATION SUMMARY")
    print("-" * 40)
    
    # Create summary table
    summary_data = []
    for name, constraint, uncertainty in zip(channel_names, constraints, uncertainties):
        relative_uncertainty = uncertainty / constraint * 100
        summary_data.append({
            'Channel': name,
            'Constraint': f"{constraint:.3f}",
            'Uncertainty': f"±{uncertainty:.3f}",
            'Relative (%)': f"{relative_uncertainty:.1f}%"
        })
    
    # Add combined result
    combined_relative = combined_uncertainty / combined_constraint * 100
    summary_data.append({
        'Channel': 'COMBINED',
        'Constraint': f"{combined_constraint:.3f}",
        'Uncertainty': f"±{combined_uncertainty:.3f}",
        'Relative (%)': f"{combined_relative:.1f}%"
    })
      # Print summary table
    print("   CONSTRAINT SUMMARY:")
    print("   " + "-" * 55)
    print(f"   {'Channel':<12} {'Constraint':<12} {'Uncertainty':<12} {'Relative'}")
    print("   " + "-" * 55)
    for row in summary_data:
        print(f"   {row['Channel']:<12} {row['Constraint']:<12} {row['Uncertainty']:<12} {row['Relative (%)']}")
    
    # Key findings
    print(f"\n   🔍 KEY FINDINGS:")
    print(f"   • Most precise channel: {channel_names[uncertainties.index(min(uncertainties))]}")
    print(f"   • Largest uncertainty: {channel_names[uncertainties.index(max(uncertainties))]}")
    print(f"   • Combined improvement: {min(uncertainties)/combined_uncertainty:.1f}x better than best single channel")
    print(f"   • Multi-channel approach reduces uncertainty by {(1 - combined_uncertainty/min(uncertainties))*100:.1f}%")
    
    print(f"\n✅ UNCERTAINTY QUANTIFICATION DEMONSTRATION COMPLETE!")
    print(f"📊 All {len(channel_names)} channels successfully analyzed with realistic uncertainties")
    print(f"🎯 Combined constraint provides optimal precision with proper uncertainty propagation")
    
    return {
        'grb': {'mean': grb_mean, 'std': grb_std},
        'uhecr': {'mean': uhecr_mean, 'std': uhecr_std},
        'vacuum': {'mean': vacuum_mean, 'std': vacuum_std},
        'hidden': {'mean': hidden_mean, 'std': hidden_std},
        'combined': {'mean': combined_constraint, 'std': combined_uncertainty}
    }

if __name__ == "__main__":
    results = simple_uq_demo()
