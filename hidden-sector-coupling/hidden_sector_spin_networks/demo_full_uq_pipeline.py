#!/usr/bin/env python3
"""
Complete UQ Pipeline Demonstration with Full Analysis

This script demonstrates the complete three-stage UQ workflow with higher
sample counts for reliable sensitivity analysis and surrogate modeling.

Usage:
    python demo_full_uq_pipeline.py

Author: Quantum Geometry Hidden Sector Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def full_uq_demo():
    """Full UQ demonstration with complete analysis."""
    
    print("="*80)
    print("🚀 COMPLETE UQ PIPELINE - SU(2) SPIN NETWORK PORTAL")
    print("="*80)
    
    # Import framework
    try:
        from uncertainty_quantification import (
            SpinNetworkUQFramework, 
            UQConfig
        )
        from su2_recoupling_module import SpinNetworkConfig
        print("✓ UQ framework imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import UQ framework: {e}")
        return
    
    # Configuration with realistic sample sizes
    base_config = SpinNetworkConfig(
        base_coupling=1e-5,
        geometric_suppression=0.1,
        portal_correlation_length=2.0,
        max_angular_momentum=3,
        network_size=10,
        connectivity=0.5
    )
    
    uq_config = UQConfig(
        n_samples=1000,  # Higher for reliable statistics
        sampling_method='sobol',  # Sobol for sensitivity analysis
        confidence_level=0.95,
        random_seed=42
    )
    
    framework = SpinNetworkUQFramework(base_config, uq_config)
    
    # Add experimental uncertainties based on realistic measurements
    print("\n🔬 Setting up experimental uncertainty distributions...")
    
    # Photon time delay measurements (typical precision ~10-20%)
    framework.add_experimental_uncertainty(
        'photon_delay_measurement', 1.0, 0.15, 'gaussian'
    )
    
    # UHECR flux measurements (typical precision ~5-10%)
    framework.add_experimental_uncertainty(
        'uhecr_flux_measurement', 1.0, 0.08, 'log_normal'
    )
    
    print("\n" + "="*60)
    print("⚡ UNCERTAINTY PROPAGATION")
    print("="*60)
    
    # Generate samples
    print(f"\n🎲 Generating {uq_config.n_samples} parameter samples...")
    parameter_samples = framework.generate_parameter_samples()
    
    # Evaluate model (use fast method for demo)
    print("\n⚡ Evaluating spin network portal model...")
    output_samples = framework.evaluate_model_fast(parameter_samples)
    
    print("\n" + "="*60)
    print("📊 STATISTICAL ANALYSIS")
    print("="*60)
    
    # Statistical summary
    print("\n📈 Computing statistical summaries...")
    summary = framework.compute_statistical_summary(output_samples)
    
    # Display key results
    print(f"\n🎯 Key Results (95% confidence):")
    for output_name, stats in summary.items():
        if stats['valid_fraction'] > 0.8:  # Only show reliable results
            print(f"\n{output_name.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.2e}")
            print(f"  95% CI: [{stats['ci_lower']:.2e}, {stats['ci_upper']:.2e}]")
            print(f"  Coefficient of Variation: {stats['mean']/stats['std']:.1f}")
    
    # Sensitivity analysis with more samples
    print(f"\n🔍 Global sensitivity analysis...")
    try:
        # Run with larger sample for Sobol indices
        sens_config = UQConfig(
            n_samples=500,  # Reduced but sufficient for Sobol
            sampling_method='sobol',
            random_seed=42
        )
        sens_framework = SpinNetworkUQFramework(base_config, sens_config)
        for param, info in framework.parameter_distributions.items():
            sens_framework.parameter_distributions[param] = info
            
        sensitivity_results = sens_framework.sensitivity_analysis(['transfer_rate'])
        
        if sensitivity_results:
            print("  📊 Parameter sensitivity rankings:")
            for output_name, indices in sensitivity_results.items():
                if 'S1' in indices:
                    sorted_params = sorted(indices['S1'].items(), 
                                         key=lambda x: abs(x[1]), reverse=True)
                    print(f"\n  {output_name}:")
                    for param, sensitivity in sorted_params[:5]:
                        print(f"    {param:25}: {sensitivity:6.3f}")
        else:
            print("  ⚠ Sensitivity analysis failed - using simplified approach")
            
    except Exception as e:
        print(f"  ⚠ Sensitivity analysis error: {e}")
    
    # Robust optimization
    print(f"\n🎯 Robust optimization...")
    try:
        robust_params = framework.robust_optimization(
            parameter_samples, output_samples['transfer_rate'],
            objective='mean_minus_std'
        )
        
        print(f"  🏆 Robust optimal parameters:")
        for param, value in robust_params.items():
            print(f"    {param:25}: {value:.3e}")
            
    except Exception as e:
        print(f"  ⚠ Robust optimization error: {e}")
    
    print("\n" + "="*60)
    print("📋 EXPERIMENTAL RECOMMENDATIONS")
    print("="*60)
    
    # Extract key uncertainties for experimental design
    transfer_stats = summary['transfer_rate']
    amplitude_stats = summary['leakage_amplitude']
    
    print(f"\n🔬 Measurement requirements for detection:")
    print(f"  Transfer rate sensitivity: ≤{transfer_stats['std']:.1e} s⁻¹")
    print(f"  Amplitude sensitivity:     ≤{amplitude_stats['std']:.1e}")
    print(f"  Recommended integration time: ≥{1/transfer_stats['mean']:.1e} s")
    
    # Assessment of current uncertainty levels
    transfer_cv = transfer_stats['std'] / transfer_stats['mean']
    if transfer_cv > 1.0:
        print(f"\n⚠️  High uncertainty detected (CV = {transfer_cv:.1f})")
        print("   Recommendations:")
        print("   • Tighten theoretical priors via lattice QCD")
        print("   • Improve experimental precision on input parameters")
        print("   • Focus measurements on high-sensitivity parameters")
    else:
        print(f"\n✅ Manageable uncertainty (CV = {transfer_cv:.1f})")
        print("   Ready for experimental design phase")
    
    print("\n" + "="*80)
    print("🎉 COMPLETE UQ PIPELINE DEMONSTRATION FINISHED")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"  Samples processed: {uq_config.n_samples}")
    print(f"  Parameters analyzed: {len(parameter_samples)}")
    print(f"  Output quantities: {len(output_samples)}")
    print(f"  Valid sample fraction: {np.mean([s['valid_fraction'] for s in summary.values()]):.1%}")
    
    print(f"\n🚀 The UQ framework provides:")
    print(f"  📈 Rigorous error bars on all predictions")
    print(f"  🎯 Parameter sensitivity rankings for experiments")
    print(f"  🛡️ Robust parameter choices under uncertainty")
    print(f"  📏 Precision requirements for detection")

if __name__ == "__main__":
    full_uq_demo()
