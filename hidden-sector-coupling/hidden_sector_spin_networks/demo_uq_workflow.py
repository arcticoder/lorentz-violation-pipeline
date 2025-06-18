#!/usr/bin/env python3
"""
Uncertainty Quantification Demonstration for SU(2) Spin Network Portal

This script demonstrates the complete three-stage UQ workflow:
1. Quantify input uncertainties with probability distributions
2. Propagate uncertainties via Monte Carlo and surrogate modeling  
3. Analyze results with sensitivity analysis and robust optimization

Usage:
    python demo_uq_workflow.py

Author: Quantum Geometry Hidden Sector Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main UQ demonstration function."""
    
    print("="*80)
    print("🌟 SU(2) SPIN NETWORK PORTAL - UNCERTAINTY QUANTIFICATION DEMO")
    print("="*80)
    
    # Test imports
    try:
        from uncertainty_quantification import (
            SpinNetworkUQFramework, 
            UQConfig,
            create_experimental_uq_config,
            demo_basic_uq
        )
        from su2_recoupling_module import SpinNetworkConfig
        print("✓ UQ framework imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import UQ framework: {e}")
        return
    
    print("\n" + "="*60)
    print("1️⃣ STAGE 1: INPUT UNCERTAINTY QUANTIFICATION")
    print("="*60)
      # Create UQ framework with experimental uncertainties
    print("\n🔬 Setting up experimental uncertainty distributions...")
    
    # Create UQ config with smaller sample size for demo
    uq_config = UQConfig(
        n_samples=100,  # Reduced for demo speed
        sampling_method='latin_hypercube',
        random_seed=42,
        confidence_level=0.95
    )
    
    base_config = SpinNetworkConfig(
        base_coupling=1e-5,
        geometric_suppression=0.1,
        portal_correlation_length=1.5,
        max_angular_momentum=3,
        network_size=8,  # Smaller for demo speed
        connectivity=0.4
    )
      
    # Use the uq_config we defined above with 100 samples
    framework = SpinNetworkUQFramework(base_config, uq_config)
    
    # Add experimental uncertainties
    print("  • Adding photon delay measurement uncertainty: 10% error")
    framework.add_experimental_uncertainty(
        'photon_delay_measurement', 1.0, 0.1, 'gaussian'
    )
    
    print("  • Adding UHECR flux measurement uncertainty: 5% log-normal error")
    framework.add_experimental_uncertainty(
        'uhecr_flux_measurement', 1.0, 0.05, 'log_normal'
    )
    
    # Display parameter distributions
    print(f"\n📋 Parameter uncertainty distributions:")
    for name, param_dist in framework.parameter_distributions.items():
        dist_info = f"{param_dist.distribution.__class__.__name__}"
        if hasattr(param_dist.distribution, 'args'):
            dist_info += f" {param_dist.distribution.args}"
        print(f"  • {name:25}: {dist_info}")
        if param_dist.description:
            print(f"    → {param_dist.description}")
    
    print("\n" + "="*60)
    print("2️⃣ STAGE 2: UNCERTAINTY PROPAGATION")
    print("="*60)
    
    # Generate parameter samples
    print(f"\n🎲 Generating {uq_config.n_samples} parameter samples using {uq_config.sampling_method}...")
    start_time = time.time()
    parameter_samples = framework.generate_parameter_samples()
    sampling_time = time.time() - start_time
    
    print(f"   ✓ Sampling complete in {sampling_time:.2f} seconds")
    print(f"   📊 Sample statistics:")
    
    for param_name, samples in parameter_samples.items():
        if param_name in ['max_angular_momentum', 'network_size']:
            # Integer parameters
            print(f"     {param_name:25}: [{np.min(samples):3.0f}, {np.max(samples):3.0f}] (discrete)")
        else:
            # Continuous parameters
            print(f"     {param_name:25}: [{np.min(samples):.2e}, {np.max(samples):.2e}]")
    
    # Evaluate model with uncertainty propagation
    print(f"\n⚡ Propagating uncertainties through spin network portal...")
    start_time = time.time()
    output_samples = framework.evaluate_model_fast(parameter_samples)
    evaluation_time = time.time() - start_time
    
    print(f"   ✓ Model evaluation complete in {evaluation_time:.1f} seconds")
    print(f"   📈 Output quantities computed:")
    
    for output_name, samples in output_samples.items():
        valid_fraction = np.sum(np.isfinite(samples)) / len(samples)
        print(f"     {output_name:25}: {valid_fraction*100:5.1f}% valid samples")
    
    print("\n" + "="*60)
    print("3️⃣ STAGE 3: UNCERTAINTY ANALYSIS")
    print("="*60)
    
    # Statistical summary
    print("\n📊 Computing statistical summaries...")
    statistical_summary = framework.compute_statistical_summary(output_samples)
    
    print("\n🎯 Statistical Summary:")
    print("-" * 40)
    
    for output_name, stats in statistical_summary.items():
        if stats['valid_fraction'] > 0:
            print(f"\n{output_name.replace('_', ' ').title()}:")
            print(f"  Mean:        {stats['mean']:.2e}")
            print(f"  Std Dev:     {stats['std']:.2e}")
            print(f"  Median:      {stats['median']:.2e}")
            print(f"  95% CI:      [{stats['ci_lower']:.2e}, {stats['ci_upper']:.2e}]")
            print(f"  Range:       [{stats['min']:.2e}, {stats['max']:.2e}]")
            print(f"  CV:          {stats['cv']:.2f}")
            print(f"  Valid:       {stats['valid_fraction']*100:.1f}%")
    
    # Sensitivity analysis
    print("\n🔍 Performing global sensitivity analysis...")
    try:
        sensitivity_indices = framework.sobol_sensitivity_analysis(parameter_samples, output_samples)
        
        if sensitivity_indices:
            print("\n🎛️ Parameter Sensitivity Ranking:")
            print("-" * 40)
            
            for output_name, sens_data in sensitivity_indices.items():
                print(f"\n{output_name.replace('_', ' ').title()}:")
                
                # Sort by first-order indices
                first_order = sens_data['first_order']
                sorted_params = sorted(first_order.items(), key=lambda x: x[1], reverse=True)
                
                for param_name, index in sorted_params[:5]:  # Top 5
                    total_index = sens_data['total_order'].get(param_name, 0)
                    print(f"  {param_name:20}: S₁={index:6.3f}, Sₜ={total_index:6.3f}")
        else:
            print("  ⚠ Sensitivity analysis failed (insufficient samples or SALib not available)")
    
    except Exception as e:
        print(f"  ⚠ Sensitivity analysis error: {e}")
    
    # Surrogate modeling
    print("\n🤖 Building polynomial chaos surrogate models...")
    try:
        surrogate_models = framework.build_polynomial_chaos_surrogate(parameter_samples, output_samples)
        
        if surrogate_models:
            print("\n📈 Surrogate Model Performance:")
            print("-" * 40)
            
            for output_name, model_data in surrogate_models.items():
                r2 = model_data['r2_score']
                n_terms = model_data['n_terms']
                degree = model_data['degree']
                
                print(f"  {output_name:20}: R² = {r2:.3f}, {n_terms} terms (degree {degree})")
                
                if r2 > 0.8:
                    print(f"    → ✓ Good surrogate quality")
                elif r2 > 0.5:
                    print(f"    → ⚠ Moderate surrogate quality")
                else:
                    print(f"    → ❌ Poor surrogate quality")
        else:
            print("  ⚠ Surrogate modeling failed (chaospy not available or insufficient data)")
    
    except Exception as e:
        print(f"  ⚠ Surrogate modeling error: {e}")
    
    # Robust optimization
    print("\n🎯 Performing robust optimization...")
    try:
        robust_optima = framework.robust_optimization(
            output_samples, parameter_samples, 
            objective='mean_minus_std',
            primary_output='transfer_rate'
        )
        
        if robust_optima:
            print(f"\n🏆 Robust Optimal Parameters (objective: {robust_optima['objective']}):")
            print("-" * 50)
            
            optimal_params = robust_optima['optimal_parameters']
            for param_name, optimal_value in optimal_params.items():
                print(f"  {param_name:25}: {optimal_value:.3e}")
            
            print(f"\nOptimal output value: {robust_optima['optimal_output']:.2e}")
            
            output_stats = robust_optima['output_statistics']
            print(f"Population statistics:")
            print(f"  Mean: {output_stats['mean']:.2e}")
            print(f"  Std:  {output_stats['std']:.2e}")
            print(f"  95th percentile: {output_stats['percentile_95']:.2e}")
        else:
            print("  ⚠ Robust optimization failed")
    
    except Exception as e:
        print(f"  ⚠ Robust optimization error: {e}")
    
    print("\n" + "="*60)
    print("4️⃣ VISUALIZATION & REPORTING")
    print("="*60)
    
    # Create comprehensive visualizations
    print("\n📊 Generating UQ visualizations...")
    
    try:
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Parameter sample distributions
        ax = axes[0]
        param_to_plot = 'base_coupling'
        if param_to_plot in parameter_samples:
            samples = parameter_samples[param_to_plot]
            ax.hist(samples, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(param_to_plot.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title('Input Distribution')
            ax.set_xscale('log')
        
        # Plot 2: Output distribution
        ax = axes[1]
        output_to_plot = 'transfer_rate'
        if output_to_plot in output_samples:
            samples = output_samples[output_to_plot]
            valid_samples = samples[np.isfinite(samples)]
            if len(valid_samples) > 10:
                ax.hist(valid_samples, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Transfer Rate')
                ax.set_ylabel('Frequency')
                ax.set_title('Output Distribution')
                ax.set_yscale('log')
        
        # Plot 3: Parameter vs Output scatter
        ax = axes[2]
        if param_to_plot in parameter_samples and output_to_plot in output_samples:
            x_data = parameter_samples[param_to_plot]
            y_data = output_samples[output_to_plot]
            valid_mask = np.isfinite(y_data)
            
            if np.sum(valid_mask) > 10:
                ax.scatter(x_data[valid_mask], y_data[valid_mask], alpha=0.6, s=20)
                ax.set_xlabel(param_to_plot.replace('_', ' ').title())
                ax.set_ylabel('Transfer Rate')
                ax.set_title('Parameter vs Output')
                ax.set_xscale('log')
                ax.set_yscale('log')
        
        # Plot 4: Sensitivity indices (if available)
        ax = axes[3]
        if sensitivity_indices and output_to_plot in sensitivity_indices:
            sens_data = sensitivity_indices[output_to_plot]['first_order']
            params = list(sens_data.keys())[:5]  # Top 5
            indices = [sens_data[p] for p in params]
            
            bars = ax.bar(range(len(params)), indices, alpha=0.7)
            ax.set_xlabel('Parameters')
            ax.set_ylabel('First-Order Sobol Index')
            ax.set_title('Sensitivity Analysis')
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels([p.replace('_', '\n') for p in params], rotation=45)
        else:
            ax.text(0.5, 0.5, 'Sensitivity Analysis\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sensitivity Analysis')
        
        # Plot 5: Confidence intervals
        ax = axes[4]
        output_names = list(statistical_summary.keys())[:4]  # Top 4 outputs
        means = [statistical_summary[name]['mean'] for name in output_names]
        ci_lower = [statistical_summary[name]['ci_lower'] for name in output_names]
        ci_upper = [statistical_summary[name]['ci_upper'] for name in output_names]
        
        yerr_lower = [m - l for m, l in zip(means, ci_lower)]
        yerr_upper = [u - m for m, u in zip(means, ci_upper)]
        
        x_pos = range(len(output_names))
        ax.errorbar(x_pos, means, yerr=[yerr_lower, yerr_upper], 
                   fmt='o', capsize=5, capthick=2)
        ax.set_xlabel('Output Quantities')
        ax.set_ylabel('Value')
        ax.set_title('95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('_', '\n') for name in output_names], rotation=45)
        ax.set_yscale('log')
        
        # Plot 6: Parameter correlation matrix
        ax = axes[5]
        param_names = list(parameter_samples.keys())[:5]  # Top 5 parameters
        if len(param_names) > 1:
            param_matrix = np.column_stack([parameter_samples[name] for name in param_names])
            corr_matrix = np.corrcoef(param_matrix.T)
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Add correlation values
            for i in range(len(param_names)):
                for j in range(len(param_names)):
                    text = ax.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                                  ha="center", va="center", color="black", fontsize=8)
            
            ax.set_xticks(range(len(param_names)))
            ax.set_yticks(range(len(param_names)))
            ax.set_xticklabels([name.replace('_', '\n') for name in param_names], rotation=45)
            ax.set_yticklabels([name.replace('_', '\n') for name in param_names])
            ax.set_title('Parameter Correlations')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Insufficient\nParameters', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Parameter Correlations')
        
        plt.tight_layout()
        plt.show()
        
        print("  ✓ Visualization complete")
        
    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")
    
    print("\n" + "="*60)
    print("5️⃣ COMPARISON: POINT ESTIMATES vs UQ")
    print("="*60)
    
    # Compare with point estimate approach
    print("\n⚖️ Point Estimates vs Uncertainty Quantification:")
    print("-" * 50)
    
    # Point estimate (using base configuration)
    try:
        from su2_recoupling_module import SpinNetworkPortal
        
        point_portal = SpinNetworkPortal(base_config)
        
        def density_of_states(E):
            return E**2 / 10
        
        point_transfer_rate = point_portal.energy_transfer_rate((1.0, 10.0), density_of_states)
        point_amplitude = abs(point_portal.energy_leakage_amplitude(10.0, 5.0))
        
        print(f"Point Estimate Results:")
        print(f"  Transfer rate:     {point_transfer_rate:.2e}")
        print(f"  Leakage amplitude: {point_amplitude:.2e}")
        
        # UQ results
        if 'transfer_rate' in statistical_summary:
            uq_stats = statistical_summary['transfer_rate']
            print(f"\nUQ Results (transfer rate):")
            print(f"  Mean:              {uq_stats['mean']:.2e}")
            print(f"  Uncertainty (±σ):  {uq_stats['std']:.2e}")
            print(f"  Coefficient of variation: {uq_stats['cv']:.2f}")
            print(f"  95% CI width:      {uq_stats['ci_upper'] - uq_stats['ci_lower']:.2e}")
            
            # Reliability assessment
            relative_uncertainty = uq_stats['std'] / abs(uq_stats['mean']) if uq_stats['mean'] != 0 else np.inf
            
            print(f"\n📋 Reliability Assessment:")
            if relative_uncertainty < 0.1:
                print("  ✓ Low uncertainty: Point estimate is reliable")
            elif relative_uncertainty < 0.5:
                print("  ⚠ Moderate uncertainty: UQ recommended for robust conclusions")
            else:
                print("  ❌ High uncertainty: Point estimates may be misleading")
            
            print(f"  Relative uncertainty: {relative_uncertainty:.1%}")
    
    except Exception as e:
        print(f"  ⚠ Comparison error: {e}")
    
    print("\n" + "="*60)
    print("6️⃣ EXPERIMENTAL DESIGN RECOMMENDATIONS")
    print("="*60)
    
    print("\n🔬 Experimental Design Under Uncertainty:")
    print("-" * 45)
    
    # Measurement precision requirements
    if 'transfer_rate' in statistical_summary:
        stats = statistical_summary['transfer_rate']
        required_precision = stats['std'] / 3  # 3σ precision for meaningful detection
        
        print(f"Recommended measurement precision:")
        print(f"  Transfer rate resolution: {required_precision:.2e} s⁻¹")
        print(f"  Energy resolution:        <0.01 eV (from theory)")
        print(f"  Time resolution:          <{1/stats['mean']/10:.1e} s" if stats['mean'] > 0 else "  Time resolution:          N/A")
    
    # Parameter prioritization for experimental control
    if sensitivity_indices and 'transfer_rate' in sensitivity_indices:
        sens_data = sensitivity_indices['transfer_rate']['first_order']
        top_param = max(sens_data.items(), key=lambda x: x[1])
        
        print(f"\nParameter control priority:")
        print(f"  Most critical parameter: {top_param[0]} (S₁ = {top_param[1]:.3f})")
        print(f"  → Focus experimental control on this parameter")
    
    # Statistical power analysis
    valid_fraction = np.mean([s['valid_fraction'] for s in statistical_summary.values()])
    print(f"\nStatistical considerations:")
    print(f"  Valid sample fraction: {valid_fraction:.1%}")
    print(f"  Recommended sample size: >{max(1000, int(1/valid_fraction*100))} for robust statistics")
    
    print("\n" + "="*80)
    print("🎉 UNCERTAINTY QUANTIFICATION DEMONSTRATION COMPLETE")
    print("="*80)
    
    print("\n📋 Summary of UQ capabilities demonstrated:")
    print("  ✓ Input uncertainty quantification with probability distributions")
    print("  ✓ Uncertainty propagation via Monte Carlo sampling")
    print("  ✓ Statistical analysis with confidence intervals")
    print("  ✓ Global sensitivity analysis (parameter ranking)")
    print("  ✓ Surrogate modeling for computational efficiency")
    print("  ✓ Robust optimization under uncertainty")
    print("  ✓ Comparison with point estimates")
    print("  ✓ Experimental design recommendations")
    
    total_time = evaluation_time + sampling_time
    print(f"\n⏱️ Total computational time: {total_time:.1f} seconds")
    print(f"  Sampling: {sampling_time:.1f}s, Evaluation: {evaluation_time:.1f}s")
    
    print("\n🚀 The framework transforms point estimates into:")
    print("  📊 Probabilistic predictions with rigorous error bars")
    print("  🎯 Parameter sensitivity rankings for experimental focus")  
    print("  🛡️ Robust designs that perform well under uncertainty")
    print("  📏 Measurement precision requirements for detection")
    
    print("\n💡 Next steps:")
    print("  1. Calibrate uncertainty distributions with experimental data")
    print("  2. Validate surrogate models against full simulations")
    print("  3. Implement Bayesian inference for parameter estimation")
    print("  4. Integrate with laboratory measurement planning")

if __name__ == "__main__":
    main()
