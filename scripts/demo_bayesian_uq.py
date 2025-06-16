#!/usr/bin/env python3
"""
Demo Script: Bayesian Uncertainty Quantification for LIV Analysis

This script demonstrates the Bayesian UQ framework by running a simplified
analysis with reduced computational requirements for testing purposes.
"""

import sys
import os
sys.path.append('scripts')

from bayesian_uq_analysis import BayesianLIVAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_demo_analysis():
    """Run a demonstration of the Bayesian UQ framework."""
    
    print("🔬 BAYESIAN UQ DEMO FOR LIV CONSTRAINTS")
    print("="*50)
    
    # Initialize analyzer
    analyzer = BayesianLIVAnalysis()
    
    # Load/generate observational data
    analyzer.load_observational_data()
    
    # Run simplified analysis (fewer steps for demo)
    print("\n📊 Running simplified MCMC analysis...")
    
    # Analyze string theory model as example
    model = 'string_theory'
    samples, log_prob = analyzer.run_mcmc(
        model_type=model,
        n_walkers=16,     # Reduced for demo
        n_steps=1000,     # Reduced for demo  
        n_burn=200        # Reduced for demo
    )
    
    # Calculate evidence
    evidence = analyzer.calculate_model_evidence(model, samples, log_prob)
    print(f"Log evidence for {model}: {evidence:.2f}")
    
    # Parameter analysis
    param_results = analyzer.analyze_parameter_correlations(samples, model)
    
    print(f"\nParameter Results for {model}:")
    print(f"log(μ/GeV): {param_results['log_mu_mean']:.2f} ± {param_results['log_mu_std']:.2f}")
    print(f"log(g): {param_results['log_coupling_mean']:.2f} ± {param_results['log_coupling_std']:.2f}")
    print(f"Correlation: {param_results['correlation_coefficient']:.3f}")
    
    # Generate corner plot
    try:
        fig = analyzer.generate_corner_plot(samples, model, "results/demo_corner_plot.png")
        print("✓ Corner plot generated")
    except Exception as e:
        print(f"⚠ Corner plot generation failed: {e}")
    
    # Create summary plot of posterior
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: log(mu) posterior
    plt.subplot(1, 2, 1)
    plt.hist(samples[:, 0], bins=50, density=True, alpha=0.7, color='blue')
    plt.axvline(param_results['log_mu_mean'], color='red', linestyle='--', 
                label=f"Mean: {param_results['log_mu_mean']:.2f}")
    plt.xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$')
    plt.ylabel('Posterior Density')
    plt.title('Energy Scale Posterior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: log(g) posterior  
    plt.subplot(1, 2, 2)
    plt.hist(samples[:, 1], bins=50, density=True, alpha=0.7, color='green')
    plt.axvline(param_results['log_coupling_mean'], color='red', linestyle='--',
                label=f"Mean: {param_results['log_coupling_mean']:.2f}")
    plt.xlabel(r'$\log_{10}(g)$')
    plt.ylabel('Posterior Density')
    plt.title('Coupling Posterior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/demo_posterior_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Posterior summary plot saved")
    
    # Create 2D parameter space plot
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1, color='blue')
    plt.xlabel(r'$\log_{10}(\mu/\mathrm{GeV})$')
    plt.ylabel(r'$\log_{10}(g)$')
    plt.title('Parameter Space Posterior Samples')
    plt.grid(True, alpha=0.3)
    
    # Add confidence ellipses
    from matplotlib.patches import Ellipse
    
    # Calculate covariance
    cov = np.cov(samples.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # 68% confidence ellipse
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * np.sqrt(eigenvals * 2.3)  # 68% confidence
    
    ellipse = Ellipse(
        xy=(param_results['log_mu_mean'], param_results['log_coupling_mean']),
        width=width, height=height, angle=angle,
        facecolor='none', edgecolor='red', linewidth=2,
        label='68% Confidence'
    )
    plt.gca().add_patch(ellipse)
    
    plt.legend()
    plt.savefig('results/demo_parameter_space.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Parameter space plot saved")
    
    # Save results summary
    summary_data = {
        'Analysis Type': ['Bayesian MCMC'],
        'Model': [model],
        'N Samples': [len(samples)],
        'Log Evidence': [evidence],
        'Log Mu Mean': [param_results['log_mu_mean']],
        'Log Mu Std': [param_results['log_mu_std']],
        'Log Coupling Mean': [param_results['log_coupling_mean']],
        'Log Coupling Std': [param_results['log_coupling_std']],
        'Correlation': [param_results['correlation_coefficient']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/demo_bayesian_summary.csv', index=False)
    
    print("\n🎉 Demo analysis complete!")
    print("📊 Results saved to:")
    print("   - results/demo_bayesian_summary.csv")
    print("   - results/demo_posterior_summary.png") 
    print("   - results/demo_parameter_space.png")
    
    return samples, param_results

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Run demo
    samples, results = run_demo_analysis()
