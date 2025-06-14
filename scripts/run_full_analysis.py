#!/usr/bin/env python3
"""
Enhanced Lorentz Invariance Violation Analysis Pipeline

This orchestrator runs the complete LIV analysis pipeline with enhanced
theoretical model testing capabilities:

1. UHECR spectrum calculation with data validation
2. Enhanced GRB analysis with polynomial dispersion relations
3. Enhanced UHECR analysis with theoretical model testing  
4. Combined figure-of-merit computation
5. Theoretical model comparison and validation

Key improvements:
- Polynomial dispersion fitting (replaces linear-only)
- Explicit theoretical model testing (polymer-QED, gravity-rainbow)
- Vacuum instability and hidden sector calculations
- Enhanced uncertainty quantification and model selection
"""

import subprocess
import sys
import os
import time

def run_step(name, cmd, critical=True):
    """Run a pipeline step with error handling and timing"""
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(f"✅ SUCCESS ({elapsed:.1f}s)")
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED ({elapsed:.1f}s)")
        print(f"Return code: {e.returncode}")
        
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
        
        if critical:
            print(f"\nCritical step failed: {name}")
            sys.exit(e.returncode)
        else:
            print(f"\nNon-critical step failed: {name} (continuing...)")
            return False

def check_enhanced_modules():
    """Check if enhanced analysis modules are available"""
    try:
        import sys
        sys.path.append('scripts')
        
        import enhanced_grb_analysis
        import enhanced_uhecr_analysis  
        import theoretical_liv_models
        
        print("✅ Enhanced analysis modules found")
        return True
        
    except ImportError as e:
        print(f"⚠️ Enhanced modules not available: {e}")
        print("   Will use basic analysis as fallback")
        return False

def print_summary():
    """Print analysis summary"""
    print(f"\n{'='*70}")
    print("LORENTZ INVARIANCE VIOLATION ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    # Check what files were generated
    result_files = [
        ("UHECR Spectrum", "data/uhecr/sd1500_spectrum.csv"),
        ("GRB Enhanced Bounds", "results/grb_enhanced_bounds.csv"),
        ("GRB Basic Bounds", "results/grb_bounds.csv"),
        ("UHECR Enhanced Exclusion", "results/uhecr_enhanced_exclusion.csv"),
        ("UHECR Basic Exclusion", "results/uhecr_exclusion.csv"),
        ("Combined FOM", "results/combined_fom.csv"),
        ("Combined Bounds", "results/combined_bounds.csv")
    ]
    
    print("\nGenerated files:")
    for name, path in result_files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {name}: {path} ({size} bytes)")
        else:
            print(f"  ❌ {name}: {path} (missing)")
    
    # Check for plots
    plot_dir = "results"
    if os.path.exists(plot_dir):
        plots = [f for f in os.listdir(plot_dir) if f.endswith(('.png', '.pdf', '.svg'))]
        if plots:
            print(f"\nGenerated plots ({len(plots)}):")
            for plot in plots:
                print(f"  📊 {plot}")
    
    print(f"\n{'='*70}")

def main():
    """Run the complete enhanced LIV analysis pipeline"""
    print("ENHANCED LORENTZ INVARIANCE VIOLATION ANALYSIS PIPELINE")
    print(f"{'='*70}")
    print("Features:")
    print("• Polynomial dispersion relation fitting")
    print("• Theoretical model testing (polymer-QED, gravity-rainbow)")
    print("• Vacuum instability calculations") 
    print("• Hidden sector energy loss analysis")
    print("• Enhanced uncertainty quantification")
    print(f"{'='*70}")
    
    # Check for enhanced modules
    enhanced_available = check_enhanced_modules()
    
    pipeline_start = time.time()
    
    # Step 1: UHECR spectrum calculation (prerequisite)
    run_step(
        "UHECR Spectrum Calculation", 
        ['python', 'scripts/uhecr_spectrum.py'],
        critical=True
    )
    
    # Step 2: Enhanced GRB analysis with polynomial dispersion
    run_step(
        "Enhanced GRB Analysis (Polynomial Dispersion)", 
        ['python', 'scripts/analyze_grb.py'],
        critical=True
    )
    
    # Step 3: Enhanced UHECR analysis with theoretical models
    run_step(
        "Enhanced UHECR Analysis (Theoretical Models)", 
        ['python', 'scripts/simulate_uhecr.py'],
        critical=True  
    )
    
    # Step 4: Combined analysis and figure-of-merit
    run_step(
        "Combined Figure-of-Merit Computation", 
        ['python', 'scripts/combined_fom.py'],
        critical=False  # Not critical if other steps succeeded
    )
    
    # Optional: Run theoretical model validation (if available)
    if enhanced_available:
        run_step(
            "Theoretical Model Validation",
            ['python', 'scripts/theoretical_liv_models.py'],
            critical=False
        )
    
    total_time = time.time() - pipeline_start
    
    print(f"\n{'='*70}")
    print(f"✅ ENHANCED LIV ANALYSIS PIPELINE COMPLETED ({total_time:.1f}s)")
    print(f"{'='*70}")
    
    # Print summary
    print_summary()
    
    print("\nNext steps:")
    print("• Review results/grb_enhanced_bounds.csv for polynomial fit results")
    print("• Check results/uhecr_enhanced_exclusion.csv for theoretical model tests")
    print("• Examine combined_fom.csv for overall constraints")
    if enhanced_available:
        print("• Compare different theoretical model predictions")
        print("• Validate polynomial vs linear dispersion fits")

if __name__ == "__main__":
    main()
