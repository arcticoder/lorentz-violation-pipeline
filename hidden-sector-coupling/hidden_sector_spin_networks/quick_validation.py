#!/usr/bin/env python3
"""
Quick LV Framework Validation
============================

This script performs a quick validation of all five LV pathways
without heavy computational overhead.
"""

import numpy as np
import sys
import traceback

def test_matter_gravity_coherence():
    """Test matter-gravity coherence pathway."""
    try:
        from matter_gravity_coherence import MatterGravityCoherence, MatterGravityConfig
        
        config = MatterGravityConfig(
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        
        system = MatterGravityCoherence(config)
        
        # Basic functionality tests
        assert system.is_pathway_active() == True
        
        # Test simple calculations
        fidelity = system.entanglement_fidelity_evolution(0.1)
        assert 0 <= fidelity <= 1
        
        power = system.total_extractable_power()
        assert power > 0
        
        print("✅ Matter-Gravity Coherence: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Matter-Gravity Coherence: FAILED - {e}")
        return False

def test_casimir_lv():
    """Test Casimir LV pathway."""
    try:
        from casimir_lv import CasimirLVCalculator, CasimirLVConfig
        
        config = CasimirLVConfig(
            mu=1e-18,
            alpha=1e-15,
            beta=1e-12
        )
        
        system = CasimirLVCalculator(config)
        
        # Basic functionality tests
        assert system.is_pathway_active() == True
        
        # Test enhancement factor (simple calculation)
        enhancement = system.lv_enhancement_factor()
        assert enhancement > 1.0
        
        print("✅ Casimir LV: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Casimir LV: FAILED - {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_hidden_sector_portal():
    """Test Hidden Sector Portal pathway."""
    try:
        from hidden_sector_portal import HiddenSectorPortal, HiddenSectorConfig
        
        config = HiddenSectorConfig(
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        
        system = HiddenSectorPortal(config)
        
        # Basic functionality tests
        assert system.is_pathway_active() == True
        
        # Test simple power calculation
        power = system.total_power_extraction()
        assert power > 0
        
        print("✅ Hidden Sector Portal: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Hidden Sector Portal: FAILED - {e}")
        return False

def test_axion_coupling():
    """Test Axion Coupling pathway."""
    try:
        from axion_coupling_lv import AxionCouplingLV, AxionCouplingConfig
        
        config = AxionCouplingConfig(
            mu_lv=1e-18,
            alpha_lv=1e-15,
            beta_lv=1e-12
        )
        
        system = AxionCouplingLV(config)
        
        # Basic functionality tests
        assert system.is_pathway_active() == True
        
        # Test simple power calculation
        power = system.coherent_oscillation_power()
        assert power > 0
        
        print("✅ Axion Coupling LV: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Axion Coupling LV: FAILED - {e}")
        return False

def test_dynamic_casimir():
    """Test Dynamic Casimir pathway (basic only)."""
    try:
        from dynamic_casimir_lv import DynamicCasimirLV, DynamicCasimirConfig
        from casimir_lv import CasimirLVConfig
        
        casimir_config = CasimirLVConfig(
            mu=1e-18,
            alpha=1e-15,
            beta=1e-12
        )
        
        config = DynamicCasimirConfig(
            casimir_config=casimir_config,
            drive_frequency=1e10
        )
        
        system = DynamicCasimirLV(config)
        
        # Basic functionality tests
        assert system.is_pathway_active() == True
        
        print("✅ Dynamic Casimir LV: PASSED (basic)")
        return True
        
    except Exception as e:
        print(f"❌ Dynamic Casimir LV: FAILED - {e}")
        return False

def main():
    """Run quick validation of all pathways."""
    print("=== QUICK LV FRAMEWORK VALIDATION ===")
    print("Testing basic functionality of all five pathways...\n")
    
    results = []
    
    # Test each pathway
    results.append(test_matter_gravity_coherence())
    results.append(test_casimir_lv())
    results.append(test_hidden_sector_portal())
    results.append(test_axion_coupling())
    results.append(test_dynamic_casimir())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Pathways tested: {total}")
    print(f"Pathways passed: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL PATHWAYS VALIDATED - Framework ready!")
    else:
        print(f"\n⚠️  {total - passed} pathway(s) need attention")
    
    return passed == total

if __name__ == "__main__":
    main()
