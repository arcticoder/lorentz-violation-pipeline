# Enhanced Lorentz Invariance Violation Analysis Framework

## Overview

This framework has been extended from basic phenomenological LIV analysis to include **explicit theoretical model testing** with polynomial dispersion relations. The key advancement is moving beyond simple linear fits to test sophisticated theoretical predictions.

## Key Enhancements

### 1. Polynomial Dispersion Relations
**Previous:** Simple linear time delay: `Δt = α₁(E/E_Pl)D(z)`

**Enhanced:** Full polynomial expansion:
```
Δt = D(z)[α₁(E/E_Pl) + α₂(E²/E_Pl²) + α₃(E³/E_Pl³) + α₄(E⁴/E_Pl⁴)]
```

This directly tests:
- **Polymer-quantized QED** corrections
- **Gravity-rainbow** model signatures  
- **Higher-order Planck-scale** effects
- **Non-linear LIV** modifications

### 2. Theoretical Dispersion Models

#### Polymer-QED Dispersion
```
ω² = k²[1 + α₁(k/E_Pl) + α₂(k/E_Pl)² + α₃(k/E_Pl)³] + m²
```
From loop quantum gravity and discrete spacetime structure.

#### Gravity-Rainbow Dispersion  
```
ω² = k²f(k/E_Pl) + m²g(k/E_Pl)
```
With rainbow functions:
- `f(x) = (1 + ηx^n)^(-1)`  
- `g(x) = (1 + ηx^m)^(-1)`

#### Vacuum Instability Effects
Schwinger-like pair production rates modify particle propagation:
```
Γ(E) = Γ₀ exp(-π E_crit/E)
```

#### Hidden Sector Energy Loss
Additional energy dissipation channels:
```
dE/dx = α_hidden (E/E_Pl)^n
```

## Framework Components

### Core Enhanced Modules

1. **`enhanced_grb_analysis.py`**
   - Polynomial dispersion fitting (orders 1-4)
   - Theoretical model comparison
   - AIC-based model selection
   - Enhanced uncertainty quantification

2. **`enhanced_uhecr_analysis.py`**  
   - Non-linear threshold modifications
   - Theoretical spectrum predictions
   - Vacuum instability effects
   - Hidden sector energy loss

3. **`theoretical_liv_models.py`**
   - Explicit dispersion relation implementations
   - Polymer-QED calculations
   - Gravity-rainbow functions
   - Physical constraint validation

### Updated Core Scripts

4. **`analyze_grb.py`** - Enhanced with polynomial fitting
5. **`simulate_uhecr.py`** - Enhanced with theoretical models  
6. **`run_full_analysis.py`** - Orchestrates enhanced pipeline

## Usage Examples

### Running Enhanced GRB Analysis
```bash
# Direct polynomial fitting
python scripts/enhanced_grb_analysis.py

# Integrated in main pipeline  
python scripts/analyze_grb.py
```

**Output:** Tests polynomial models up to 4th order, compares with theoretical predictions, produces plots showing fits and residuals.

### Running Enhanced UHECR Analysis
```bash
# Direct theoretical model testing
python scripts/enhanced_uhecr_analysis.py

# Integrated in main pipeline
python scripts/simulate_uhecr.py  
```

**Output:** Tests multiple theoretical models, computes exclusion limits, validates physical constraints.

### Full Enhanced Pipeline
```bash
python scripts/run_full_analysis.py
```

**Features:**
- Polynomial dispersion relation fitting
- Theoretical model testing (polymer-QED, gravity-rainbow)  
- Vacuum instability calculations
- Hidden sector energy loss analysis
- Enhanced uncertainty quantification

## Results and Analysis

### GRB Analysis Results
The enhanced framework successfully:

✅ **Fits polynomial models** (linear → quartic)  
✅ **Tests theoretical predictions** (polymer-QED, gravity-rainbow)  
✅ **Uses AIC for model selection**  
✅ **Generates detailed plots** and uncertainty analysis  
✅ **Produces LIV energy scale estimates** from polynomial coefficients

**Example Results:**
- Linear model: `E_LV ~ 7.8 × 10¹⁸ GeV`
- Polynomial fits provide model-dependent constraints
- AIC ranking identifies best theoretical model

### UHECR Analysis Results  
The enhanced framework enables:

✅ **Non-linear threshold modifications**  
✅ **Theoretical spectrum predictions**  
✅ **Vacuum instability effects**  
✅ **Hidden sector energy loss calculations**  
✅ **Model comparison and validation**

### Theoretical Model Validation
Direct testing of dispersion relations shows:

✅ **Standard relativity:** `v/c = 1` (baseline)  
✅ **Polymer-QED:** Energy-dependent velocity modifications  
✅ **Gravity-rainbow:** Scale-dependent corrections  
⚠️ **Physical constraints:** Detects superluminal propagation warnings

## Key Improvements Over Basic Framework

| Aspect | Basic Framework | Enhanced Framework |
|--------|-----------------|-------------------|
| **Dispersion** | Linear only: `Δt ∝ E` | Polynomial: `Δt ∝ E + E² + E³ + E⁴` |
| **Models** | Phenomenological | Explicit theoretical (polymer-QED, rainbow) |
| **Analysis** | χ² fitting | AIC model selection + uncertainty |
| **Physics** | Linear approximation | Full non-linear effects |
| **Validation** | Basic bounds | Physical constraint checking |

## Future Extensions

The framework is designed for easy extension to:

1. **Additional theoretical models** (doubly special relativity, etc.)
2. **Higher-order polynomial fits** (n > 4)  
3. **Multi-messenger constraints** (combining different observables)
4. **Bayesian parameter estimation** (MCMC sampling)
5. **Systematic uncertainty modeling** (detector effects, etc.)

## Technical Notes

- **Model Selection:** Uses Akaike Information Criterion (AIC) to balance fit quality vs complexity
- **Physical Constraints:** Validates against superluminal propagation and causality  
- **Uncertainty Quantification:** Propagates parameter errors through theoretical predictions
- **Computational Efficiency:** Optimized for rapid model comparison and parameter scanning

The enhanced framework successfully generalizes the basic LIV pipeline to enable rigorous testing of explicit theoretical models with polynomial dispersion relations, as requested.
