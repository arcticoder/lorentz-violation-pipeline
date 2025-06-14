# Lorentz-Violation Pipeline

End-to-end toolkit for setting bounds on Planck-scale Lorentz invariance violation using:

- **GRB dispersion** (Fermi-LAT time-tagged events)
- **UHECR threshold shifts** (Pierre Auger spectrum)

## Setup

```bash
git clone https://github.com/arcticoder/lorentz-violation-pipeline
cd lorentz-violation-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place your data under `data/`:

```cpp
data/
  grbs/
    GRB090510.csv      # columns: delta_E_GeV, delta_t_s
    GRB221009A.csv
  uhecr/
    auger_spectrum.csv # columns: energy_EeV, flux, error
```

## Usage

### GRB analysis

```bash
python scripts/analyze_grb.py
```

### UHECR exclusion

```bash
python scripts/simulate_uhecr.py
```

### Combined FOM

```bash
python scripts/combined_fom.py
```

Results will be printed as tables—you can redirect or extend them to CSV/plots.

```