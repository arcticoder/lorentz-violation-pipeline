#!/usr/bin/env python3
import subprocess
import sys

def run_step(name, cmd):
    print(f"\n=== {name} ===")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during step: {name}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    run_step("UHECR spectrum calculation",    ['python', 'scripts/uhecr_spectrum.py'])
    run_step("GRB dispersion analysis",       ['python', 'scripts/analyze_grb.py'])
    run_step("UHECR LIV exclusion",           ['python', 'scripts/simulate_uhecr.py'])
    run_step("Combined FOM computation",      ['python', 'scripts/combined_fom.py'])
    print("\n✅ Full LIV‐analysis pipeline completed.")

if __name__ == "__main__":
    main()
