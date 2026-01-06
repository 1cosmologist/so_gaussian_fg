#!/usr/bin/env python
"""
Script to generate Gaussian foreground simulations using YAML configuration files.
"""

from pathlib import Path
from generate_gaussian_fg import GaussianForegroundSimulator


# ============== Configuration ==============
# Path to foreground parameters YAML file
FG_PARAMS = "../resources/config/fg_params.yaml"

# Path to instrument parameters YAML file
INSTR_PARAMS = "../resources/config/instr_params_baselineEHF_pessimistic.yaml"

# Number of simulations to generate
N_SIMS = 100

# Output directory for simulations
OUTPUT_DIR = "../output/foreground_sims/gaussian_fg_EHF/"
# ===========================================


def main():
    # Check if config files exist
    fg_params_path = Path(FG_PARAMS)
    instr_params_path = Path(INSTR_PARAMS)
    
    if not fg_params_path.exists():
        print(f"Warning: Foreground params file not found: {fg_params_path}")
        print("Using default foreground parameters.")
        fg_params_path = None
    
    if not instr_params_path.exists():
        print(f"Warning: Instrument params file not found: {instr_params_path}")
        print("Using default instrument parameters.")
        instr_params_path = None
    
    # Initialize simulator with YAML config files
    simulator = GaussianForegroundSimulator(
        fg_params=fg_params_path,
        instr_params=instr_params_path
    )
    
    # Run simulations
    simulator.run_simulations(n_sims=N_SIMS, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
