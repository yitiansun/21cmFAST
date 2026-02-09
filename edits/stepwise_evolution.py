"""
Step-by-step evolution script for 21cmFAST
Based on the coeval_cubes.ipynb tutorial

This script performs evolution at multiple redshifts and plots:
- Neutral fraction slices
- Gas kinetic temperature slices

Outputs are saved to the 'output_plots' directory.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import py21cmfast as p21c
from py21cmfast import plotting

# Set up logging
logger = logging.getLogger("py21cmfast")
logger.setLevel(logging.INFO)

# Create output directory
output_dir = Path("output_plots")
output_dir.mkdir(exist_ok=True)

print(f"Using 21cmFAST version {p21c.__version__}")
print(f"Output plots will be saved to: {output_dir.absolute()}")

# Define redshifts for evolution
redshifts = [20, 18, 16, 14, 12]

# Set up input parameters (small and fast for demonstration)
inputs = p21c.InputParameters.from_template(
    ["simple", "small"], random_seed=1234
)

# Create a cache directory
cache_dir = Path("cache_stepwise")
cache_dir.mkdir(exist_ok=True)
cache = p21c.OutputCache(cache_dir)

print("\n" + "="*60)
print("STEP 1: Computing Initial Conditions")
print("="*60)
initial_conditions = p21c.compute_initial_conditions(
    inputs=inputs,
    cache=cache,
    write=True
)
print(f"Initial conditions computed with random seed: {inputs.random_seed}")

# Loop through each redshift
for i, redshift in enumerate(redshifts, 1):
    print("\n" + "="*60)
    print(f"STEP {i+1}: Computing fields at z = {redshift}")
    print("="*60)

    # Perturb the field to this redshift
    print(f"  - Perturbing field to z = {redshift}...")
    perturbed_field = p21c.perturb_field(
        redshift=redshift,
        initial_conditions=initial_conditions
    )

    # Compute ionization field
    print(f"  - Computing ionization field...")
    ionized_field = p21c.compute_ionization_field(
        initial_conditions=initial_conditions,
        perturbed_field=perturbed_field
    )

    # Compute brightness temperature
    print(f"  - Computing brightness temperature...")
    brightness_temp = p21c.brightness_temperature(
        ionized_box=ionized_field,
        perturbed_field=perturbed_field
    )

    # Create plots
    print(f"  - Creating plots...")

    # Plot 1: Neutral Fraction
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7))
    plotting.coeval_sliceplot(
        ionized_field,
        kind="neutral_fraction",
        ax=ax1,
        fig=fig1
    )
    ax1.set_title(f"Neutral Fraction at z = {redshift}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save neutral fraction plot
    neutral_fraction_file = output_dir / f"neutral_fraction_z{redshift}.png"
    fig1.savefig(neutral_fraction_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {neutral_fraction_file}")
    plt.close(fig1)

    # Plot 2: Gas Kinetic Temperature
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    plotting.coeval_sliceplot(
        ionized_field,
        kind="kinetic_temperature",
        ax=ax2,
        fig=fig2
    )
    ax2.set_title(f"Gas Kinetic Temperature at z = {redshift}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save kinetic temperature plot
    kinetic_temp_file = output_dir / f"kinetic_temperature_z{redshift}.png"
    fig2.savefig(kinetic_temp_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {kinetic_temp_file}")
    plt.close(fig2)

    # Optional: Create a combined plot with both fields side by side
    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(14, 6))

    plotting.coeval_sliceplot(
        ionized_field,
        kind="neutral_fraction",
        ax=ax3_1,
        fig=fig3
    )
    ax3_1.set_title(f"Neutral Fraction", fontsize=12)

    plotting.coeval_sliceplot(
        ionized_field,
        kind="kinetic_temperature",
        ax=ax3_2,
        fig=fig3
    )
    ax3_2.set_title(f"Gas Kinetic Temperature", fontsize=12)

    fig3.suptitle(f"z = {redshift}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save combined plot
    combined_file = output_dir / f"combined_z{redshift}.png"
    fig3.savefig(combined_file, dpi=150, bbox_inches='tight')
    print(f"    Saved: {combined_file}")
    plt.close(fig3)

print("\n" + "="*60)
print("EVOLUTION COMPLETE!")
print("="*60)
print(f"\nAll plots saved to: {output_dir.absolute()}")
print(f"Total redshifts computed: {len(redshifts)}")
print(f"Redshifts: {redshifts}")
print(f"\nFiles created:")
for redshift in redshifts:
    print(f"  - neutral_fraction_z{redshift}.png")
    print(f"  - kinetic_temperature_z{redshift}.png")
    print(f"  - combined_z{redshift}.png")
