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
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import py21cmfast as p21c

# Set up logging
logger = logging.getLogger("py21cmfast")
logger.setLevel(logging.INFO)

# Create output directory
output_dir = Path("output_plots")
output_dir.mkdir(exist_ok=True)

print(f"Using 21cmFAST version {p21c.__version__}")
print(f"Output plots will be saved to: {output_dir.absolute()}")

# Define redshifts for evolution
redshifts = np.linspace(20, 19, 5)

# Set up input parameters (small and fast for demonstration)
inputs = p21c.InputParameters.from_template(
    ["simple", "small"], random_seed=1234
)

# Create a cache directory and clear it
cache_dir = Path("cache_stepwise")
if cache_dir.exists():
    print(f"\nClearing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)
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

# Store ionized fields for all redshifts
ionized_fields = []

# Loop through each redshift and compute fields
for i, redshift in enumerate(redshifts, 1):
    print("\n" + "="*60)
    print(f"STEP {i+1}: Computing fields at z = {redshift:.2f}")
    print("="*60)

    # Perturb the field to this redshift
    print(f"  - Perturbing field to z = {redshift:.2f}...")
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

    ionized_fields.append(ionized_field)

# Extract arrays and compute global min/max for consistent colorbars
print("\n" + "="*60)
print("Extracting arrays and computing global scales...")
print("="*60)

neutral_fraction_slices = []
kinetic_temp_slices = []

for ionized_field in ionized_fields:
    # Get the middle slice for visualization
    neutral_frac_array = ionized_field.get("neutral_fraction")
    kinetic_temp_array = ionized_field.get("kinetic_temperature")

    # Take middle slice along z-axis
    mid_idx = neutral_frac_array.shape[2] // 2
    neutral_fraction_slices.append(neutral_frac_array[:, :, mid_idx])
    kinetic_temp_slices.append(kinetic_temp_array[:, :, mid_idx])

# Compute global min/max for each field type
neutral_vmin = min(slice.min() for slice in neutral_fraction_slices)
neutral_vmax = max(slice.max() for slice in neutral_fraction_slices)
temp_vmin = min(slice.min() for slice in kinetic_temp_slices)
temp_vmax = max(slice.max() for slice in kinetic_temp_slices)

print(f"Neutral fraction range: [{neutral_vmin:.4f}, {neutral_vmax:.4f}]")
print(f"Kinetic temperature range: [{temp_vmin:.2f}, {temp_vmax:.2f}] K")

# Create the 2x5 plot
print("\n" + "="*60)
print("Creating combined evolution plot...")
print("="*60)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Plot neutral fractions (row 0)
for i, (redshift, slice_data) in enumerate(zip(redshifts, neutral_fraction_slices)):
    im1 = axes[0, i].imshow(
        slice_data.T,  # Transpose for correct orientation
        origin='lower',
        cmap='viridis',
        vmin=neutral_vmin,
        vmax=neutral_vmax,
        extent=[0, inputs.simulation_options.BOX_LEN,
                0, inputs.simulation_options.BOX_LEN]
    )
    axes[0, i].set_title(f"z = {redshift:.2f}", fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel("Mpc", fontsize=10)
    if i == 0:
        axes[0, i].set_ylabel("Neutral Fraction\nMpc", fontsize=10)

# Add colorbar for neutral fraction
cbar1 = fig.colorbar(im1, ax=axes[0, :], orientation='horizontal',
                     pad=0.05, aspect=40, shrink=0.8)
cbar1.set_label('Neutral Fraction', fontsize=11)

# Plot kinetic temperatures (row 1)
for i, (redshift, slice_data) in enumerate(zip(redshifts, kinetic_temp_slices)):
    im2 = axes[1, i].imshow(
        slice_data.T,  # Transpose for correct orientation
        origin='lower',
        cmap='inferno',
        vmin=temp_vmin,
        vmax=temp_vmax,
        extent=[0, inputs.simulation_options.BOX_LEN,
                0, inputs.simulation_options.BOX_LEN]
    )
    axes[1, i].set_xlabel("Mpc", fontsize=10)
    if i == 0:
        axes[1, i].set_ylabel("Kinetic Temperature\nMpc", fontsize=10)

# Add colorbar for kinetic temperature
cbar2 = fig.colorbar(im2, ax=axes[1, :], orientation='horizontal',
                     pad=0.05, aspect=40, shrink=0.8)
cbar2.set_label('Kinetic Temperature [K]', fontsize=11)

# Overall title
fig.suptitle('21cmFAST Evolution: Neutral Fraction and Gas Kinetic Temperature',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the combined plot
output_file = output_dir / "evolution_combined.png"
fig.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"Saved: {output_file}")

print("\n" + "="*60)
print("EVOLUTION COMPLETE!")
print("="*60)
print(f"\nPlot saved to: {output_file.absolute()}")
print(f"Total redshifts computed: {len(redshifts)}")
print(f"Redshifts: {[f'{z:.2f}' for z in redshifts]}")
