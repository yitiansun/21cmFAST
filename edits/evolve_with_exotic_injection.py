"""
Step-by-step evolution script for 21cmFAST with exotic energy injection

This script demonstrates how to inject exotic energy into:
- Heating (affects gas kinetic temperature)
- Ionization (affects ionized fraction)
- Lyman-alpha flux (affects spin temperature via Wouthuysen-Field coupling)

Based on edits/evolve.py
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
output_dir = Path("output_plots_exotic")
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
cache_dir = Path("cache_exotic")
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
    print(f"STEP {i+1}: Computing fields at z = {redshift:.2f} WITH EXOTIC INJECTION")
    print("="*60)

    # Perturb the field to this redshift
    print(f"  - Perturbing field to z = {redshift:.2f}...")
    perturbed_field = p21c.perturb_field(
        redshift=redshift,
        initial_conditions=initial_conditions
    )

    # ===================================================================
    # CREATE EXOTIC ENERGY INJECTION BOXES
    # ===================================================================
    print(f"  - Creating exotic energy injection boxes...")

    # Initialize the input boxes (creates arrays filled with zeros)
    input_heating = p21c.InputHeating.new(inputs=inputs, redshift=redshift)
    input_ionization = p21c.InputIonization.new(inputs=inputs, redshift=redshift)
    input_jalpha = p21c.InputJAlpha.new(inputs=inputs, redshift=redshift)

    # Compute (initializes the C structures)
    input_heating.compute()
    input_ionization.compute()
    input_jalpha.compute()

    # Now modify the arrays to add exotic injection
    # These are 3D arrays with shape (HII_DIM, HII_DIM, HII_DIM)

    # Example 1: Uniform injection everywhere
    # Add uniform heating: 100 K per timestep
    input_heating.input_heating[:] = 100.0

    # Add uniform ionization increase: 0.001 per timestep
    input_ionization.input_ionization[:] = 0.001

    # Add uniform Lyman-alpha flux increase
    input_jalpha.input_jalpha[:] = 1e-12

    # Example 2: Spatial variation - stronger injection in overdensities
    # Get the density field
    density = perturbed_field.get("density")

    # Scale injection by local overdensity (1 + delta)
    # Positive values = overdense regions get more injection
    delta = density / np.mean(density) - 1.0

    # Apply spatial scaling (only where overdense, delta > 0)
    input_heating.input_heating[:] = 50.0 * np.maximum(delta, 0)
    input_ionization.input_ionization[:] = 0.0005 * np.maximum(delta, 0)
    input_jalpha.input_jalpha[:] = 5e-13 * np.maximum(delta, 0)

    # Example 3: Redshift-dependent injection
    # Stronger at higher redshifts
    z_factor = (redshift / 20.0) ** 2
    input_heating.input_heating[:] *= z_factor
    input_ionization.input_ionization[:] *= z_factor
    input_jalpha.input_jalpha[:] *= z_factor

    # Push modified arrays back to C backend
    input_heating.push_to_backend()
    input_ionization.push_to_backend()
    input_jalpha.push_to_backend()

    print(f"     Exotic injection applied:")
    print(f"       Heating: mean = {np.mean(input_heating.input_heating):.2f} K")
    print(f"       Ionization: mean = {np.mean(input_ionization.input_ionization):.6f}")
    print(f"       J_alpha: mean = {np.mean(input_jalpha.input_jalpha):.3e}")

    # ===================================================================
    # NOTE: The compute_ionization_field function would need to be updated
    # to accept input_heating, input_ionization, and input_jalpha parameters.
    # This requires modifications to the drivers/single_field.py module.
    #
    # For now, this demonstrates the API. The actual integration requires:
    # 1. Updating drivers/single_field.py compute_ionization_field() signature
    # 2. Ensuring the C code in IonisationBox.c and SpinTemperatureBox.c
    #    uses these input structures (see TODO items below)
    # ===================================================================

    print(f"  - Computing ionization field...")
    print(f"     WARNING: Full integration not yet complete!")
    print(f"     The C code needs to be updated to use the injection boxes.")

    # This call will work once the C code integration is complete:
    # ionized_field = p21c.compute_ionization_field(
    #     initial_conditions=initial_conditions,
    #     perturbed_field=perturbed_field,
    #     input_heating=input_heating,
    #     input_ionization=input_ionization,
    #     input_jalpha=input_jalpha,
    # )

    # For now, compute without exotic injection
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

print("\n" + "="*60)
print("EXOTIC INJECTION DEMONSTRATION COMPLETE!")
print("="*60)
print("\nNext steps to complete the integration:")
print("1. Update src/py21cmfast/src/IonisationBox.c to use input_ionization")
print("2. Update src/py21cmfast/src/SpinTemperatureBox.c to use input_heating and input_jalpha")
print("3. Update drivers/single_field.py to pass input structures to C functions")
print("\nSee the v3 implementation in previous_changes/ for reference on how to")
print("integrate the input structures into the evolution equations.")

# Extract arrays and plot (same as evolve.py)
print("\n" + "="*60)
print("Extracting arrays and computing global scales...")
print("="*60)

neutral_fraction_slices = []
kinetic_temp_slices = []

for ionized_field in ionized_fields:
    neutral_frac_array = ionized_field.get("neutral_fraction")
    kinetic_temp_array = ionized_field.get("kinetic_temperature")

    mid_idx = neutral_frac_array.shape[2] // 2
    neutral_fraction_slices.append(neutral_frac_array[:, :, mid_idx])
    kinetic_temp_slices.append(kinetic_temp_array[:, :, mid_idx])

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
        slice_data.T,
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
cbar1 = fig.colorbar(im1, ax=axes[0, :], orientation='vertical',
                     pad=0.05, aspect=40, shrink=0.8)
cbar1.set_label('Neutral Fraction', fontsize=11)

# Plot kinetic temperatures (row 1)
for i, (redshift, slice_data) in enumerate(zip(redshifts, kinetic_temp_slices)):
    im2 = axes[1, i].imshow(
        slice_data.T,
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
cbar2 = fig.colorbar(im2, ax=axes[1, :], orientation='vertical',
                     pad=0.05, aspect=40, shrink=0.8)
cbar2.set_label('Kinetic Temperature [K]', fontsize=11)

# Overall title
fig.suptitle('21cmFAST Evolution with Exotic Energy Injection (Demo)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the combined plot
output_file = output_dir / "evolution_exotic_injection.png"
fig.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"Saved: {output_file}")

print(f"\nPlot saved to: {output_file.absolute()}")
