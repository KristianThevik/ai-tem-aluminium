#!/usr/bin/env python
# coding: utf-8

# The Strength Model
# This code uses the Holmedal model to calculate the total yield strength of an aluminium alloy condition.

import numpy as np
import pandas as pd
from testMaster.HolmedalStrengthModel import fit_omega, calculate_solid_solution_strength, calculate_yield_strength_single, omega, non_shearable_precipitate_integrand, shearable_precipitate_integrand

# Constants and alloy variables

kappa = 2   # Pinning force parameter
M = 2.7     # Taylor factor
G = 27E3    # [MPa] Shear modulus Al matrix
b = 0.286   # [nm] Burgers vector Al, b = a/sqrt(2)

alloy_composition = {'Mg': 0.61, 'Si': 1.11, 'Cu': 0.77, 'Fe': 0.20, 'Mn': 0.57, 'Cr': 0.13} # Found in SumAL
atomic_weights = {'Al': 26.982, 'Mg': 24.305, 'Si': 28.085, 'Cu': 63.546, 'Fe': 55.845, 'Mn': 54.938, 'Cr': 51.996}
volume_fraction = 1.11  # DOUBLE CHECK
strengthening_coefficients = {'Mg': 29.0, 'Si': 66.3, 'Cu': 46.4} # Found in Literature( reference )

# First run the percipitate statistics models, then load the csv files

# Length statistics
lengths_file = "statistics_lengths.csv"
df_length = pd.read_csv(lengths_file)
length_col = [col for col in df_length.columns if "Length" in col]
precipitate_lengths = df_length[length_col[0]].dropna().tolist()
mean_col_l = [col for col in df_length.columns if "Average" in col]
mean_length = df_length[mean_col_l[0]].dropna().iloc[0]
print(len(precipitate_lengths))
print(mean_length)

# Cross-section statistics
cross_section_file = "statistics_cross.csv"
df_cross = pd.read_csv(cross_section_file)
cross_col = [col for col in df_cross.columns if "Cross section" in col]
precipitate_cross = df_cross[cross_col[0]].dropna().tolist()
mean_col_c = [col for col in df_cross.columns if "Average" in col]
mean_cross = df_cross[mean_col_c[0]].dropna().iloc[0]
print(len(precipitate_cross))
print(mean_cross)

# Number statistics
dark_field_file = "statistics_df.csv"
df_cross_dark = pd.read_csv(dark_field_file)
density_col = [col for col in df_cross_dark.columns if "Number Density [nm^-2]" in col]
number_density = df_cross_dark[density_col[0]].dropna().iloc[0]
print(number_density)

critical_cross_section = mean_cross

# Compute aspect ratio parameters
aspect_ratio_params = fit_omega(precipitate_lengths, np.array(precipitate_lengths) / np.sqrt(mean_cross))

# Compute solid solution strengthening
sigma_ss = calculate_solid_solution_strength(alloy_composition, atomic_weights, volume_fraction, strengthening_coefficients)
print(f"Solid Solution Strengthening: {sigma_ss:.2f} MPa")

# Compute yield strength for a single condition
yield_strength = calculate_yield_strength_single(
    precipitate_lengths=precipitate_lengths,
    mean_length=mean_length,
    mean_cross_section= mean_cross,
    number_density= number_density,
    aspect_ratio_params=aspect_ratio_params,
    critical_cross_section= critical_cross_section,
    kappa=1.5,
    shear_modulus=G,
    burgers_vector=b,
    taylor_factor=M,
    solid_solution_strength=sigma_ss,
    base_strength=10,
    calibration_point=90,
    omega_func=omega,
    shearable_integrand=shearable_precipitate_integrand,
    non_shearable_integrand=non_shearable_precipitate_integrand
)

print(f"Yield Strength: {yield_strength:.2f} MPa")
