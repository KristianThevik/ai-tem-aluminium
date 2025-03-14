#!/usr/bin/env python
# coding: utf-8

# The Strength Model
# This code uses the Holmedal model to calculate the total yield strength of an aluminium alloy condition.

import numpy as np
from scipy.optimize import least_squares
from scipy import special
from scipy.integrate import quad

def omega(x: list, l):
    """
    Computes the aspect ratio function, modeled using a power law:

        Ω(l) = a * l^b

    where:
        - Ω(l) is the aspect ratio of the precipitate.
        - 'a' (x[0]) is the scaling factor.
        - 'b' (x[1]) is the exponent.
        - 'l' is the precipitate length (scalar or array).

    The theoretical aspect ratio for a cylindrical precipitate is:

        Ω = l / sqrt(A)

    Instead of using this directly, we approximate the aspect ratio using a power-law function.

    To avoid values below 1, we enforce Ω(l) ≥ 1.
    """
    arr = x[0] * np.power(l, x[1])  # Compute the power law function

    # Ensure all values in arr are at least 1
    if isinstance(arr, np.ndarray):  
        arr[arr < 1] = 1  
    else:  
        arr = max(arr, 1)  # Ensures scalars do not fall below 1

    return arr


def fit_omega(l, aspect_ratio):
    """
    Finds the best fit for the aspect ratio function Ω(l) = a * l^b.
    
    l: Array of measured precipitate lengths
    aspect_ratio: Corresponding measured aspect ratios (l / sqrt(A))?

    Returns optimized values of a and b.
    """
    x0_fit = [0.7, 0.7]  # Initial guess for parameters (a, b)
    residual = lambda x: aspect_ratio - omega(x, l)  # Residual function
    result = least_squares(residual, x0_fit)  # Perform least-squares optimization
    return result.x  # Return fitted parameters


def f(a, a_c, kappa):
    """
    Input:
        a     : Precipitate cross-section [nm²]
        a_c   : Critical cross-section defining transition between 
                shearable and non-shearable precipitates [nm²]
        kappa : Exponent controlling scaling behavior

    Output:
        f     : Obstacle length
    """
    return np.min([(a / a_c) ** kappa, 1])


def tau_p(alpha_p, G, b, n_p, f_bar):
    """
    Strength contribution from precipitate-based obstacles.

    Input:
        alpha_p : Scaling factor
        G       : Shear modulus [MPa]
        b       : Burgers vector [nm]
        n_p     : Number density of precipitate-based obstacles per slip plane [#/nm²]
        f_bar   : Mean obstacle length (dimensionless)

    Output:
        tau_p   : Strength contribution from obstacles [MPa]
    """
    return alpha_p * G * b * np.sqrt(n_p) * f_bar**(3/2) * (1 - 1/6 * f_bar**5)


def phi_tilde(l: float, lengths_data: float, h: float):
    """
    Uncorrected distribution of precipitate lengths.

    Input:
        l  : The length interval to evaluate the distribution of lengths at
        lengths_data : Precipitate length data used to fit the distribution
        h  : Kernel bandwidth for smoothing, determined using Scott's rule

    Output:
        Probability density at length l (before correction)
    """
    return (1 / len(lengths_data)) * sum(
        np.sqrt(2) * np.exp(-0.5 * (([l] * len(lengths_data) - lengths_data) / h) ** 2) /
        ((1 + special.erf(lengths_data / (np.sqrt(2) * h))) * h * np.sqrt(np.pi))
    )


def phi(l, length_data, h):
    """
    Normalized precipitate length distribution.

    This function corrects the unnormalized kernel density estimate 
    phi_tilde, ensuring that the precipitate length distribution meets 
    the required boundary condition, reaching zero at l = 0.

    Input:
        l           : The length interval to evaluate the distribution of lengths at
        length_data : Array of measured precipitate lengths
        h           : Kernel bandwidth for smoothing, determined using Scott's rule

    Output:
        phi         : Normalized probability density of precipitate lengths [1/nm]

    """
    return (phi_tilde(l, length_data, h) - phi_tilde(0, length_data, h) * np.exp(-0.5 * (l / h) ** 2)) / \
           (1 - 0.5 * h * np.sqrt(2 * np.pi) * phi_tilde(0, length_data, h))


def calculated_kernel_bandwidth(length_data):
    """
    Computes the kernel bandwidth (h) for a single experimental condition 
    using Scott's rule:

        h ≈ d * N_l^(-0.2) * sigma_l

    where:
        - h      : Kernel bandwidth for KDE smoothing [nm]
        - d      : Empirical scaling factor (0.8)
        - N_l    : Number of measured precipitate lengths
        - sigma_l: Standard deviation of precipitate lengths

    Input:
        length_data : List or array of measured precipitate length values.

    Output:
        h          : Calculated bandwidth (h) for the given condition.
    """
    sigma = np.std(length_data)  # Compute standard deviation (σ_l)
    N_l = len(length_data)       # Get number of data points (N_l)

    # Apply Scott's rule: h = d * N_l^(-0.2) * σ_l
    h = 0.8 * N_l**(-0.2) * sigma

    return h  # Return the bandwidth value


def shearable_precipitate_integrand(l, aspect_ratio_params, kappa, length_distribution):
    """
    Computes the integrand for the shearable precipitate contribution.

    Based on the integral term:
        ∫ (l^(2κ+1) / Ω(l)^(2κ)) * φ(l) dl

    where:
        - l                   : Precipitate length [nm]
        - aspect_ratio_params  : Parameters for the aspect ratio function Ω(l)
        - kappa               : Scaling exponent for strength model
        - length_distribution : Normalized precipitate length distribution φ(l)

    Output:
        Contribution of shearable precipitates to the mean obstacle strength.
    """
    return (l**(2*kappa + 1) / omega(aspect_ratio_params, l)**(2*kappa)) * length_distribution(l)


def non_shearable_precipitate_integrand(l, length_distribution):
    """
    Computes the integrand for the non-shearable precipitate contribution.

    Based on the integral term:
        ∫ l * φ(l) dl

    where:
        - l                   : Precipitate length [nm]
        - length_distribution : Normalized precipitate length distribution φ(l)

    Output:
        Contribution of non-shearable precipitates to the mean obstacle strength.
    """
    return l * length_distribution(l)


def weight_to_atomic_fraction(weight_percent, atomic_weights, element):
    """
    Convert weight percent (wt%) to atomic fraction (at%).

    Input:
        weight_percent  : Dictionary containing element weight fractions {Element: wt%}
        atomic_weights  : Dictionary containing atomic weights {Element: atomic weight}
        element         : The element for which conversion is performed

    Output:
        atomic_fraction : Atomic fraction (at%) of the specified element
    """
    return (weight_percent[element] / atomic_weights[element]) / \
           sum(weight_percent[el] / atomic_weights[el] for el in atomic_weights)


def atomic_to_weight_fraction(atomic_fraction, atomic_weights, element, solid_fraction):
    """
    Convert atomic fraction (at%) to weight percent (wt%).

    Input:
        atomic_fraction : Dictionary containing atomic fractions {Element: at%}
        atomic_weights  : Dictionary containing atomic weights {Element: atomic weight}
        element         : The element for which conversion is performed
        solid_fraction  : Scaling factor for volume fraction corrections

    Output:
        weight_percent  : Weight percent (wt%) of the specified element
    """
    return (solid_fraction * atomic_weights[element]) / \
           sum(weight_to_atomic_fraction(atomic_fraction, atomic_weights, el) * atomic_weights[el] 
               for el in atomic_weights)


# Critical cross section is here just taken to be the mean cross section. It should be
# the mean cross section at peak at the ageing condition near peak strength.

def calculate_solid_solution_strength(alloy_composition, atomic_weights, volume_fraction, strengthening_coefficients):
    """
    Calculates the solid solution strengthening contribution from alloying elements for a **single alloy condition**.

    This function follows the same approach as `calculate_solid()` to ensure identical results.

    Input:
        alloy_composition         : Dictionary containing element weight fractions {Element: wt%}
        atomic_weights            : Dictionary containing atomic weights {Element: atomic weight}
        volume_fraction           : Precipitate volume fraction (single float value, not a list)
        strengthening_coefficients: Dictionary of strengthening coefficients {Element: MPa}

    Output:
        sigma_ss                  : Solid solution strengthening contribution [MPa]
    """

    # Convert volume fraction to solid fraction adjustment (single value)
    solid_fraction = np.array(volume_fraction) * 22 / 24  

    # Ensure the aluminum fraction is calculated correctly
    alloy_composition['Al'] = 100 - sum(alloy_composition.values())

    # Manually enforce Si content (specific assumption from `calculate_solid()`)
    alloy_composition['Si'] = 0.95

    # **Use the same betaDP fractions as `calculate_solid()`**
    betaDP = {'Mg': 0.42, 'Si': 0.30, 'Cu': 0.03}  

    # Compute weight percent of each element remaining in solid solution (direct subtraction)
    weight_mg = alloy_composition['Mg'] - (betaDP['Mg'] * solid_fraction)
    weight_si = alloy_composition['Si'] - (betaDP['Si'] * solid_fraction)
    weight_cu = alloy_composition['Cu'] - (betaDP['Cu'] * solid_fraction)

    # Ensure Cu content does not go negative
    weight_cu = max(0, weight_cu)  

    # Compute the solid solution strengthening contribution using the power law relation
    sigma_ss = (
        strengthening_coefficients['Mg'] * weight_mg**(2/3) +
        strengthening_coefficients['Si'] * weight_si**(2/3) +
        strengthening_coefficients['Cu'] * weight_cu**(2/3)
    )

    return sigma_ss  # Single scalar value


def calculate_yield_strength_single(precipitate_lengths, mean_length, mean_cross_section,
                                    number_density, aspect_ratio_params, critical_cross_section, 
                                    kappa, shear_modulus, burgers_vector, taylor_factor, 
                                    solid_solution_strength, base_strength, calibration_point,
                                    omega_func, shearable_integrand, non_shearable_integrand):
    """
    Computes the yield strength for a single alloy condition.

    Input:
        precipitate_lengths      : List of precipitate lengths for the alloy condition [nm]
        mean_length              : Mean precipitate length in this condition [nm]
        mean_cross_section       : Mean precipitate cross-section in this condition [nm²]
        number_density           : Number density of precipitates in this condition [#/nm³]
        aspect_ratio_params      : Parameters for the aspect ratio function Ω(l)
        critical_cross_section   : Critical cross-section a_c defining shearable/non-shearable transition [nm²]
        kappa                   : Scaling exponent for strength model
        shear_modulus            : Shear modulus G [MPa]
        burgers_vector           : Burgers vector b [nm]
        taylor_factor            : Taylor factor M (polycrystalline strengthening factor)
        solid_solution_strength  : Solid solution strengthening contribution [MPa]
        base_strength            : Baseline yield strength σ₀ [MPa]
        calibration_point        : Experimental calibration Vickers hardness (HV) for this condition
        omega_func               : Function to compute aspect ratio Ω(l)
        shearable_integrand      : Function computing the integral for shearable precipitates
        non_shearable_integrand  : Function computing the integral for non-shearable precipitates

    Output:
        yield_strength           : Calibrated yield strength [MPa]
    """

    # Solve for the critical length l_c using least squares
    residual_func = lambda l: np.sqrt(critical_cross_section) * omega_func(aspect_ratio_params, l) - l
    critical_length = least_squares(residual_func, 16).x[0]  

    # Compute kernel bandwidth for KDE smoothing
    kernel_bandwidth = 0.8 * len(precipitate_lengths)**(-0.2) * np.std(precipitate_lengths)

    # Compute mean obstacle strength f_bar
    # ADD CORRECT ARGUMENTS IN INTEGRAND
    f_bar = (quad(shearable_integrand, 0, critical_length, args=(precipitate_lengths, kernel_bandwidth))[0] / (critical_cross_section**kappa) +
             quad(non_shearable_integrand, critical_length, 1000, args=(precipitate_lengths, kernel_bandwidth))[0]) / mean_length

    # Compute number density of precipitate-based obstacles per slip plane
    obstacle_density = (np.sqrt(3) / 3) * mean_length * number_density

    # Compute precipitate strengthening contribution σ_p
    sigma_p = (taylor_factor * shear_modulus * burgers_vector * np.sqrt(obstacle_density) * 
               f_bar**(3/2) * (1 - (1/6) * f_bar**5))

    # Apply calibration using experimental hardness data REMOVE OR FIND IN LITERATURE, BASE_STRENGTH ASWELL
    if calibration_point < 1:  # Calibration constant is directly given
        calibration_factor = calibration_point
    else:  # Compute calibration constant from experimental hardness
        calibration_factor = ((calibration_point - 16) * 3 - solid_solution_strength - base_strength) / sigma_p

    # Compute final yield strength
    yield_strength = (sigma_p * calibration_factor + solid_solution_strength + base_strength) / 3 + 16

    return yield_strength
