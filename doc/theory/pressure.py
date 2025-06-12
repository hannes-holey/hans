# %% [markdown]
# # Pressure Calculation Models

# %% [markdown]
# Accurate pressure modeling is essential for simulating fluid behavior, especially in multiphase and compressible flows. 
# Various empirical and theoretical equations of state (EOS) relate pressure to density and other thermodynamic quantities. 
# This page provides an overview of the implemented models in the `models.pressure` module
#
# - Dowson-Higginson
# - Power Law
# - van der Waals
# - Murnaghan-Tait
# - Cubic Polynomial
# - BWR (Benedict-Webb-Rubin)
# - Bayada-Chupin
#
# We describe each model, explain its parameters, and plot pressure-density relationships over a representative range.

# %%
import hans.models.pressure as models_p
import numpy as np
import matplotlib.pyplot as plt

# Define common density range
densities = np.linspace(500, 1000, 100)  # kg/m^3

# Precompute pressures for all models
pressures = {
    "Dowson-Higginson": models_p.dowson_higginson(densities, 877.7007, 101325, 3.5e12, 1.23),
    "Power Law": models_p.power_law(densities, 877.7007, 101325, 7.0),
    "van der Waals": models_p.van_der_waals(densities, 18.01528, 300, 0.55, 3e-05),
    "Murnaghan-Tait": models_p.murnaghan_tait(densities, 1000, 1e5, 2.2e9, 7.15),
    "Cubic": models_p.cubic(densities, 1e-6, -1e-3, 2, 0),
    "Bayada-Chupin": models_p.bayada_chupin(densities, 850., 0.019, 1600., 352.),
}

# "BWR": models_p.bwr(densities, 300),

# %% [hide-input]
def plot_model(current_model):
    plt.figure(figsize=(10, 6))
    for name, P in pressures.items():
        if not name == current_model:
            continue
        alpha = 1.0 if name == current_model else 0.2
        lw = 2.5 if name == current_model else 1.0
        label = name if name == current_model else None
        plt.plot(densities, P, label=label, alpha=alpha, lw=lw)
    plt.xlabel("Density [kg/m³]")
    plt.ylabel("Pressure [Pa]")
    plt.title(f"Pressure vs Density – Highlighting {current_model}")
    plt.grid(True)
    plt.legend()
    plt.show()

# %% [markdown]
# ## Dowson-Higginson Model
#
# Used in hydrodynamic lubrication, this empirical model defines pressure with respect to density deviations.
#
# **Arguments:**
# - `dens`: Density [kg/m³]
# - `rho0`: Reference density (typically fluid at ambient pressure)
# - `P0`: Reference pressure (ambient or initial pressure)
# - `C1`, `C2`: Material constants related to compressibility and nonlinearity
#
# Typical values can be found in tribology and lubrication handbooks.


# %%
plot_model("Dowson-Higginson")

# %% [markdown]
# ## Power Law Model
#
# This simple model assumes pressure scales as a power of density.
#
# **Arguments:**
# - `dens`: Density [kg/m³]
# - `rho0`: Reference density
# - `P0`: Reference pressure
# - `alpha`: Exponent (often derived from compressibility characteristics)

# %%
plot_model("Power Law")

# %% [markdown]
# ## van der Waals Equation of State
#
# A thermodynamic model accounting for molecular attraction and volume exclusion.
#
# **Arguments:**
# - `dens`: Density [kg/m³]
# - `M`: Molar mass [g/mol] (e.g., 18.01528 for water)
# - `T`: Temperature [K]
# - `a`, `b`: Attraction and volume exclusion constants, respectively

# %%
plot_model("van der Waals")


# %% [markdown]
# ## Murnaghan-Tait Equation
#
# Widely used for liquids under pressure, e.g., in underwater acoustics.
#
# **Arguments:**
# - `dens`: Density [kg/m³]
# - `rho0`: Reference density
# - `P0`: Ambient pressure
# - `K`: Bulk modulus [Pa]
# - `n`: Empirical exponent (usually ~7 for water)

# %%
plot_model("Murnaghan-Tait")


# %% [markdown]
# ## Cubic Polynomial Model
#
# A generic polynomial fit for tabulated EOS data.
#
# **Arguments:**
# - `a`, `b`, `c`, `d`: Polynomial coefficients
# - `dens`: Density [kg/m³]

# %%
plot_model("Cubic")


# %% [markdown]
# ## BWR (Benedict-Webb-Rubin) Model
#
# A real-gas EOS used in high-pressure thermodynamics.
#
# **Arguments:**
# - `rho`: Density [kg/m³]
# - `T`: Temperature [K]
# - `gamma`: Optional compressibility factor (simplified in this case)

# %%
# plot_model("BWR")


# %% [markdown]
# ## Bayada-Chupin Model
#
# Computes pressure using the Bayada-Chupin cavitation model.
#
#    Models lubricated film pressure in the presence of phase change.
#    Reference: Bayada, G., & Chupin, L. (2013). *Journal of Tribology, 135(4), 041703*.
#
# **Arguments:**
# - `rho`: Density [kg/m³]
# - `rho_l`, `rho_v`: Liquid and vapor densities
# - `c_l`, `c_v`: Sound speeds in liquid and vapor phases

# %%
plot_model("Bayada-Chupin")