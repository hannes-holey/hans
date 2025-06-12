# %% [markdown]
# # Density Calculation Models

# %% [markdown]
#
# The density calculation may be required e.g. to calculate density boundary values from user-specified pressure boundary values
# This page provides an overview of the implemented models in the `models.density` module
#
# The following models are implemented and compared:
# - Dowson–Higginson
# - Power Law
# - Murnaghan–Tait
# - Bayada–Chupin

# %%
import hans.models.density as models_rho
import numpy as np
import matplotlib.pyplot as plt

# Pressure range for evaluation [Pa]
pressures = np.linspace(1e4, 2e7, 100)
    
densities = {
    "Dowson–Higginson": models_rho.dowson_higginson(pressures, rho0=850., P0=1e5, C1=3.5e12, C2=1.23),
    "Power Law":        models_rho.power_law(pressures, rho0=850., P0=1e5, alpha=6.0),
    "Murnaghan–Tait":   models_rho.murnaghan_tait(pressures, rho0=850., P0=1e5, K=2.2e9, n=7.15),
    "Bayada–Chupin":    models_rho.bayada_chupin(pressures, rho_l=850., rho_v=0.019, c_l=1600., c_v=352.)
}

# %%
def plot_density_models(pressures, densities, highlight_model):
    """
    Plots all density models over pressure range.
    Highlights one model, greys out the others.
    """
    plt.figure(figsize=(8, 5))
    for model, rho in densities.items():
        if model == highlight_model:
            plt.plot(pressures, rho, label=model, color='tab:blue', linewidth=2.5)
        else:
            continue
            plt.plot(pressures, rho, label=model, color='grey', alpha=0.3, linewidth=1)
    plt.xlabel("Pressure [Pa]")
    plt.ylabel("Density [kg/m³]")
    plt.title(f"Density vs. Pressure — Highlight: {highlight_model}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Dowson–Higginson Model
#
# 
# $\rho(p) = \rho_0 \left( 1 + C_1 (p - P_0) + C_2 (p - P_0)^2 \right)$
# 
#
# - **rho0**: Reference density at atmospheric pressure [kg/m³], e.g. 850
# - **P0**: Reference pressure [Pa], typically 1e5
# - **C1**, **C2**: Empirical constants [1/Pa, 1/Pa²]
#
# Common in lubrication theory for lightly compressible oils.

# %%
plot_density_models(pressures, densities, highlight_model="Dowson–Higginson")

# %% [markdown]
# ## Power Law Model
#
# 
# $\rho(p) = \rho_0 \left(\frac{p}{P_0}\right)^{1/\alpha}$
# 
#
# - **rho0**: Reference density [kg/m³]
# - **P0**: Reference pressure [Pa]
# - **alpha**: Compressibility exponent (typical: 5 to 7 for liquids)
#
# Used in barotropic flow approximations.

# %%
plot_density_models(pressures, densities, highlight_model="Power Law")

# %% [markdown]
# ## Murnaghan–Tait Model
#
#
# $\rho(p) = \rho_0 \left(\frac{p + K}{P_0 + K}\right)^{1/n}$
# 
#
# - **rho0**: Reference density [kg/m³]
# - **P0**: Reference pressure [Pa]
# - **K**: Bulk modulus [Pa]
# - **n**: Material exponent (~7 for water)
#
# Popular in compressible fluid modeling for liquids.

# %%
plot_density_models(pressures, densities, highlight_model="Murnaghan–Tait")

# %% [markdown]
# ## Bayada–Chupin Model
#
# Captures phase transition via two regimes:
#
# 
# $\rho(p) = \frac{\rho_l}{1 + \frac{\rho_l (p - p_{\mathrm{sat}})}{c_l^2}}, p > p_{\mathrm{sat}}$
#
# $\rho(p) = \frac{\rho_v}{1 - \frac{\rho_v (p_{\mathrm{sat}} - p)}{c_v^2}}, p \le p_{\mathrm{sat}}$
# 
#
# - **rho_l**, **rho_v**: Liquid/vapor densities [kg/m³]
# - **c_l**, **c_v**: Sound speeds in liquid/vapor [m/s]
# - **p_sat**: Saturation pressure (assumed 10^5 Pa here)
#
# Used in cavitation and phase-change simulations.

# %%
plot_density_models(pressures, densities, highlight_model="Bayada–Chupin")