"""
fitting.py — Chi-squared / Poisson-likelihood fit of oscillation parameters.

The standard analysis fits one free oscillation parameter (typically Δm²₃₂)
while holding all others fixed, independently for the Normal-Ordering (NO) and
Inverted-Ordering (IO) hypotheses.  The ordering preferred by the data is the
one that yields the smaller minimised −2 ln L.

Functions
---------
poisson_likelihood   : −2 ln L for the Poisson hypothesis test.
osc_fit              : Minimise the likelihood over one free parameter.
plot_best_fit        : Overlay the best-fit spectrum on data.
plot_llh_profile     : 1-D Δχ² profile in Δm²₃₂ for both orderings.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from spectrum import createSpec


# ---------------------------------------------------------------------------
# Likelihood function
# ---------------------------------------------------------------------------

def poisson_likelihood(params, data, fixed_params):
    """
    Compute −2 ln L for the Poisson hypothesis test between data and model.

    The statistic is

        −2 ln L = 2 Σ_i [ μ_i − n_i + n_i ln(n_i / μ_i) ]

    where μ_i is the model prediction and n_i is the observed count in bin i.
    The factor of 2 makes this directly comparable to a χ² distribution.

    Parameters
    ----------
    params : dict
        Dictionary of free-parameter values.  Keys must be a subset of
        {'s23sq', 'delta', 'Dmsq32'}.
    data : list of 4 ndarrays
        Observed event counts in each channel [ν_μ, ν̄_μ, ν_e, ν̄_e].
    fixed_params : list
        [s23sq, delta, Dmsq32] with None in the position of the free parameter.

    Returns
    -------
    float
        The test statistic −2 ln L (≥ 0).
    """
    s23sq, delta, Dmsq32 = fixed_params

    if 's23sq' in params:
        s23sq = params['s23sq']
    if 'delta' in params:
        delta = params['delta']
    if 'Dmsq32' in params:
        Dmsq32 = params['Dmsq32']

    # Penalise unphysical parameter values
    if not (0 < s23sq < 1):
        return 1e5

    model, _, _ = createSpec(s23sq=s23sq, delta=delta, Dmsq32=Dmsq32)

    llh = 0.0
    for n in range(4):
        for i in range(len(data[n])):
            mu = model[n][i]
            ni = data[n][i]
            if mu > 0:
                llh += mu - ni + (ni * np.log(ni / mu) if ni > 0 else 0.0)

    return 2.0 * llh


# ---------------------------------------------------------------------------
# Minimisation
# ---------------------------------------------------------------------------

def osc_fit(data, fixed_params, initial_guess, likelihood_func):
    """
    Minimise the likelihood over a single free oscillation parameter using
    a bounded Brent's-method scalar search.

    Parameters
    ----------
    data : list of 4 ndarrays
        Observed event counts [ν_μ, ν̄_μ, ν_e, ν̄_e].
    fixed_params : list
        [s23sq, delta, Dmsq32].  Set the entry of the free parameter to None.
    initial_guess : dict
        Starting value for the free parameter, e.g. {'Dmsq32': 2.4e-3}.
        The sign of Dmsq32 here selects the NO (+) or IO (−) bracket.
    likelihood_func : callable
        The likelihood function to minimise (e.g. poisson_likelihood).

    Returns
    -------
    fitted_params : ndarray, shape (1,)
        Best-fit value of the free parameter.
    standard_errors : ndarray, shape (1,)
        ``[NaN]`` (Brent's method does not provide an inverse Hessian).
    min_llh : float
        Minimum value of the likelihood at the best fit.
    """
    param_names = ['s23sq', 'delta', 'Dmsq32']
    fit_index   = fixed_params.index(None)
    fit_name    = param_names[fit_index]

    # Physical bounds — generous so the optimum is never at a boundary
    if fit_name == 'Dmsq32':
        # Sign-aware bracket: NO if the initial guess is positive, IO otherwise
        bounds = (1e-3, 5e-3) if initial_guess[fit_name] > 0 else (-5e-3, -1e-3)
    elif fit_name == 's23sq':
        bounds = (0.0, 1.0)
    elif fit_name == 'delta':
        bounds = (-np.pi, np.pi)
    else:
        raise ValueError(f"unknown fit parameter {fit_name!r}")

    result = minimize_scalar(
        lambda x: likelihood_func({fit_name: x}, data, fixed_params),
        bounds=bounds,
        method='bounded',
    )

    fitted_params   = np.array([result.x])
    standard_errors = np.array([np.nan])
    min_llh         = result.fun

    return fitted_params, standard_errors, min_llh


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_best_fit(data, best_fit_pred, bin_centers, bin_edges,
                  label_model='Best fit', label_data='Pseudo-data'):
    """
    Plot data alongside the best-fit model for all four channels.

    Parameters
    ----------
    data : list of 4 ndarrays
        Observed (Poisson-smeared) event counts.
    best_fit_pred : list of 4 ndarrays
        Model prediction at the best-fit parameter values.
    bin_centers, bin_edges : lists
        As returned by createSpec.
    label_model, label_data : str
        Legend labels.
    """
    channel_labels = [
        r'$\nu_{\mu}$', r'$\bar{\nu}_{\mu}$',
        r'$\nu_{e}$',   r'$\bar{\nu}_{e}$'
    ]
    ylims = [80, 30, 50, 15]

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    for n, ax in enumerate(axs.flatten()):
        errors    = np.sqrt(data[n])
        bin_w     = np.diff(bin_edges[n])
        ax.errorbar(bin_centers[n], data[n], yerr=errors, xerr=bin_w / 2,
                    fmt='o', color='black', label=label_data)
        ax.hist(bin_centers[n], bins=bin_edges[n],
                weights=best_fit_pred[n], color='blue', alpha=0.5,
                label=label_model)
        ax.set_xlabel('Energy (GeV)')
        ax.set_ylabel('Events')
        ax.set_title(channel_labels[n])
        ax.legend(fontsize='x-large')
        ax.set_ylim(0, ylims[n])
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_llh_profile(data, likelihood_func, fixed_params,
                     no_range=(2.360e-3, 2.540e-3),
                     io_range=(-2.590e-3, -2.410e-3),
                     npoints=101,
                     output_path=None):
    """
    Draw the 1-D Δχ² profile in Δm²₃₂ for both mass orderings.

    Parameters
    ----------
    data : list of 4 ndarrays
        Observed event counts.
    likelihood_func : callable
        e.g. poisson_likelihood.
    fixed_params : list
        [s23sq, delta, None] where None marks Δm²₃₂ as free.
    no_range : tuple
        (min, max) of the Δm²₃₂ scan for NO (positive values, eV²).
    io_range : tuple
        (min, max) of the Δm²₃₂ scan for IO (negative values, eV²).
    npoints : int
        Number of scan points per ordering.
    output_path : str or None
        File path to save the figure.  Set to None to skip saving.

    Returns
    -------
    float
        Global minimum of the likelihood (useful as a reference).
    """
    plt.rcParams.update({'font.size': 20})

    dm2_NO = np.linspace(*no_range, npoints)
    dm2_IO = np.linspace(*io_range, npoints)

    # IO uses flipped δ_CP
    fixed_IO    = fixed_params.copy()
    fixed_IO[1] = -fixed_IO[1]

    llh_NO = np.array([likelihood_func({'Dmsq32': d}, data, fixed_params) for d in dm2_NO])
    llh_IO = np.array([likelihood_func({'Dmsq32': d}, data, fixed_IO)     for d in dm2_IO])

    llh_global_min = min(llh_NO.min(), llh_IO.min())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    ax1.plot(dm2_IO * 1e3, llh_IO - llh_global_min,
             color='#D55E00', linestyle='--', linewidth=2.5,
             label=r'IO, $\delta_{CP}$ = $3\pi/2$')
    ax2.plot(dm2_NO * 1e3, llh_NO - llh_global_min,
             color='#0072B2', linestyle='-',  linewidth=2,
             label=r'NO, $\delta_{CP}$ = $\pi/2$')

    ax1.set_xlabel(r'IO $\Delta m^2_{32}$ ($\times10^{-3}$ eV$^2$)')
    ax2.set_xlabel(r'NO $\Delta m^2_{32}$ ($\times10^{-3}$ eV$^2$)')
    ax1.set_ylabel(r'$\Delta \chi^2$')
    ax1.set_ylim(0, 10.5)
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    plt.subplots_adjust(wspace=0.001)
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)

    return llh_global_min
