"""
two_flavor.py — Simplified 2-flavour neutrino-oscillation analysis.

Provides the physics (νμ → νμ survival probability in vacuum, Gaussian flux
model) and ML/fitting tools (regression NN, Poisson LLH minimisation) needed
to benchmark a dense neural-network regressor against the standard
chi²/log-likelihood minimisation for the Δm²₃₂ value.

Used by run_2f_comparison.py and the Fig 4 helpers in make_figures.py.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
L_KM        = 810.0      # baseline (NOvA-like)
SIN22TH     = 0.9951     # effective sin²(2θ) value used in the paper
DM2_REF     = 2.45e-3    # reference Δm² (eV²) for the test pseudo-experiments

# Spectrum binning
ENERGY_MIN  = 1.0        # GeV
ENERGY_MAX  = 3.0        # GeV
N_BINS      = 10

# Gaussian flux × cross-section approximation
FLUX_MU     = 1.95       # GeV
FLUX_SIGMA  = 0.35       # GeV
FLUX_NORM   = 304.749


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def numu_psurv(E, sin22th=SIN22TH, dm2=DM2_REF, L=L_KM):
    """Two-flavour νμ → νμ survival probability in vacuum."""
    return 1.0 - sin22th * np.sin(1.27 * dm2 * L / E) ** 2


def gaussian_flux(E):
    """Gaussian model of the non-oscillated νμ event rate at energy E [GeV]."""
    return (FLUX_NORM / (FLUX_SIGMA * np.sqrt(2 * np.pi))
            * np.exp(-(E - FLUX_MU) ** 2 / (2 * FLUX_SIGMA ** 2)))


def make_bins():
    """Return (bin_edges, bin_centers) for the standard 2-flavour binning."""
    bin_edges   = np.linspace(ENERGY_MIN, ENERGY_MAX, N_BINS + 1)
    bin_width   = (ENERGY_MAX - ENERGY_MIN) / N_BINS
    bin_centers = bin_edges[:-1] + bin_width / 2
    return bin_edges, bin_centers


def oscillated_spectrum(bin_centers, dm2=DM2_REF, sin22th=SIN22TH, L=L_KM):
    """Expected (Asimov) oscillated event-rate spectrum at the bin centres."""
    return gaussian_flux(bin_centers) * numu_psurv(bin_centers,
                                                    sin22th=sin22th, dm2=dm2, L=L)


def generate_histograms(num_samples, bin_centers, dm2=0.0,
                        sin22th=SIN22TH, L=L_KM):
    """Vectorised generator of Poisson-fluctuated pseudo-experiments.

    If dm2 == 0.0 the Δm² of each sample is drawn from Uniform(1e-3, 4e-3);
    otherwise the same Δm² is used for every sample.

    Returns
    -------
    histograms : (num_samples, N_BINS) int32 array
    dm2_values : (num_samples,) float array — the true Δm² of each sample
    """
    if dm2 == 0.0:
        dm2_values = np.random.uniform(1e-3, 4e-3, size=num_samples)
    else:
        dm2_values = np.full(num_samples, float(dm2))

    base  = gaussian_flux(bin_centers)
    phase = 1.27 * dm2_values[:, None] * L / bin_centers[None, :]
    surv  = 1.0 - sin22th * np.sin(phase) ** 2
    mu    = np.clip(base[None, :] * surv, 1e-9, None)
    return np.random.poisson(mu).astype(np.int32), dm2_values


# ---------------------------------------------------------------------------
# Poisson LLH fit
# ---------------------------------------------------------------------------

def _poisson_nll(dm2, counts, bin_centers):
    """Saturated Poisson negative log-likelihood (1 free parameter: Δm²)."""
    mu          = np.clip(oscillated_spectrum(bin_centers, dm2=dm2), 1e-9, None)
    counts_safe = np.clip(counts, 1e-9, None)
    return 2 * np.sum(mu - counts + counts * np.log(counts_safe / mu))


def dm2_uncertainty_scan(counts, bin_centers, dm2_hat,
                          scan_range=(2.25e-3, 2.65e-3), n_points=800):
    """Compute ±1σ uncertainty on Δm² via Wilks' theorem (ΔNLL = 1).

    Returns
    -------
    sigma_lo, sigma_hi : float
        Lower and upper 1σ uncertainties (Δm² units, eV²).
    dm2_vals, nll_vals : ndarray
        The dm² scan grid and the corresponding NLL values, useful for plotting.
    """
    dm2_vals = np.linspace(*scan_range, n_points)
    nll_vals = np.array([_poisson_nll(d, counts, bin_centers) for d in dm2_vals])
    target   = nll_vals.min() + 1.0
    mask     = nll_vals <= target
    if not mask.any():
        return float('nan'), float('nan'), dm2_vals, nll_vals
    dm2_lo   = dm2_vals[mask][0]
    dm2_hi   = dm2_vals[mask][-1]
    return dm2_hat - dm2_lo, dm2_hi - dm2_hat, dm2_vals, nll_vals


def fit_dm2(counts, bin_centers):
    """Minimise the Poisson NLL to recover the best-fit Δm² (eV²).

    Uses a bounded Brent's-method scalar search (``minimize_scalar`` with
    ``method='bounded'``).

    Returns ``np.nan`` if the fit hits a bound or produces a non-finite
    value; otherwise the best-fit Δm².
    """
    lo, hi = 1e-3, 4e-3
    result = minimize_scalar(
        _poisson_nll, args=(counts, bin_centers),
        bounds=(lo, hi), method='bounded',
    )
    dm2 = float(result.x)
    if not np.isfinite(result.fun) or dm2 <= lo or dm2 >= hi:
        return float('nan')
    return dm2


# ---------------------------------------------------------------------------
# Regression NN
# ---------------------------------------------------------------------------

def create_regressor(num_bins=N_BINS):
    """Dense regressor mapping a (num_bins,) histogram to a scaled scalar Δm²."""
    model = models.Sequential(name="dm2_regressor")
    model.add(layers.Input(shape=(num_bins,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_regressor(model, X_train, y_train_scaled,
                    epochs=100, batch_size=256, validation_split=0.1):
    """Fit the regressor with early stopping + LR-plateau scheduling."""
    early_stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6,
                                restore_best_weights=True, verbose=1)
    lr_sched   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                    min_delta=1e-6, min_lr=1e-6, verbose=1)
    return model.fit(X_train, y_train_scaled, epochs=epochs,
                     batch_size=batch_size, validation_split=validation_split,
                     callbacks=[early_stop, lr_sched], verbose=1)
