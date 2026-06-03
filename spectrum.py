"""
spectrum.py — NOvA-like energy spectrum generation.

Generates the expected event spectra in four detection channels
(ν_μ, ν̄_μ, ν_e, ν̄_e) for a NOvA-geometry long-baseline experiment,
and draws training samples for the neural network classifier.

The beam parameters (baseline, matter density, normalisations) mirror
the NOvA experiment at Fermilab (L = 810 km).
"""

import multiprocessing as mp
import os

import numpy as np
from scipy.integrate import simpson
from tqdm import tqdm

from oscillations import Probability_Matter_LBL


# Composite Simpson's rule needs an odd number of evaluation points.
# 17 nodes = 16 intervals = 4th-order accurate.  For the ~0.2 GeV-wide bins
# used here, this matches scipy.integrate.quad to ~10⁻⁶ relative — well below
# the Poisson statistical floor on each bin.
_N_SIMPSON_NODES = 17

# Half-width of the Δm²₃₂ window sampled during NN training, in eV².
# Set to ≈5% of the NO central value (2.45×10⁻³ eV²), so the classifier sees
# spectra spanning a generous band around each ordering's central value.
_DM2_TRAIN_HALFWIDTH = 0.1215e-3   # ≈ ±5% of 2.45×10⁻³ eV²


# ---------------------------------------------------------------------------
# Flux model
# ---------------------------------------------------------------------------

def gauss_flux(x, norm, mu=1.95, sigma=0.35):
    """
    Gaussian approximation to the accelerator neutrino energy spectrum.

    Parameters
    ----------
    x : float
        Neutrino energy (GeV).
    norm : float
        Integral normalisation (event count when integrated over all E).
    mu : float
        Peak energy (GeV).  Default 1.95 GeV (NOvA-like).
    sigma : float
        Width (GeV).  Default 0.35 GeV.

    Returns
    -------
    float
        Flux value at energy x.
    """
    return (norm / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def _oscillated_integrand(x, row, col, numu, s12sq, s13sq, s23sq, delta,
                           Dmsq21, Dmsq31, L, rho, Ye, N_Newton, norm):
    """
    Integrand for one energy bin: oscillation probability × flux.

    Parameters
    ----------
    x : float
        Energy (GeV).
    row, col : int
        Flavour indices (0=e, 1=μ, 2=τ) for P(ν_row → ν_col).
    numu : int
        +1 for neutrinos, -1 for anti-neutrinos (sign convention for NuFast).
    norm : float
        Gaussian flux normalisation.
    All other parameters are passed directly to Probability_Matter_LBL.

    Returns
    -------
    float
        Oscillated flux at energy x.
    """
    probs = Probability_Matter_LBL(
        s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31,
        L, numu * x, rho, Ye, N_Newton
    )
    return probs[row][col] * gauss_flux(x, norm)


# ---------------------------------------------------------------------------
# Spectrum generation
# ---------------------------------------------------------------------------

def createSpec(s23sq=0.55, delta=0.0, Dmsq32=2.45e-3):
    """
    Compute the expected oscillated event spectra for four detection channels.

    The four channels are: ν_μ disappearance, ν̄_μ disappearance,
    ν_e appearance, and ν̄_e appearance.  Bin edges mimic the NOvA analysis.

    Parameters
    ----------
    s23sq : float
        sin²(θ₂₃).  Default 0.55.
    delta : float
        CP-violating phase δ_CP (radians).  Default 0.
    Dmsq32 : float
        Δm²₃₂ = m₃² − m₂²  (eV²).
        Use a positive value for Normal Ordering (NO) and a negative
        value for Inverted Ordering (IO).  Default 2.45 × 10⁻³ eV².

    Returns
    -------
    specs : list of 4 ndarrays
        Bin contents [ν_μ, ν̄_μ, ν_e, ν̄_e].
    bin_centers : list of 4 ndarrays
        Bin centre energies (GeV) for each channel.
    bin_edges : list of 4 ndarrays
        Bin edge energies (GeV) for each channel.
    """
    # Experiment geometry
    L        = 810       # km (NOvA baseline)
    rho      = 2.84      # g/cm³ (NOvA-doc-49397-v1)
    Ye       = 0.5
    N_Newton = 0

    # Global oscillation parameters (PDG 2023 central values)
    s12sq  = 0.307
    s13sq  = 0.0216
    Dmsq21 = 7.53e-5    # eV²
    Dmsq31 = Dmsq32 + Dmsq21   # eV²  (Δm²₃₁ = Δm²₃₂ + Δm²₂₁)

    # Beam normalisations (NOvA-like event counts)
    numuN  = 2171 * 218.8 / 225.9
    AnumuN = numuN / 5.0

    # Energy binning for each channel [ν_μ, ν̄_μ, ν_e, ν̄_e]
    xmin  = np.array([0.80, 0.75, 1.00, 1.00])
    xmax  = np.array([3.00, 3.25, 3.00, 3.00])
    nbins = np.array([11,   5,    4,    2   ])

    bin_edges   = []
    bin_centers = []
    bin_width   = (xmax - xmin) / nbins
    for i in range(4):
        edges = np.linspace(xmin[i], xmax[i], nbins[i] + 1)
        bin_edges.append(edges)
        bin_centers.append(edges[:-1] + bin_width[i] / 2)

    # Integration settings for each channel:
    #   (flavour_row, flavour_col, numu_sign, normalisation)
    channel_cfg = [
        (1, 1,  1, numuN ),   # ν_μ survival
        (1, 1, -1, AnumuN),   # ν̄_μ survival
        (1, 0,  1, numuN ),   # ν_e appearance
        (1, 0, -1, AnumuN),   # ν̄_e appearance
    ]

    # Group physics args once so they aren't re-bundled per bin.
    phys_args = (s12sq, s13sq, s23sq, delta, Dmsq21, Dmsq31,
                 L, rho, Ye, N_Newton)

    specs = []
    for n, (row, col, sign, norm) in enumerate(channel_cfg):
        spec = np.empty(nbins[n])
        for i in range(nbins[n]):
            x_nodes = np.linspace(bin_edges[n][i],
                                  bin_edges[n][i + 1],
                                  _N_SIMPSON_NODES)
            y_nodes = np.fromiter(
                (_oscillated_integrand(x, row, col, sign, *phys_args, norm)
                 for x in x_nodes),
                dtype=float, count=_N_SIMPSON_NODES,
            )
            spec[i] = simpson(y_nodes, x=x_nodes)
        specs.append(spec)

    return specs, bin_centers, bin_edges


# ---------------------------------------------------------------------------
# Training-sample generation
# ---------------------------------------------------------------------------

def _generate_one_spectrum(task):
    """Worker for createNSpec — generates one labelled training sample.

    Module-level so it is picklable for multiprocessing.Pool workers.
    Each task gets its own ``np.random.default_rng(master_seed + idx)``,
    so every sample is deterministic and no two workers can ever draw the
    same Poisson realisation.
    """
    idx, master_seed, dm2, s23sq, delta = task
    rng = np.random.default_rng(master_seed + idx)

    if dm2 == 0:
        # Uniform window ±_DM2_TRAIN_HALFWIDTH (≈±1.5%) around the NO central value
        dm2_current = (rng.random() * 2 - 1) * _DM2_TRAIN_HALFWIDTH + 2.45e-3
    else:
        dm2_current = dm2

    # Alternate between NO (+1) and IO (−1)
    MO = -1 if idx % 2 else 1
    dm2_signed = dm2_current
    if MO == -1:
        # Shift to the IO central value (accounts for the different best-fit)
        dm2_signed = dm2_current - (2.45e-3 - 2.53e-3)

    specs, _, _ = createSpec(
        s23sq=s23sq,
        delta=delta * MO,
        Dmsq32=dm2_signed * MO,
    )
    specs = [rng.poisson(s) for s in specs]
    return specs, dm2_signed * MO, delta * MO


def createNSpec(nsamples, dm2=0, s23sq=0.45, delta=0.0,
                 n_workers=None, seed=42):
    """
    Generate a labelled training dataset of Poisson-fluctuated spectra.

    For each sample the mass ordering (NO/IO) is alternated and a random
    Δm²₃₂ value is drawn from a uniform window around the known best-fit.
    The corresponding CP phase is also flipped when switching to IO.

    The 2 M-sample default is generated in parallel using a multiprocessing
    pool — each sample is fully independent, so the speed-up is ~linear in
    the number of cores.

    Parameters
    ----------
    nsamples : int
        Total number of spectra to generate.
    dm2 : float, optional
        If zero (default) Δm²₃₂ is drawn randomly for each event.
        If non-zero the given value is used for every event (useful for
        generating an Asimov dataset with a fixed mass ordering).
    s23sq : float
        sin²(θ₂₃).  Default 0.45.
    delta : float
        Magnitude of the CP phase δ_CP (radians); the sign is flipped for IO.
        Default 0.
    n_workers : int or None
        Number of worker processes.  ``None`` (default) uses ``os.cpu_count()``;
        pass ``1`` for serial execution (useful for debugging).
    seed : int
        Master random seed.  Sample ``idx`` uses ``default_rng(seed + idx)``,
        so the dataset is reproducible across runs and across worker counts.

    Returns
    -------
    histograms : list of lists
        Each element is a list of four numpy arrays (one per channel)
        containing the Poisson-smeared event counts.
    labels : ndarray, shape (nsamples,)
        True Δm²₃₂ value with sign (+ for NO, − for IO) in eV².
    deltaCP_values : ndarray, shape (nsamples,)
        True δ_CP value (with sign) used for each sample.
    """
    n_workers = n_workers or os.cpu_count()
    tasks     = [(idx, seed, dm2, s23sq, delta) for idx in range(nsamples)]

    histograms     = []
    labels         = []
    deltaCP_values = []

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            iterator = pool.imap(_generate_one_spectrum, tasks, chunksize=64)
            desc = f"Generating spectra (n_workers={n_workers})"
            for specs, label, dcp in tqdm(iterator, total=nsamples, desc=desc):
                histograms.append(specs)
                labels.append(label)
                deltaCP_values.append(dcp)
    else:
        for task in tqdm(tasks, desc="Generating spectra"):
            specs, label, dcp = _generate_one_spectrum(task)
            histograms.append(specs)
            labels.append(label)
            deltaCP_values.append(dcp)

    return histograms, np.array(labels), np.array(deltaCP_values)
