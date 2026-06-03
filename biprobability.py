"""
biprobability.py — Bi-probability and oscillation-probability plots.

The *bi-probability* diagram (Figure 1 of the paper) shows P(ν_μ→ν_e) on the
x-axis vs P(ν̄_μ→ν̄_e) on the y-axis as the CP phase δ_CP sweeps 0–360°, for
both Normal Ordering (NO) and Inverted Ordering (IO).  At a fixed baseline and
energy, NO and IO trace out two separate ellipses in this plane, giving a
visual picture of how much the two orderings can be distinguished.

Functions
---------
plot_bi_probability_static   : Static bi-probability ellipses (→ Figure 1).
plot_prob_vs_energy          : P(ν_α→ν_β) as a function of neutrino energy.
plot_prob_vs_distance        : P(ν_α→ν_β) vs baseline at fixed L/E.
plot_asymmetry_vs_distance   : CP asymmetry A_CP vs baseline at fixed L/E.
make_bi_probability_gif      : Animated version of Figure 1 (requires Pillow).

Helper functions (used internally)
-----------------------------------
compute_biprobability_curve  : Sweep δ_CP and return (P_μe_ν, P_μe_ν̄) arrays.
compute_prob_vs_energy       : P(i→j) as an array over an energy range.
compute_prob_vs_distance     : P(i→j) as an array over a baseline range.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from oscillations import Probability_Matter_LBL


# ---------------------------------------------------------------------------
# Default NOvA-like physics constants
# ---------------------------------------------------------------------------
_S12SQ    = 0.307
_S13SQ    = 0.0216
_S23SQ    = 0.535
_DMSQ21   = 7.53e-5     # eV²
_DMSQ31_NO = +2.451e-3 + 7.53e-5   # eV²  (NO: Δm²₃₁ = Δm²₃₂ + Δm²₂₁)
_DMSQ31_IO = -2.527e-3 + 7.53e-5   # eV²  (IO)
_RHO      = 2.84        # g/cm³  (NOvA-doc-49397-v1)
_YE       = 0.5
_N_NEWTON = 0

# Bi-probability axis range (percent)
_AXIS_MIN_PCT = 1.5
_AXIS_MAX_PCT = 7.0

# Map flavour indices to LaTeX labels
_FLAVOR = {0: r'e', 1: r'\mu', 2: r'\tau'}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latex_sci(x, digits=3):
    """Return x formatted as LaTeX scientific notation, e.g. 2.45×10⁻³."""
    s = f"{x:.{digits}g}"
    if 'e' in s:
        coeff, exp = s.split('e')
        return rf"{coeff}\times 10^{{{int(exp)}}}"
    return s


def _annotate_cardinal_deltas(ax, x, y, deg_step, color, is_NO):
    """
    Mark the four cardinal CP phases (0°, 90°, 180°, 270°) on a bi-probability
    curve and label them with their values in radians.
    """
    ref_degs   = [0, 90, 180, 270]
    rad_labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$']
    # Text offset (in units of 18 pt) to avoid overlapping the ellipses
    if is_NO:
        xoff = [ 0.2, -1.2, -0.7,  0.2]
        yoff = [ 0.2,  0.3, -0.7, -0.8]
    else:
        xoff = [-0.7, -1.3,  0.2,  0.2]
        yoff = [-0.7,  0.3,  0.2, -0.9]

    for k, deg in enumerate(ref_degs):
        if deg % deg_step == 0:
            idx = int(deg / deg_step)
            if 0 <= idx < len(x):
                ax.scatter(x[idx], y[idx], s=25, color=color, zorder=3)
                ax.annotate(
                    rad_labels[k], (x[idx], y[idx]),
                    textcoords='offset points',
                    xytext=(xoff[k] * 18, yoff[k] * 18),
                    fontsize=18, color=color
                )


def compute_biprobability_curve(L_km, E_GeV, Dmsq31, deg_step=1,
                                s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
                                Dmsq21=_DMSQ21, rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON):
    """
    Compute P(ν_μ→ν_e) and P(ν̄_μ→ν̄_e) as δ_CP sweeps 0°–360°.

    Parameters
    ----------
    L_km : float
        Baseline (km).
    E_GeV : float
        Neutrino energy (GeV; positive value used for both ν and ν̄).
    Dmsq31 : float
        Δm²₃₁ (eV²); positive for NO, negative for IO.
    deg_step : int
        Step size in degrees for the δ_CP sweep.
    s12sq, s13sq, s23sq, Dmsq21, rho, Ye, N_Newton : float / int
        Oscillation and matter parameters.

    Returns
    -------
    pme_nu : ndarray
        P(ν_μ→ν_e) for each δ_CP value (length 361/deg_step).
    pme_anu : ndarray
        P(ν̄_μ→ν̄_e) for each δ_CP value.
    deltas_deg : ndarray
        Corresponding δ_CP values in degrees.
    """
    deltas_deg = np.arange(0, 360 + deg_step, deg_step, dtype=float)
    deltas_rad = np.deg2rad(deltas_deg)

    pme_nu  = np.empty(len(deltas_rad))
    pme_anu = np.empty(len(deltas_rad))

    for k, delta in enumerate(deltas_rad):
        prob_nu  = Probability_Matter_LBL(s12sq, s13sq, s23sq, delta,
                                          Dmsq21, Dmsq31, L_km,  E_GeV,
                                          rho, Ye, N_Newton)
        prob_anu = Probability_Matter_LBL(s12sq, s13sq, s23sq, delta,
                                          Dmsq21, Dmsq31, L_km, -E_GeV,
                                          rho, Ye, N_Newton)
        pme_nu[k]  = prob_nu[1, 0]   # P(ν_μ → ν_e)
        pme_anu[k] = prob_anu[1, 0]  # P(ν̄_μ → ν̄_e)

    return pme_nu, pme_anu, deltas_deg


def compute_prob_vs_energy(i, j, L_km, delta_rad, Dmsq31,
                           E_min=0.2, E_max=5.0, n_points=300,
                           antineutrino=False,
                           s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
                           Dmsq21=_DMSQ21, rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON):
    """
    Compute P(ν_i→ν_j) over an energy range.

    Parameters
    ----------
    i, j : int
        Flavour indices (0=e, 1=μ, 2=τ).
    L_km : float
        Baseline (km).
    delta_rad : float
        CP phase (radians).
    Dmsq31 : float
        Δm²₃₁ (eV²).
    E_min, E_max : float
        Energy range (GeV).
    n_points : int
        Number of energy samples.
    antineutrino : bool
        If True, compute for anti-neutrinos (pass E < 0 to NuFast).

    Returns
    -------
    E_vals : ndarray  (GeV)
    P_vals : ndarray  (probability)
    """
    E_vals = np.linspace(max(E_min, 1e-4), E_max, n_points)
    sign   = -1.0 if antineutrino else +1.0
    P_vals = np.empty(n_points)
    for k, E in enumerate(E_vals):
        probs    = Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_rad,
                                          Dmsq21, Dmsq31, L_km, sign * E,
                                          rho, Ye, N_Newton)
        P_vals[k] = probs[i, j]
    return E_vals, P_vals


def compute_prob_vs_distance(i, j, LE_ratio, Dmsq31, delta_rad=0.0,
                             L_min=0.0, L_max=1000.0, n_points=400,
                             antineutrino=False,
                             s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
                             Dmsq21=_DMSQ21, rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON):
    """
    Compute P(ν_i→ν_j) vs baseline at a fixed L/E ratio.

    Parameters
    ----------
    LE_ratio : float
        Fixed L/E (km/GeV); the energy at each point is E = L / LE_ratio.
    Dmsq31 : float
        Δm²₃₁ (eV²).
    delta_rad : float
        CP phase (radians).
    L_min, L_max : float
        Baseline range (km).

    Returns
    -------
    L_vals : ndarray  (km)
    P_vals : ndarray  (probability)
    """
    L_vals = np.linspace(max(L_min, 1e-6), L_max, n_points)
    E_vals = L_vals / LE_ratio
    sign   = -1.0 if antineutrino else +1.0
    P_vals = np.empty(n_points)
    for k, (L, E) in enumerate(zip(L_vals, E_vals)):
        probs    = Probability_Matter_LBL(s12sq, s13sq, s23sq, delta_rad,
                                          Dmsq21, Dmsq31, L, sign * E,
                                          rho, Ye, N_Newton)
        P_vals[k] = probs[i, j]
    return L_vals, P_vals


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_bi_probability_static(
    L_km=810.0, E0_GeV=2.0, deg_step=1,
    s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
    Dmsq21=_DMSQ21,
    Dmsq31_NO=_DMSQ31_NO, Dmsq31_IO=_DMSQ31_IO,
    rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON,
    output_path="Fig1_biprobability.pdf"
):
    """
    Plot the bi-probability diagram for a NOvA-like experiment (Figure 1).

    Each curve traces out P(ν_μ→ν_e) vs P(ν̄_μ→ν̄_e) as δ_CP sweeps
    0°–360° at fixed L and E.  Normal Ordering (NO) and Inverted Ordering (IO)
    form distinct ellipses, illustrating the experimental sensitivity.

    Parameters
    ----------
    L_km : float
        Baseline (km).  Default 810 km (NOvA).
    E0_GeV : float
        Neutrino energy (GeV).  Default 2.0 GeV.
    deg_step : int
        Angular resolution of the δ_CP sweep (degrees).
    Dmsq31_NO, Dmsq31_IO : float
        Δm²₃₁ for each ordering (eV²).
    output_path : str or None
        File path to save the figure.  Set to None to skip saving.

    Returns
    -------
    dict with keys 'pme_nu_NO', 'pme_anu_NO', 'pme_nu_IO', 'pme_anu_IO',
    'deltas_deg', so the raw curve data is accessible for further analysis.
    """
    plt.rcParams.update({'font.size': 20,
                         'mathtext.fontset': 'stix',
                         'font.family': 'STIXGeneral'})

    pme_nu_NO, pme_anu_NO, deltas_deg = compute_biprobability_curve(
        L_km, E0_GeV, Dmsq31_NO, deg_step,
        s12sq, s13sq, s23sq, Dmsq21, rho, Ye, N_Newton
    )
    pme_nu_IO, pme_anu_IO, _          = compute_biprobability_curve(
        L_km, E0_GeV, Dmsq31_IO, deg_step,
        s12sq, s13sq, s23sq, Dmsq21, rho, Ye, N_Newton
    )

    # Convert to % and clip to the plot window
    def to_pct(p):
        return np.clip(p * 100.0, _AXIS_MIN_PCT, _AXIS_MAX_PCT)

    x_NO, y_NO = to_pct(pme_nu_NO),  to_pct(pme_anu_NO)
    x_IO, y_IO = to_pct(pme_nu_IO),  to_pct(pme_anu_IO)

    fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=130)
    # Wong colour-blind-safe palette: NO = blue (#0072B2), IO = vermillion (#D55E00)
    ax.plot(x_NO, y_NO, color='#0072B2', lw=2,           label='Normal')
    ax.plot(x_IO, y_IO, color='#D55E00', lw=2, ls='--',  label='Inverted')

    _annotate_cardinal_deltas(ax, x_NO, y_NO, deg_step, color='#0072B2', is_NO=True)
    _annotate_cardinal_deltas(ax, x_IO, y_IO, deg_step, color='#D55E00', is_NO=False)

    ax.set_xlabel(r'P($\nu_{\mu} \rightarrow \nu_{e}$) (%)')
    ax.set_ylabel(r'P($\bar{\nu}_{\mu} \rightarrow \bar{\nu}_{e}$) (%)')
    ax.set_xlim(_AXIS_MIN_PCT, _AXIS_MAX_PCT)
    ax.set_ylim(_AXIS_MIN_PCT, _AXIS_MAX_PCT)
    ax.set_title(rf'$L = {L_km:.0f}$ km, $E = {E0_GeV:.1f}$ GeV', fontsize=20)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Fig 1] Saved: {output_path}")
    plt.show()

    return dict(pme_nu_NO=pme_nu_NO, pme_anu_NO=pme_anu_NO,
                pme_nu_IO=pme_nu_IO, pme_anu_IO=pme_anu_IO,
                deltas_deg=deltas_deg)


def plot_prob_vs_energy(
    i=1, j=0, L_km=810.0, cp_degree=90.0,
    E_min=0.2, E_max=5.0, n_points=300,
    show_antineutrino=True, show_IO=True,
    percent=True, y_max_pct=_AXIS_MAX_PCT,
    s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
    Dmsq21=_DMSQ21,
    Dmsq31_NO=_DMSQ31_NO, Dmsq31_IO=_DMSQ31_IO,
    rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON,
    output_path=None
):
    """
    Plot P(ν_i→ν_j) vs neutrino energy for fixed L and δ_CP.

    Parameters
    ----------
    i, j : int
        Flavour indices (0=e, 1=μ, 2=τ).
    L_km : float
        Baseline (km).
    cp_degree : float
        CP phase in degrees.
    show_antineutrino : bool
        Overlay the anti-neutrino curve.
    show_IO : bool
        Overlay IO curves in addition to NO.
    percent : bool
        If True, y-axis is in percent.
    output_path : str or None
        Save path.
    """
    delta_rad = np.deg2rad(cp_degree)
    scale     = 100.0 if percent else 1.0
    ylabel    = (rf"P($\nu_{{{_FLAVOR[i]}}} \rightarrow \nu_{{{_FLAVOR[j]}}}$)"
                 + (" (%)" if percent else ""))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=130)

    # NO neutrino
    E_vals, P = compute_prob_vs_energy(i, j, L_km, delta_rad, Dmsq31_NO,
                                       E_min, E_max, n_points,
                                       s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                       Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                       N_Newton=N_Newton)
    ax.plot(E_vals, P * scale, 'C0', lw=2, label=r'$\nu$ NO')

    if show_antineutrino:
        _, P = compute_prob_vs_energy(i, j, L_km, delta_rad, Dmsq31_NO,
                                      E_min, E_max, n_points, antineutrino=True,
                                      s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                      Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                      N_Newton=N_Newton)
        ax.plot(E_vals, P * scale, 'C0', lw=2, ls='--', label=r'$\bar\nu$ NO')

    if show_IO:
        _, P = compute_prob_vs_energy(i, j, L_km, delta_rad, Dmsq31_IO,
                                      E_min, E_max, n_points,
                                      s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                      Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                      N_Newton=N_Newton)
        ax.plot(E_vals, P * scale, 'C1', lw=2, label=r'$\nu$ IO')

        if show_antineutrino:
            _, P = compute_prob_vs_energy(i, j, L_km, delta_rad, Dmsq31_IO,
                                          E_min, E_max, n_points, antineutrino=True,
                                          s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                          Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                          N_Newton=N_Newton)
            ax.plot(E_vals, P * scale, 'C1', lw=2, ls='--', label=r'$\bar\nu$ IO')

    ax.set_xlabel(r'$E_\nu$ (GeV)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(E_min, E_max)
    if percent and y_max_pct:
        ax.set_ylim(0, y_max_pct)
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend()
    ax.set_title(rf'$L = {L_km:.0f}$ km, $\delta_{{CP}} = {cp_degree:.0f}^\circ$',
                 fontsize=12)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_prob_vs_distance(
    i=1, j=0, LE_ratio=405.0, cp_degree=90.0,
    L_min=0.0, L_max=1000.0, n_points=400,
    show_antineutrino=True, show_IO=True,
    percent=True, y_max_pct=_AXIS_MAX_PCT,
    s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
    Dmsq21=_DMSQ21,
    Dmsq31_NO=_DMSQ31_NO, Dmsq31_IO=_DMSQ31_IO,
    rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON,
    output_path=None
):
    """
    Plot P(ν_i→ν_j) vs baseline at a fixed L/E ratio.

    Parameters match plot_prob_vs_energy; see that docstring for details.
    LE_ratio is the fixed L/E in km/GeV.
    """
    delta_rad = np.deg2rad(cp_degree)
    scale     = 100.0 if percent else 1.0
    ylabel    = (rf"P($\nu_{{{_FLAVOR[i]}}} \rightarrow \nu_{{{_FLAVOR[j]}}}$)"
                 + (" (%)" if percent else ""))

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=130)

    L_vals, P = compute_prob_vs_distance(i, j, LE_ratio, Dmsq31_NO, delta_rad,
                                         L_min, L_max, n_points,
                                         s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                         Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                         N_Newton=N_Newton)
    ax.plot(L_vals, P * scale, 'C0', lw=2,
            label=rf'$\nu$ NO ($\Delta m^2_{{31}}={_latex_sci(Dmsq31_NO)}$ eV²)')

    if show_antineutrino:
        _, P = compute_prob_vs_distance(i, j, LE_ratio, Dmsq31_NO, delta_rad,
                                        L_min, L_max, n_points, antineutrino=True,
                                        s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                        Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                        N_Newton=N_Newton)
        ax.plot(L_vals, P * scale, 'C0', lw=2, ls='--',
                label=r'$\bar\nu$ NO')

    if show_IO:
        _, P = compute_prob_vs_distance(i, j, LE_ratio, Dmsq31_IO, delta_rad,
                                        L_min, L_max, n_points,
                                        s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                        Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                        N_Newton=N_Newton)
        ax.plot(L_vals, P * scale, 'C1', lw=2,
                label=rf'$\nu$ IO ($\Delta m^2_{{31}}={_latex_sci(Dmsq31_IO)}$ eV²)')

        if show_antineutrino:
            _, P = compute_prob_vs_distance(i, j, LE_ratio, Dmsq31_IO, delta_rad,
                                            L_min, L_max, n_points, antineutrino=True,
                                            s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                            Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                            N_Newton=N_Newton)
            ax.plot(L_vals, P * scale, 'C1', lw=2, ls='--', label=r'$\bar\nu$ IO')

    ax.set_xlabel(r'Baseline $L$ (km)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(L_min, L_max)
    if percent and y_max_pct:
        ax.set_ylim(0, y_max_pct)
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(fontsize=10)
    ax.set_title(rf'$L/E = {LE_ratio:.3g}$ km/GeV, $\delta_{{CP}} = {cp_degree:.0f}^\circ$',
                 fontsize=12)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_asymmetry_vs_distance(
    i=1, j=0, LE_ratio=405.0, cp_degree=90.0,
    L_min=0.0, L_max=1000.0, n_points=400,
    show_IO=True,
    s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
    Dmsq21=_DMSQ21,
    Dmsq31_NO=_DMSQ31_NO, Dmsq31_IO=_DMSQ31_IO,
    rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON,
    output_path=None
):
    """
    Plot the CP asymmetry A_CP = (P_ν − P_ν̄)/(P_ν + P_ν̄) vs baseline.

    Parameters match plot_prob_vs_distance; see that docstring for details.
    """
    delta_rad = np.deg2rad(cp_degree)

    def asymmetry(Dmsq31):
        L, P_nu  = compute_prob_vs_distance(i, j, LE_ratio, Dmsq31, delta_rad,
                                             L_min, L_max, n_points,
                                             s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                             Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                             N_Newton=N_Newton)
        _, P_anu = compute_prob_vs_distance(i, j, LE_ratio, Dmsq31, delta_rad,
                                             L_min, L_max, n_points, antineutrino=True,
                                             s12sq=s12sq, s13sq=s13sq, s23sq=s23sq,
                                             Dmsq21=Dmsq21, rho=rho, Ye=Ye,
                                             N_Newton=N_Newton)
        return L, (P_nu - P_anu) / (P_nu + P_anu)

    L_vals, A_NO = asymmetry(Dmsq31_NO)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=130)
    ax.plot(L_vals, A_NO, 'C0', lw=2, label='NO')

    if show_IO:
        _, A_IO = asymmetry(Dmsq31_IO)
        ax.plot(L_vals, A_IO, 'C1', lw=2, label='IO')

    ax.set_xlabel(r'Baseline $L$ (km)')
    ax.set_ylabel(r'$A_{CP}$')
    ax.set_xlim(L_min, L_max)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend()
    ax.set_title(rf'$\delta_{{CP}} = {cp_degree:.0f}^\circ$, $L/E = {LE_ratio:.3g}$ km/GeV',
                 fontsize=12)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Optional: animated GIF
# ---------------------------------------------------------------------------

def make_bi_probability_gif(
    scan_param="Dmsq31",
    start=2.3e-3, end=2.6e-3, frames=50,
    LE_ratio=405.0, L_km=810.0, E_GeV=2.0,
    deg_step=2, fps=10,
    gif_name="bi_prob_scan.gif",
    s12sq=_S12SQ, s13sq=_S13SQ, s23sq=_S23SQ,
    Dmsq21=_DMSQ21, rho=_RHO, Ye=_YE, N_Newton=_N_NEWTON
):
    """
    Animate the bi-probability diagram while scanning one parameter.

    Parameters
    ----------
    scan_param : str
        Which parameter to animate: 'L' (baseline), 's23sq', or 'Dmsq31'.
    start, end : float
        Range of the scanned parameter.
    frames : int
        Number of animation frames.
    LE_ratio : float
        Fixed L/E (km/GeV); used only when scan_param='L'.
    L_km, E_GeV : float
        Fixed values when scan_param is not 'L'.
    fps : int
        Frames per second.
    gif_name : str
        Output file name.  Requires Pillow: pip install Pillow.
    """
    plt.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
    values = np.linspace(start, end, frames)

    fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=100)
    (line_NO,) = ax.plot([], [], 'C0', lw=2, label='Normal')
    (line_IO,) = ax.plot([], [], 'C1', lw=2, ls='--', label='Inverted')
    ax.set_xlabel(r'P($\nu_{\mu} \rightarrow \nu_{e}$) (%)')
    ax.set_ylabel(r'P($\bar{\nu}_{\mu} \rightarrow \bar{\nu}_{e}$) (%)')
    ax.set_xlim(_AXIS_MIN_PCT, _AXIS_MAX_PCT)
    ax.set_ylim(_AXIS_MIN_PCT, _AXIS_MAX_PCT)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend()
    title_obj = ax.set_title("")

    def update(frame_idx):
        v = values[frame_idx]
        if scan_param == "L":
            L, E = v, v / LE_ratio
            dm31_NO, dm31_IO = _DMSQ31_NO, _DMSQ31_IO
            s23 = s23sq
            label = rf"$L={v:.0f}$ km, $E={E:.2f}$ GeV"
        elif scan_param == "s23sq":
            L, E = L_km, E_GeV
            dm31_NO, dm31_IO = _DMSQ31_NO, _DMSQ31_IO
            s23 = v
            label = rf"$\sin^2\theta_{{23}} = {v:.3f}$"
        elif scan_param == "Dmsq31":
            L, E = L_km, E_GeV
            dm31_NO, dm31_IO = v, -v
            s23 = s23sq
            label = rf"$|\Delta m^2_{{31}}| = {_latex_sci(v)}$ eV²"
        else:
            raise ValueError("scan_param must be 'L', 's23sq', or 'Dmsq31'")

        def curve(dm31):
            pnu, panu, _ = compute_biprobability_curve(
                L, E, dm31, deg_step, s12sq, s13sq, s23, Dmsq21, rho, Ye, N_Newton
            )
            return np.clip(pnu * 100, 0, _AXIS_MAX_PCT), np.clip(panu * 100, 0, _AXIS_MAX_PCT)

        xNO, yNO = curve(dm31_NO)
        xIO, yIO = curve(dm31_IO)
        line_NO.set_data(xNO, yNO)
        line_IO.set_data(xIO, yIO)
        title_obj.set_text(label)
        return line_NO, line_IO, title_obj

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(gif_name, writer=PillowWriter(fps=fps))
    print(f"[GIF saved] {gif_name}")
    plt.close(fig)
