"""
plotting.py — Visualisation utilities and reproducibility helpers.

Functions
---------
set_global_seeds      : Fix random seeds for reproducible results.
scatter_hist          : Scatter plot with marginal histograms (broken x-axis for NO/IO).
plot_roc_comparison   : ROC curve with the NN result and the LLH operating point.
plot_2f_spectra       : 2-flavour νμ energy spectra overlay  (Fig 4 left).
plot_dm2_nn_vs_llh    : Joint scatter + marginals comparing NN vs LLH Δm² (Fig 4 right).
plot_spec_comparison  : 4-channel row comparing NO vs IO predicted spectra (Fig 2).
plot_nll_profile      : ΔNLL profile with best-fit and 1σ band      (Extra plot).
plot_nn_bootstrap     : Histogram of NN-bootstrap Δm² with quantiles (Extra plot).
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seeds(seed=42):
    """
    Set random seeds for Python, NumPy, and TensorFlow.

    Note: Some GPU operations remain non-deterministic even with fixed seeds.

    Parameters
    ----------
    seed : int
        Seed value.  Default 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Main comparison plot
# ---------------------------------------------------------------------------

def scatter_hist(llh_dm2, nn_scores, true_dm2=None,
                 io_xlim=(-2.75, -2.30),
                 no_xlim=( 2.25,  2.70),
                 output_path="Fig_classifier_LLH_comp.pdf"):
    """
    Scatter plot of the NN classifier score vs best-fit Δm²₃₂ from the LLH fit.

    The x-axis is split into an IO lobe (negative Δm²₃₂) and a NO lobe
    (positive Δm²₃₂) with separate x-scales to show both regions clearly.
    Marginal histograms are shown on top (Δm²₃₂ distribution) and on the
    right (NN score distribution).

    Parameters
    ----------
    llh_dm2 : array-like
        Best-fit Δm²₃₂ values in units of 10⁻³ eV², one per toy experiment.
        Positive values indicate the LLH prefers NO; negative values indicate IO.
    nn_scores : array-like
        NN classifier scores ∈ [0, 1], one per toy experiment.
        Values > 0.5 indicate the NN prefers NO.
    true_dm2 : float or None
        True Δm²₃₂ (×10⁻³ eV²) used to generate the pseudo-data.
        A dashed vertical line is drawn at this value if provided.
    io_xlim : tuple
        (x_min, x_max) for the IO (negative) panel in units of 10⁻³ eV².
    no_xlim : tuple
        (x_min, x_max) for the NO (positive) panel in units of 10⁻³ eV².
    output_path : str or None
        File path to save the figure.  Set to None to skip saving.
    """
    llh_dm2   = np.asarray(llh_dm2)
    nn_scores = np.asarray(nn_scores)

    plt.rcParams.update({'font.size': 18})
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    outer = fig.add_gridspec(
        2, 3,
        width_ratios=(3, 3, 1.2),
        height_ratios=(1.2, 4.5),
        wspace=0.01, hspace=0.05
    )

    axL = fig.add_subplot(outer[1, 0])
    axR = fig.add_subplot(outer[1, 1], sharey=axL)
    histxL = fig.add_subplot(outer[0, 0], sharex=axL)
    histxR = fig.add_subplot(outer[0, 1], sharex=axR)
    histy  = fig.add_subplot(outer[1, 2], sharey=axL)

    histxL.tick_params(axis="x", labelbottom=False)
    histxR.tick_params(axis="x", labelbottom=False)
    histy.tick_params(axis="y",  labelleft=False)

    maskL = (llh_dm2 >= io_xlim[0]) & (llh_dm2 <= io_xlim[1])
    maskR = (llh_dm2 >= no_xlim[0]) & (llh_dm2 <= no_xlim[1])

    axL.scatter(llh_dm2[maskL], nn_scores[maskL], color='k', alpha=0.5, s=15)
    axR.scatter(llh_dm2[maskR], nn_scores[maskR], color='k', alpha=0.5, s=15)
    axR.tick_params(axis="y", labelleft=False)

    if true_dm2 is not None:
        if io_xlim[0] <= true_dm2 <= io_xlim[1]:
            axL.axvline(true_dm2, color='#D55E00', linestyle=':')  # IO = vermillion
        if no_xlim[0] <= true_dm2 <= no_xlim[1]:
            axR.axvline(true_dm2, color='#0072B2', linestyle=':')  # NO = blue

    axL.set_xlim(io_xlim)
    axR.set_xlim(no_xlim)
    axL.set_ylim(0, 1)
    axL.set_yticks(np.arange(0, 1.1, 0.25))

    def _bins(xlim, bw=0.02):
        return np.arange(xlim[0], xlim[1] + bw, bw)

    histxL.hist(llh_dm2[maskL], bins=_bins(io_xlim), color='#D55E00', alpha=0.7)  # IO LLH
    histxR.hist(llh_dm2[maskR], bins=_bins(no_xlim), color='#0072B2', alpha=0.7)  # NO LLH
    histy.hist(nn_scores, bins=60, color='#D55E00', alpha=0.7, orientation='horizontal')
    histy.grid(True, axis='y', ls='--', color='black', alpha=0.5, linewidth=0.8)

    axL.set_ylabel("NN Classifier Output")
    axR.set_xlabel(
        r"$\Delta m^2_{32}$ best fit from LLH ($\times10^{-3}$ eV$^2$)",
        loc='right'
    )

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    plt.show()


def plot_roc_comparison(nn_scores_NO, nn_scores_IO, llh_dm2_NO, llh_dm2_IO,
                        output_path="Fig_ROC_comparison.pdf"):
    """
    ROC curve for the NN with the LLH operating point overlaid.

    Parameters
    ----------
    nn_scores_NO : array-like
        NN classifier scores for NO-true toy experiments.
    nn_scores_IO : array-like
        NN classifier scores for IO-true toy experiments.
    llh_dm2_NO : array-like
        LLH best-fit Δm²₃₂ (×10⁻³ eV²) for NO-true experiments.
    llh_dm2_IO : array-like
        LLH best-fit Δm²₃₂ (×10⁻³ eV²) for IO-true experiments.
    output_path : str or None
        File path to save the figure.

    Returns
    -------
    dict with keys 'nn_tpr', 'nn_fpr', 'llh_tpr', 'llh_fpr', 'auc'.
    """
    plt.rcParams.update({'font.size': 20})

    # Build ROC curve for the NN
    y_true   = [1] * len(nn_scores_NO) + [0] * len(nn_scores_IO)
    y_scores = list(nn_scores_NO) + list(nn_scores_IO)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc              = auc(fpr, tpr)

    # Threshold that maximises the F1 score
    precision, recall, pr_thr = precision_recall_curve(y_true, y_scores)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    best_thr  = pr_thr[np.argmax(f1[:-1])]
    idx       = np.argmin(np.abs(thresholds - best_thr))
    nn_tpr, nn_fpr = tpr[idx], fpr[idx]

    # Fixed-threshold (=0.5) operating point — the naive symmetric cut
    idx_05         = np.argmin(np.abs(thresholds - 0.5))
    nn_tpr_05      = tpr[idx_05]
    nn_fpr_05      = fpr[idx_05]

    # LLH operating point: positive best-fit Δm²₃₂ → predicts NO
    llh_tpr = np.mean(np.asarray(llh_dm2_NO) > 0)
    llh_fpr = np.mean(np.asarray(llh_dm2_IO) > 0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fpr, tpr, color='#0072B2', lw=2, label=f'NN (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.scatter(nn_fpr, nn_tpr, color='black', s=400, zorder=10,
               label=f'NN best F1 (thr={best_thr:.2f})')
    ax.scatter(nn_fpr_05, nn_tpr_05, color='#D55E00', marker='D', s=200, zorder=10,
               label='NN thr = 0.5')
    ax.scatter(llh_fpr, llh_tpr, color='#009E73', marker='*', s=400, zorder=10,
               label='LLH fit')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('NO False Positive Rate')
    ax.set_ylabel('NO True Positive Rate')
    ax.legend(loc='lower right')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    plt.show()

    return dict(nn_tpr=nn_tpr, nn_fpr=nn_fpr,
                nn_tpr_05=nn_tpr_05, nn_fpr_05=nn_fpr_05,
                llh_tpr=llh_tpr, llh_fpr=llh_fpr,
                auc=roc_auc)


# ---------------------------------------------------------------------------
# 2-flavour benchmark figures
# ---------------------------------------------------------------------------

def plot_2f_spectra(bin_edges, bin_centers, non_osc, osc, fluct,
                    output_path=None):
    """Figure 4 (left) — overlay of non-oscillated, oscillated, and Poisson-
    fluctuated 2-flavour νμ energy spectra.

    Parameters
    ----------
    bin_edges, bin_centers : ndarrays
        Energy-bin edges and centres (GeV).
    non_osc : ndarray
        Non-oscillated event counts per bin.
    osc : ndarray
        Oscillated event counts per bin.
    fluct : ndarray
        Poisson-fluctuated observed counts per bin.
    output_path : str or None
        File path to save the figure.  None → show interactively.
    """
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(bin_centers, bins=bin_edges, weights=non_osc, linewidth=2,
            alpha=0.35, hatch='x', label='Non-oscillated')
    ax.hist(bin_centers, bins=bin_edges, weights=osc, color='#D55E00',
            alpha=0.7, label='Oscillated')

    errors = np.sqrt(np.clip(fluct, 0, None))
    ax.errorbar(bin_centers, fluct, yerr=errors, fmt='o', color='black',
                ecolor='black', elinewidth=1, capsize=2, label='Fake data')

    ax.set_xlabel("Muon Neutrino Energy (GeV)")
    ax.set_ylabel("Entries / 0.2 GeV")
    ax.set_xlim([0.9, 3.5])
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)


def plot_dm2_nn_vs_llh(a_fit, a_nn, true_dm2=2.45, output_path=None):
    """Figure 4 (right) — joint scatter + marginal histograms comparing the Δm²
    recovered by the NN regressor (y-axis) and the standard Poisson LLH fit
    (x-axis), for a series of toy experiments at fixed Δm² = ``true_dm2``.

    Parameters
    ----------
    a_fit, a_nn : sequences
        Best-fit Δm² (×10⁻³ eV²) per pseudo-experiment, from LLH and NN.
    true_dm2 : float
        True Δm² used to generate the pseudo-experiments (×10⁻³ eV²).
    output_path : str or None
        File path to save the figure.  None → show interactively.
    """
    plt.rcParams.update({'font.size': 20})

    fig = plt.figure(figsize=(6, 6))
    gs  = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)
    ax       = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    a_fit_arr = np.asarray(a_fit)
    a_nn_arr  = np.asarray(a_nn)

    ax.scatter(a_fit_arr, a_nn_arr, color='black', alpha=0.5, edgecolor='black')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=true_dm2, color='#0072B2', linestyle=':')  # LLH truth
    ax.axhline(y=true_dm2, color='#D55E00', linestyle=':')  # NN  truth
    ax.axline((true_dm2, true_dm2), slope=1, color='gray', linestyle="--")

    binwidth = 2e-2
    xymax = max(np.max(np.abs(a_fit_arr)), np.max(np.abs(a_nn_arr)))
    xymin = min(np.min(np.abs(a_fit_arr)), np.min(np.abs(a_nn_arr)))
    limU  = (int(xymax / binwidth) + 1) * binwidth
    limD  = (int(xymin / binwidth) - 1) * binwidth
    bins  = np.arange(limD, limU + binwidth, binwidth)

    ax_histx.hist(a_fit_arr, bins=bins, color='#0072B2', alpha=0.7)
    ax_histx.grid(True, alpha=0.3)
    ax_histy.hist(a_nn_arr,  bins=bins, color='#D55E00', alpha=0.7,
                   orientation='horizontal')
    ax_histy.grid(True, alpha=0.3)

    ax.set_xlabel(r"$\Delta m^2$ fitted with LLH ($\times 10^{-3}$ eV$^2$)")
    ax.set_ylabel(r"$\Delta m^2$ fitted with NN  ($\times 10^{-3}$ eV$^2$)")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)


def plot_spec_comparison(no_spec, io_spec, bin_centers, bin_edges,
                          no_label='NO', io_label='IO',
                          stat_label='Stat. uncertainty',
                          channels=(r'$\nu_{\mu}$', r'$\bar{\nu}_{\mu}$',
                                    r'$\nu_{e}$', r'$\bar{\nu}_{e}$'),
                          ylim_max=(80, 30, 50, 15),
                          channel_fontsize=34,
                          output_path=None):
    """Figure 2 — 2×2 grid comparing NO vs IO predicted spectra in 4 channels.

    NO Asimov is drawn as a filled solid-edged histogram (Wong blue,
    #0072B2); IO Asimov as a dashed contour-only histogram (Wong vermillion,
    #D55E00); and the ±1σ Poisson statistical uncertainty on NO as a hashed
    band.  Channel symbol is inside each subplot (top-right); the legend
    appears only in the first subplot.

    Parameters
    ----------
    no_spec, io_spec : sequence of 4 ndarrays
        Asimov event counts per channel for the two orderings.
    bin_centers, bin_edges : sequence of 4 ndarrays
        Bin centres and edges for each channel (from createSpec).
    no_label, io_label, stat_label : str
        Legend labels.
    channels : tuple of 4 strings
        Channel symbols (LaTeX) drawn inside each subplot.
    ylim_max : tuple of 4 floats
        Per-channel y-axis upper limits.
    channel_fontsize : int
        Font size for the in-plot channel symbol.
    output_path : str or None
        File path to save the figure.  None → show interactively.
    """
    with plt.rc_context({'font.size': plt.rcParamsDefault['font.size']}):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()

        for n in range(4):
            edges   = bin_edges[n]
            centers = bin_centers[n]
            widths  = np.diff(edges)
            err     = np.sqrt(np.clip(no_spec[n], 0, None))

            # NO — filled Wong-blue with solid outline
            axs[n].hist(centers, bins=edges, weights=no_spec[n],
                        color='#0072B2', alpha=0.45,
                        edgecolor='#0072B2', linewidth=2.0,
                        linestyle='-', histtype='stepfilled',
                        label=no_label)

            # ±1σ Poisson stat uncertainty on NO — hashed box per bin
            axs[n].bar(centers, 2 * err, bottom=no_spec[n] - err, width=widths,
                       facecolor='none',
                       edgecolor=(0.25, 0.25, 0.25, 0.55),
                       hatch='x', linewidth=0, label=stat_label)

            # IO — Wong-vermillion dashed contour-only histogram
            axs[n].hist(centers, bins=edges, weights=io_spec[n],
                        histtype='step', color='#D55E00',
                        linewidth=3.5, linestyle='--',
                        label=io_label)

            # Channel symbol inside the plot (top-right), enlarged
            axs[n].text(0.95, 0.95, channels[n], transform=axs[n].transAxes,
                        ha='right', va='top', fontsize=channel_fontsize)

            axs[n].set_xlabel('Energy (GeV)', fontsize=26)
            if n in (0, 2):                        # left column only
                axs[n].set_ylabel('Counts', fontsize=26)
            axs[n].tick_params(axis='both', labelsize=22)
            axs[n].grid(True, alpha=0.3)
            axs[n].set_ylim(0, ylim_max[n])

        # Legend only in the first subplot
        axs[0].legend(loc='upper left', fontsize=22)

        fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)


def plot_nll_profile(dm2_vals, nll_vals, dm2_hat,
                      sigma_lo, sigma_hi, true_dm2=None,
                      output_path=None):
    """Diagnostic — ΔNLL profile from the Poisson LLH scan.

    Parameters
    ----------
    dm2_vals : ndarray
        Δm² scan grid (×10⁻³ eV²).
    nll_vals : ndarray
        −2 ln L at each scan point.
    dm2_hat : float
        Best-fit Δm² (×10⁻³ eV²).
    sigma_lo, sigma_hi : float
        Lower / upper 1σ uncertainties from Wilks' theorem (×10⁻³ eV²).
    true_dm2 : float or None
        True Δm² used to generate the data (×10⁻³ eV²).  Drawn as a vertical
        dotted line if given.
    output_path : str or None
    """
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8, 6))

    delta_nll = nll_vals - nll_vals.min()
    ax.plot(dm2_vals, delta_nll, 'b-', lw=2, label=r'$\Delta$NLL')
    ax.axhline(1.0, color='green', linestyle='--', lw=2,
               label=r'$\Delta$NLL = 1 (1$\sigma$)')
    ax.axvspan(dm2_hat - sigma_lo, dm2_hat + sigma_hi,
               color='gray', alpha=0.2,
               label=fr'Best fit = {dm2_hat:.3f}$^{{+{sigma_hi:.3f}}}_{{-{sigma_lo:.3f}}}$')
    ax.axvline(dm2_hat, color='red', linestyle='--', lw=2)
    if true_dm2 is not None:
        ax.axvline(true_dm2, color='black', linestyle=':', lw=2,
                   label=f'True = {true_dm2:.3g}')

    ax.set_xlabel(r'$\Delta m^2$ ($\times 10^{-3}$ eV$^2$)')
    ax.set_ylabel(r'$\Delta$NLL')
    ax.set_ylim(0, max(10.5, delta_nll[delta_nll <= 12].max() if any(delta_nll <= 12) else 12))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)


def plot_nn_bootstrap(nn_boot, true_dm2=None, output_path=None,
                       label='NN bootstrap', color='red'):
    """Diagnostic — histogram of NN-bootstrap Δm² values with quantiles.

    Median and 68 % central interval (15.865 / 84.135 percentiles) are marked.
    """
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8, 6))

    arr    = np.asarray(nn_boot)
    median = np.median(arr)
    lo     = np.percentile(arr, 15.865)
    hi     = np.percentile(arr, 84.135)

    ax.hist(arr, bins=30, color=color, alpha=0.6, label=label)
    ax.axvline(median, color='black', linestyle='-', lw=2,
               label=fr'Median = {median:.3f}$^{{+{hi - median:.3f}}}_{{-{median - lo:.3f}}}$')
    ax.axvline(lo, color='black', linestyle='--', lw=1.5, label='68% interval')
    ax.axvline(hi, color='black', linestyle='--', lw=1.5)
    if true_dm2 is not None:
        ax.axvline(true_dm2, color='blue', linestyle=':', lw=2,
                   label=f'True = {true_dm2:.3g}')

    ax.set_xlabel(r'$\Delta m^2$ ($\times 10^{-3}$ eV$^2$)')
    ax.set_ylabel('Bootstrap count')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=14)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved {output_path}")
    plt.show()
    plt.close(fig)
