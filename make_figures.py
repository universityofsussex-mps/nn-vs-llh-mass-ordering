"""
make_figures.py — Reproduce all eight paper figures.

The 3-flavour analysis (Figs 1, 2, 3, 5, 6, 7) is driven by run_comparison.py;
the 2-flavour benchmark (Fig 4 right + the `extra` diagnostic plots) is
driven by run_2f_comparison.py.

Figure overview
---------------
  Fig 1  Bi-probability diagram                  — analytic, no files needed
  Fig 2  NO vs IO best-fit spectra (4 channels)  — analytic, no files needed
  Fig 3  Δχ² profile in Δm²₃₂                  — NO Asimov spectrum, instant
  Fig 4  Left:  2-flavour energy spectra         — instant
         Right: NN vs LLH Δm² regression scatter — needs 2f_results.pkl
  Fig 5  Left:  NN classifier score histogram    — needs --model and --data
         Right: Training-data ROC curve          — needs --model and --data
  Fig 6  NN score vs LLH Δm²₃₂ scatter plot    — needs NO_output.pkl + IO_output.pkl
  Fig 7  Full ROC curve + LLH operating point    — needs NO_output.pkl + IO_output.pkl
  extra  Diagnostic plots — not in the paper:
           Extra_nll_profile.pdf   (LLH NLL profile + 1σ band)
           Extra_nn_bootstrap.pdf  (NN-bootstrap Δm² distribution)
                                                  — needs 2f_results.pkl

Combined figures (4, 5) are saved as two PDFs each (FigN_left_*.pdf and
FigN_right_*.pdf) for composition as subfigures in LaTeX.

Quick-start examples
--------------------
# Figures that need no pre-computed files:
python make_figures.py --fig 1,2,3

# Fig 5 (needs model + training data):
python make_figures.py --fig 5 --model MO_model.h5 --data train_sample.pkl

# All figures (requires NO_output.pkl, IO_output.pkl, 2f_results.pkl):
python make_figures.py --fig all --model MO_model.h5 --data train_sample.pkl

# Files in a sub-directory:
python make_figures.py --fig all --model MO_model.h5 --data train_sample.pkl \\
                       --results-dir results/
"""

import argparse
import sys
from pathlib import Path
from pickle import dump, load

import numpy as np
from sklearn.model_selection import train_test_split

from biprobability   import plot_bi_probability_static
from spectrum        import createSpec
from fitting         import poisson_likelihood, plot_llh_profile, osc_fit
from neural_network  import (load_nn_model, plot_classifier_output, plot_roc_curve)
from plotting        import (set_global_seeds, scatter_hist, plot_roc_comparison,
                              plot_2f_spectra, plot_dm2_nn_vs_llh,
                              plot_spec_comparison,
                              plot_nll_profile, plot_nn_bootstrap)
from run_comparison  import (S23SQ, DELTA_CP, DM2_NO, DM2_IO,
                              load_or_generate_training_data, load_or_train_model)
from two_flavor      import (DM2_REF, make_bins, gaussian_flux, oscillated_spectrum)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reproduce paper figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--fig", default="all",
        help='Comma-separated list of figures to produce, e.g. "1,2,5" or "all".'
    )
    p.add_argument(
        "--model", default=None,
        help="Path to trained Keras model (.h5). Required for Fig 5."
    )
    p.add_argument(
        "--data", default=None,
        help="Path to training-data .pkl file. Required for Fig 5."
    )
    p.add_argument(
        "--results-dir", default=".",
        help="Directory containing NO_output.pkl, IO_output.pkl (Figs 6 & 7) "
             "and 2f_results.pkl (Fig 4 right)."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (for reproducibility)."
    )
    p.add_argument(
        "--output-dir", default="figures",
        help="Directory where figure files are saved."
    )
    p.add_argument(
        "--show", action="store_true",
        help="Pop up interactive matplotlib windows for every figure.  "
             "Off by default — figures are written to --output-dir and the "
             "script runs headless."
    )
    return p.parse_args(argv)


def _parse_fig_list(fig_arg):
    """Return a set of figure identifiers (ints or 'extra') from a string."""
    if fig_arg.strip().lower() == "all":
        return {1, 2, 3, 4, 5, 6, 7, 'extra'}
    out = set()
    for s in fig_arg.split(","):
        s = s.strip().lower()
        if s == 'extra':
            out.add('extra')
        else:
            out.add(int(s))
    return out


# ---------------------------------------------------------------------------
# Data-preparation helpers
# ---------------------------------------------------------------------------

# Default companion-data filenames (see the "Companion data" section of the README).
_DEFAULT_MODEL = "MO_model.h5"
_DEFAULT_DATA  = "train_sample.pkl"


def _require_model_and_data(args, which_fig):
    """Resolve --model / --data for figures that need the trained classifier.

    When the user does not pass them explicitly, fall back to the standard
    companion-data filenames (MO_model.h5, train_sample.pkl) if they are present
    in the working directory.  If they still cannot be found, exit with a
    message pointing to how to obtain or generate them.
    """
    if args.model is None and Path(_DEFAULT_MODEL).exists():
        print(f"[INFO] --model not given; using default '{_DEFAULT_MODEL}'")
        args.model = _DEFAULT_MODEL
    if args.data is None and Path(_DEFAULT_DATA).exists():
        print(f"[INFO] --data not given; using default '{_DEFAULT_DATA}'")
        args.data = _DEFAULT_DATA

    if args.model is None or args.data is None:
        sys.exit(
            f"[ERROR] Figure {which_fig} needs the trained NN classifier and its "
            "training data.\n"
            f"  Looked for the defaults '{_DEFAULT_MODEL}' and '{_DEFAULT_DATA}' in "
            "the current directory but did not find them.\n"
            "  Fix this in one of two ways:\n"
            "    1. Download the companion-data archive (see the 'Companion data' "
            "section of the README) and unpack it here, or\n"
            f"    2. Generate it yourself:  python run_comparison.py --model {_DEFAULT_MODEL}\n"
            "  You can also point to files elsewhere with --model <path> --data <path>."
        )


def prepare_nn_predictions(args, out_dir):
    """Load or compute NN scores on the training dataset (cached to nn_predictions.pkl)."""
    cache = out_dir / "nn_predictions.pkl"
    if cache.exists():
        print(f"[INFO] Loading cached NN predictions from {cache}")
        with cache.open("rb") as f:
            return load(f)

    print("[INFO] Computing NN predictions on training data...")
    flat, binary_labels, nbins = load_or_generate_training_data(args, out_dir)
    model = load_or_train_model(args, flat, binary_labels, nbins)

    flat_train, _, y_train, _ = train_test_split(
        flat, binary_labels, test_size=0.1, random_state=args.seed
    )
    y_pred_all   = model.predict(flat[..., np.newaxis],       verbose=0).ravel()
    y_pred_train = model.predict(flat_train[..., np.newaxis], verbose=0).ravel()

    result = dict(y_pred_all=y_pred_all, labels_all=binary_labels,
                  y_pred_train=y_pred_train, labels_train=y_train)
    with cache.open("wb") as f:
        dump(result, f)
    print(f"[INFO] Predictions cached to {cache}")
    return result


def load_toy_results(results_dir):
    """Load NO_output.pkl and IO_output.pkl or exit with a helpful message."""
    no_file = results_dir / "NO_output.pkl"
    io_file = results_dir / "IO_output.pkl"
    missing = [str(f) for f in (no_file, io_file) if not f.exists()]
    if missing:
        sys.exit(
            f"[ERROR] Missing results file(s): {', '.join(missing)}\n"
            f"  Check the file name or generate them by running:\n"
            f"    python run_comparison.py --model <model.h5> --data <train.pkl>"
        )
    print(f"[INFO] Loading toy-experiment results from {results_dir}")
    with no_file.open("rb") as f:
        nn_scores_NO, llh_dm2_NO = load(f)
    with io_file.open("rb") as f:
        nn_scores_IO, llh_dm2_IO = load(f)
    return nn_scores_NO, llh_dm2_NO, nn_scores_IO, llh_dm2_IO


def load_2f_results(results_dir):
    """Load 2f_results.pkl or exit with a helpful message."""
    path = results_dir / "2f_results.pkl"
    if not path.exists():
        sys.exit(
            f"[ERROR] Missing results file: {path}\n"
            f"  Check the file name or generate it by running:\n"
            f"    python run_2f_comparison.py --model <dm2_model.h5>"
        )
    print(f"[INFO] Loading 2-flavour results from {path}")
    with path.open("rb") as f:
        return load(f)


# ---------------------------------------------------------------------------
# Figure-making functions
# ---------------------------------------------------------------------------

def make_fig1(out_dir):
    """Figure 1 — Bi-probability diagram P(ν_μ→ν_e) vs P(ν̄_μ→ν̄_e)."""
    print("\n[Figure 1] Bi-probability diagram")
    plot_bi_probability_static(
        L_km=810.0, E0_GeV=2.0,
        output_path=str(out_dir / "Fig1_biprobability.pdf")
    )


def make_fig2(out_dir):
    """Figure 2 — NO vs IO best-fit spectra in the 4 detection channels.

    Uses the NO Asimov spectrum (same one Fig 3 fits) as the pseudo-data, then
    minimises Δm² independently under the NO (δ_CP = π/2) and IO (δ_CP = 3π/2)
    hypotheses.  The two best-fit spectra are overlaid in 4 panels (1×4 row).
    """
    print("\n[Figure 2] NO vs IO best-fit spectra comparison")
    asimov_NO, bin_centers, bin_edges = createSpec(
        s23sq=S23SQ, delta=DELTA_CP, Dmsq32=2.45e-3,
    )
    fp_NO, _, _ = osc_fit(asimov_NO,
                          fixed_params=[S23SQ,  DELTA_CP, None],
                          initial_guess={'Dmsq32':  2.4e-3},
                          likelihood_func=poisson_likelihood)
    fp_IO, _, _ = osc_fit(asimov_NO,
                          fixed_params=[S23SQ, -DELTA_CP, None],
                          initial_guess={'Dmsq32': -2.5e-3},
                          likelihood_func=poisson_likelihood)
    best_dm2_NO = float(fp_NO[0])
    best_dm2_IO = float(fp_IO[0])
    print(f"  Best-fit NO Δm² = {best_dm2_NO * 1e3:.4f} ×10⁻³ eV²")
    print(f"  Best-fit IO Δm² = {best_dm2_IO * 1e3:.4f} ×10⁻³ eV²")

    no_spec, _, _ = createSpec(s23sq=S23SQ, delta=DELTA_CP,  Dmsq32=best_dm2_NO)
    io_spec, _, _ = createSpec(s23sq=S23SQ, delta=-DELTA_CP, Dmsq32=best_dm2_IO)

    plot_spec_comparison(no_spec, io_spec, bin_centers, bin_edges,
                         no_label=r'NO, $\delta_{CP} = \pi/2$',
                         io_label=r'IO, $\delta_{CP} = 3\pi/2$',
                         output_path=str(out_dir / "Fig2_NO_IO_spectra.pdf"))


def make_fig3(out_dir):
    """Figure 3 — 1-D Δχ² profile in Δm²₃₂ using the NO Asimov spectrum."""
    print("\n[Figure 3] Δχ² profile in Δm²₃₂")
    asimov, _, _ = createSpec(s23sq=S23SQ, delta=DELTA_CP, Dmsq32=2.45e-3)
    plot_llh_profile(
        asimov, poisson_likelihood, fixed_params=[S23SQ, DELTA_CP, None],
        no_range=(2.360e-3, 2.540e-3),
        io_range=(-2.590e-3, -2.410e-3),
        output_path=str(out_dir / "Fig3_llh_profile.pdf")
    )


def make_fig4(results_2f, out_dir, seed):
    """Figure 4 — 2-flavour benchmark.

    Left:  νμ energy spectra (non-osc / osc / Poisson-fluctuated fake data).
    Right: scatter plot of NN-fitted Δm² vs LLH-fitted Δm² over 1000 toys.
    """
    # Left panel
    print("\n[Figure 4 left] 2-flavour νμ energy spectra")
    np.random.seed(seed)
    bin_edges, bin_centers = make_bins()
    non_osc = gaussian_flux(bin_centers)
    osc     = oscillated_spectrum(bin_centers)
    fluct   = np.random.poisson(osc)
    plot_2f_spectra(bin_edges, bin_centers, non_osc, osc, fluct,
                    output_path=str(out_dir / "Fig4_left_2f_spectra.pdf"))

    # Right panel
    print("\n[Figure 4 right] NN vs LLH Δm² regression scatter")
    plot_dm2_nn_vs_llh(results_2f['a_fit'], results_2f['a_nn'],
                       true_dm2=DM2_REF * 1e3,
                       output_path=str(out_dir / "Fig4_right_dm2_nn_vs_llh.pdf"))


def make_fig5(nn_preds, out_dir):
    """Figure 5 — NN training diagnostics.

    Left:  histogram of NN classifier scores split by true ordering.
    Right: ROC curve of the NN on the training dataset.
    """
    print("\n[Figure 5 left] NN classifier score distribution")
    plot_classifier_output(nn_preds['y_pred_all'], nn_preds['labels_all'],
                           output_path=str(out_dir / "Fig5_left_classifier_output.pdf"))

    print("\n[Figure 5 right] Training-data ROC curve")
    plot_roc_curve(nn_preds['y_pred_train'], nn_preds['labels_train'],
                   output_path=str(out_dir / "Fig5_right_roc_curve.pdf"))


def make_fig6(nn_scores_IO, llh_dm2_IO, out_dir):
    """Figure 6 — Scatter plot of NN score vs LLH best-fit Δm²₃₂ (IO true)."""
    print("\n[Figure 6] NN score vs LLH best-fit scatter plot (IO true)")
    scatter_hist(
        llh_dm2_IO, nn_scores_IO,
        true_dm2=DM2_IO * 1e3,
        output_path=str(out_dir / "Fig6_scatter_IO.pdf")
    )


def make_extra(bootstrap, out_dir):
    """Extra diagnostic plots (not in the paper) — single-experiment Δm²
    uncertainty from the 2-flavour bootstrap.

    Produces:
      Extra_nll_profile.pdf   — ΔNLL profile from the Poisson LLH scan + 1σ band
      Extra_nn_bootstrap.pdf  — histogram of NN-bootstrap Δm² + 68 % interval
    """
    print("\n[Extra] LLH ΔNLL profile")
    plot_nll_profile(
        bootstrap['nll_dm2_vals'], bootstrap['nll_vals'],
        dm2_hat=bootstrap['llh_best'],
        sigma_lo=bootstrap['llh_sigma_lo'],
        sigma_hi=bootstrap['llh_sigma_hi'],
        true_dm2=DM2_REF * 1e3,
        output_path=str(out_dir / "Extra_nll_profile.pdf"),
    )

    print("\n[Extra] NN bootstrap Δm² distribution")
    plot_nn_bootstrap(
        bootstrap['a_nn'], true_dm2=DM2_REF * 1e3,
        output_path=str(out_dir / "Extra_nn_bootstrap.pdf"),
    )


def make_fig7(nn_scores_NO, llh_dm2_NO, nn_scores_IO, llh_dm2_IO, out_dir):
    """Figure 7 — Full ROC curve with LLH operating point overlaid."""
    print("\n[Figure 7] ROC curve with LLH comparison")
    rates = plot_roc_comparison(
        nn_scores_NO, nn_scores_IO, llh_dm2_NO, llh_dm2_IO,
        output_path=str(out_dir / "Fig7_ROC_comparison.pdf")
    )
    print(f"  LLH                TPR = {rates['llh_tpr']:.4f}   FPR = {rates['llh_fpr']:.4f}")
    print(f"  NN (best F1)       TPR = {rates['nn_tpr']:.4f}   FPR = {rates['nn_fpr']:.4f}")
    print(f"  NN (thr = 0.5)     TPR = {rates['nn_tpr_05']:.4f}   FPR = {rates['nn_fpr_05']:.4f}")
    print(f"  NN ROC AUC = {rates['auc']:.4f}")
    return rates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args        = parse_args(argv)
    figs        = _parse_fig_list(args.fig)
    out_dir     = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)

    # Suppress interactive pop-ups unless explicitly requested.  Every plot
    # helper still calls plt.show() at the end, but with this in place that
    # call becomes a no-op and the figures are only written to --output-dir.
    if not args.show:
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None

    # ── Figures 1, 2, 3 (analytic — no pre-computed files needed) ────────
    if 1 in figs:
        make_fig1(out_dir)
    if 2 in figs:
        make_fig2(out_dir)
    if 3 in figs:
        make_fig3(out_dir)

    # ── Figure 4 (left is instant; right needs 2f_results.pkl) ───────────
    if 4 in figs:
        results_2f = load_2f_results(results_dir)
        make_fig4(results_2f, out_dir, seed=args.seed)

    # ── Figure 5 (needs --model and --data) ──────────────────────────────
    if 5 in figs:
        _require_model_and_data(args, "5")
        nn_preds = prepare_nn_predictions(args, out_dir)
        make_fig5(nn_preds, out_dir)

    # ── Figures 6 & 7 (load toy-experiment results from disk) ────────────
    if figs & {6, 7}:
        nn_scores_NO, llh_dm2_NO, nn_scores_IO, llh_dm2_IO = \
            load_toy_results(results_dir)
        if 6 in figs:
            make_fig6(nn_scores_IO, llh_dm2_IO, out_dir)
        if 7 in figs:
            make_fig7(nn_scores_NO, llh_dm2_NO, nn_scores_IO, llh_dm2_IO, out_dir)

    # ── Extra diagnostic plots (single-experiment bootstrap) ─────────────
    if 'extra' in figs:
        results_2f = load_2f_results(results_dir)
        boot = results_2f.get('bootstrap')
        if boot is None:
            sys.exit(
                f"[ERROR] {results_dir / '2f_results.pkl'} has no 'bootstrap' "
                "data.\n"
                "  Re-run with: python run_2f_comparison.py --model "
                "<dm2_model.h5> --force"
            )
        make_extra(boot, out_dir)

    print(f"\n[Done] Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
