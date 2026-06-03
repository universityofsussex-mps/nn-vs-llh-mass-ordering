"""
run_2f_comparison.py — 2-flavour Δm² regression benchmark.

Trains a dense NN regressor on simulated 2-flavour νμ disappearance histograms
and compares its Δm² predictions to a standard Poisson LLH fit on a series of
fluctuated pseudo-experiments at Δm² = 2.45 × 10⁻³ eV².

Each step is automatically skipped if its output file already exists.
Use --force to redo everything from scratch.

Outputs written to <output-dir>/
  2f_results.pkl   — dict(a_fit, a_nn, dm2_mean, dm2_std, nsamp)

The trained Keras model is saved to whatever path is given by --model.
After this script finishes, run `make_figures.py --fig 4` to produce the
2-flavour figures (Fig 4 left + right).

Usage examples
--------------
# Results already on disk — just print the summary:
python run_2f_comparison.py

# Train a model (if missing) and run 1000 pseudo-experiments:
python run_2f_comparison.py --model dm2_model.h5

# Smaller smoke test:
python run_2f_comparison.py --model dm2_model.h5 --nsamples-train 100000 \\
                            --epochs 30 --nsamp 100

# Rerun from scratch:
python run_2f_comparison.py --model dm2_model.h5 --force
"""

import argparse
import sys
from pathlib import Path
from pickle import dump, load

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tqdm import tqdm

from plotting import set_global_seeds
from two_flavor import (
    DM2_REF, N_BINS, make_bins, generate_histograms,
    create_regressor, train_regressor, fit_dm2, dm2_uncertainty_scan,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="2-flavour NN vs LLH Δm² regression benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=None,
                   help="Path to Keras .h5 file. Required unless 2f_results.pkl "
                        "already exists. Will be trained and saved if missing.")
    p.add_argument("--nsamples-train", type=int, default=1_000_000,
                   help="Number of training histograms.")
    p.add_argument("--epochs", type=int, default=100,
                   help="Maximum training epochs (early-stopping is enabled).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of the training set held out for validation-loss "
                        "monitoring during NN training.  Must be in (0, 1).")
    p.add_argument("--nsamp", type=int, default=1000,
                   help="Number of pseudo-experiments at fixed Δm² = 2.45e-3.")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Number of bootstrap iterations on a single experiment "
                        "(used to estimate the NN Δm² uncertainty).")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed for reproducibility.")
    p.add_argument("--output-dir", default=".",
                   help="Directory where output .pkl files are saved.")
    p.add_argument("--force", action="store_true",
                   help="Ignore cached files and rerun every step from scratch.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args    = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed numpy, Python's random, and TensorFlow together so that the NN
    # weight initialisation, batch shuffling, and Poisson draws are all
    # deterministic across runs.
    set_global_seeds(args.seed)

    results_file = out_dir / "2f_results.pkl"

    if not args.force and results_file.exists():
        # ── Results already on disk — skip everything ──────────────────────
        print(f"[INFO] {results_file} already exists — loading and printing summary.")
        with results_file.open("rb") as f:
            results = load(f)
    else:
        # ── Full pipeline ──────────────────────────────────────────────────
        if args.model is None:
            sys.exit(
                "[ERROR] --model is required when 2f_results.pkl is not present.\n"
                "        Provide --model <path> or use --force to regenerate everything."
            )
        model_path = Path(args.model)

        bin_edges, bin_centers = make_bins()

        # Step 1: generate training set (also yields the dm² scaling parameters)
        print(f"[INFO] Generating {args.nsamples_train:,} training histograms...")
        hist_all, dm2_all = generate_histograms(
            args.nsamples_train, bin_centers, dm2=0.0,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            hist_all, dm2_all, test_size=0.1, random_state=args.seed,
        )
        dm2_mean = float(y_train.mean())
        dm2_std  = float(y_train.std())
        print(f"[INFO] dm² scaling: mean = {dm2_mean:.4e}, std = {dm2_std:.4e}")

        # Step 2: train or load the NN regressor
        if not args.force and model_path.exists():
            print(f"[INFO] Loading model from {model_path}")
            # compile=False: inference only (predict), avoids the harmless
            # Keras "compiled metrics have yet to be built" warning.
            model = load_model(model_path, compile=False)
        else:
            print(f"[INFO] Training regressor "
                  f"(epochs={args.epochs}, batch={args.batch_size}, "
                  f"val_split={args.val_split})")
            Y_train_scaled = (y_train - dm2_mean) / dm2_std
            model = create_regressor(N_BINS)
            train_regressor(model, X_train, Y_train_scaled,
                            epochs=args.epochs, batch_size=args.batch_size,
                            validation_split=args.val_split)
            model.save(str(model_path))
            print(f"[INFO] Model saved to {model_path}")

        # Step 3: pseudo-experiments at fixed Δm² = 2.45e-3
        print(f"\n[INFO] Running {args.nsamp} pseudo-experiments "
              f"at Δm² = {DM2_REF}...")
        a_fit, a_nn = [], []
        for _ in tqdm(range(args.nsamp), desc="NN + LLH on fluctuated hists"):
            hist, _ = generate_histograms(1, bin_centers, dm2=DM2_REF)
            # NN prediction (scaled → physical units)
            scaled  = float(model.predict(hist, verbose=0)[0][0])
            nn_dm2  = scaled * dm2_std + dm2_mean
            # Standard LLH fit (returns NaN if it hit a bound / non-finite)
            llh_dm2 = fit_dm2(hist[0].astype(int), bin_centers)
            if not np.isnan(llh_dm2):
                a_fit.append(llh_dm2 * 1e3)   # ×10⁻³ eV²
                a_nn.append (nn_dm2  * 1e3)

        # Step 4: Bootstrap uncertainty on a single experiment.
        # Generates one pseudo-experiment at Δm² = DM2_REF, runs an LLH
        # uncertainty scan (Wilks' theorem ΔNLL=1), then Poisson-resamples
        # the data N times and applies both NN and LLH to each resample.
        # The spread of the NN predictions gives an empirical 1σ band for
        # the NN that the model itself doesn't provide.
        print(f"\n[INFO] Bootstrap on a single Δm² = {DM2_REF} experiment "
              f"({args.n_bootstrap} resamples)")
        hist_single, _ = generate_histograms(1, bin_centers, dm2=DM2_REF)
        hist_single    = hist_single[0].astype(np.int32)

        llh_best = fit_dm2(hist_single, bin_centers)
        sigma_lo, sigma_hi, nll_dm2_vals, nll_vals = dm2_uncertainty_scan(
            hist_single, bin_centers, llh_best,
        )
        print(f"  LLH best fit = {llh_best * 1e3:.4f} "
              f"(-{sigma_lo * 1e3:.4f} / +{sigma_hi * 1e3:.4f}) ×10⁻³ eV²")

        nn_boot, llh_boot = [], []
        for _ in tqdm(range(args.n_bootstrap), desc="Bootstrap NN + LLH"):
            resample   = np.random.poisson(hist_single).astype(np.int32)
            scaled     = float(model.predict(resample.reshape(1, -1), verbose=0)[0][0])
            nn_boot.append((scaled * dm2_std + dm2_mean) * 1e3)
            llh_dm2 = fit_dm2(resample, bin_centers)
            if not np.isnan(llh_dm2):
                llh_boot.append(llh_dm2 * 1e3)

        results = dict(
            a_fit=a_fit, a_nn=a_nn,
            dm2_mean=dm2_mean, dm2_std=dm2_std,
            nsamp=args.nsamp,
            bootstrap=dict(
                hist=hist_single,
                bin_edges=bin_edges, bin_centers=bin_centers,
                llh_best=llh_best * 1e3,
                llh_sigma_lo=sigma_lo * 1e3,
                llh_sigma_hi=sigma_hi * 1e3,
                nll_dm2_vals=nll_dm2_vals * 1e3,
                nll_vals=nll_vals,
                a_nn=nn_boot, a_llh=llh_boot,
                n_bootstrap=args.n_bootstrap,
            ),
        )
        with results_file.open("wb") as f:
            dump(results, f)
        print(f"[INFO] Results saved to {results_file}")

    # ── Summary ───────────────────────────────────────────────────────────
    a_fit = np.asarray(results['a_fit'])
    a_nn  = np.asarray(results['a_nn'])
    print("\n" + "=" * 60)
    print(f"  2-flavour Δm² regression summary  (N = {len(a_fit)})")
    print("=" * 60)
    print(f"  NN   mean = {a_nn.mean():.4f}   std = {a_nn.std():.4f}   ×10⁻³ eV²")
    print(f"  LLH  mean = {a_fit.mean():.4f}   std = {a_fit.std():.4f}   ×10⁻³ eV²")
    print(f"  Correlation = {np.corrcoef(a_fit, a_nn)[0, 1]:.4f}")
    print("=" * 60)

    boot = results.get('bootstrap')
    if boot is not None:
        nn_b  = np.asarray(boot['a_nn'])
        llh_b = np.asarray(boot['a_llh'])
        nn_med = np.median(nn_b)
        nn_lo  = np.percentile(nn_b, 15.865)
        nn_hi  = np.percentile(nn_b, 84.135)
        ll_med = np.median(llh_b)
        ll_lo  = np.percentile(llh_b, 15.865)
        ll_hi  = np.percentile(llh_b, 84.135)
        print(f"  Single-experiment uncertainty  (N_boot = {len(nn_b)})")
        print(f"  LLH (NLL scan): {boot['llh_best']:.4f} "
              f"(-{boot['llh_sigma_lo']:.4f} / +{boot['llh_sigma_hi']:.4f}) ×10⁻³ eV²")
        print(f"  LLH (bootstrap median + 68% interval): "
              f"{ll_med:.4f} (-{ll_med - ll_lo:.4f} / +{ll_hi - ll_med:.4f}) ×10⁻³ eV²")
        print(f"  NN  (bootstrap median + 68% interval): "
              f"{nn_med:.4f} (-{nn_med - nn_lo:.4f} / +{nn_hi - nn_med:.4f}) ×10⁻³ eV²")
        print("=" * 60)

    print("\n[Done] Run `make_figures.py --fig 4,extra` to produce the "
          "2-flavour paper figure and bootstrap diagnostics.")


if __name__ == "__main__":
    main()
