"""
run_comparison.py — Bulk analysis pipeline.

Compares a dense neural-network (NN) classifier against Poisson chi-squared
minimisation for distinguishing Normal Ordering (NO) and Inverted Ordering (IO)
of neutrino masses, using NOvA-like pseudo-data.

Each step is automatically skipped if its output file already exists.
Use --force to rerun everything from scratch.

Outputs written to <output-dir>/
  train_sample.pkl   — training dataset  (generated if --data is not given)
  NO_output.pkl      — toy-experiment results for the NO hypothesis (Figs 5 & 6)
  IO_output.pkl      — toy-experiment results for the IO hypothesis (Figs 5 & 6)

After this script finishes, run make_figures.py to produce all paper figures.

Usage examples
--------------
# Results already computed — just print the summary (no arguments needed):
python run_comparison.py

# Use a pre-trained model and saved training data (skips training):
python run_comparison.py --model MO_model.h5 --data train_sample.pkl

# Train from scratch (~2M spectra; takes hours):
python run_comparison.py --model MO_model.h5

# Quick smoke-test with fewer toy experiments:
python run_comparison.py --model MO_model.h5 --data train_sample.pkl --nsamp 100

# Rerun everything even if cached files already exist:
python run_comparison.py --model MO_model.h5 --data train_sample.pkl --force
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from pickle import dump, load

import numpy as np

from spectrum import createSpec, createNSpec
from fitting import osc_fit, poisson_likelihood
from neural_network import (
    create_nn_model, train_nn_model, evaluate_nn_model, load_nn_model,
)
from plotting import set_global_seeds


# ---------------------------------------------------------------------------
# Physical parameters (NOvA-like Asimov point)
# ---------------------------------------------------------------------------
S23SQ    = 0.535
DELTA_CP = np.pi / 2.0
DM2_NO   =  2.451e-3   # eV²  (Normal Ordering Asimov)
DM2_IO   = -2.527e-3   # eV²  (Inverted Ordering Asimov)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="NN vs LLH neutrino mass-ordering comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--model", default=None,
        help="Path to Keras model file (.h5 or SavedModel dir). "
             "Required unless NO_output.pkl and IO_output.pkl already exist. "
             "If the file does not exist the model is trained and saved here."
    )
    p.add_argument(
        "--data", default=None,
        help="Path to training-data .pkl file. "
             "If omitted, data is generated and saved to <output-dir>/train_sample.pkl."
    )
    p.add_argument(
        "--nsamples-train", type=int, default=2_000_000,
        help="Number of training spectra to generate when the data file is absent."
    )
    p.add_argument(
        "--epochs", type=int, default=2000,
        help="Maximum training epochs (early stopping may terminate sooner)."
    )
    p.add_argument(
        "--patience", type=int, default=20,
        help="EarlyStopping patience: epochs without val_loss improvement before "
             "training stops.  Best weights are restored automatically."
    )
    p.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of the training set held out for validation-loss "
             "monitoring (used by EarlyStopping).  Must be in (0, 1); pass 0 "
             "to disable validation and early stopping."
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size. Defaults to the full training set (gradient descent)."
    )
    p.add_argument(
        "--nsamp", type=int, default=1000,
        help="Number of toy experiments per mass ordering (NO and IO each)."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility."
    )
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel worker processes used to generate training "
             "spectra.  Defaults to os.cpu_count().  Pass 1 for serial."
    )
    p.add_argument(
        "--output-dir", default=".",
        help="Directory where output .pkl files are saved."
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore all cached files and rerun every step from scratch."
    )
    p.add_argument(
        "--retrain", action="store_true",
        help="Keep the existing training data but force the model to be "
             "retrained from scratch.  Toy-experiment results are also "
             "regenerated so they reflect the new model."
    )
    p.add_argument(
        "--rerun-toys", action="store_true",
        help="Keep the existing training data and trained model; only "
             "regenerate the 2000 toy-experiment results.  Useful for "
             "re-running with a different --nsamp or after a fit-code change."
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_or_generate_training_data(args, out_dir):
    """Return (flat_hists, binary_labels, nbins) ready for the NN.

    Loads from disk if the data file exists; generates and saves it otherwise.
    """
    data_path = out_dir / "train_sample.pkl" if args.data is None else Path(args.data)
    force     = getattr(args, 'train', False)  # internal override used by --force

    if force or not data_path.exists():
        if not force:
            print(f"[INFO] Training data not found at {data_path}. Generating now.")
        print(f"[INFO] Generating {args.nsamples_train:,} training spectra...")
        hist_list, labels, delta_list = createNSpec(
            args.nsamples_train, s23sq=S23SQ, delta=DELTA_CP,
            n_workers=args.n_workers, seed=args.seed,
        )
        with open(data_path, "wb") as f:
            dump((hist_list, labels, delta_list), f)
        print(f"[INFO] Training data saved to {data_path}")
    else:
        print(f"[INFO] Loading training data from {data_path}")
        with open(data_path, "rb") as f:
            hist_list, labels, delta_list = load(f)

    # Convert signed Δm²₃₂ labels to binary (1 = NO, 0 = IO)
    binary_labels = np.where(labels > 0, 1, 0)

    # Flatten the 4-channel histograms into one feature vector per sample
    flat  = np.vstack([np.concatenate(h) for h in hist_list])
    nbins = flat.shape[1]

    return flat, binary_labels, nbins


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------

def load_or_train_model(args, flat_hists, binary_labels, nbins):
    """Return a trained Keras model.

    Loads from disk if the model file exists; trains and saves it otherwise.
    ``--force`` and ``--retrain`` both force retraining (the latter without
    regenerating the training data).
    """
    model_path = Path(args.model)
    force      = (getattr(args, 'train',   False) or
                  getattr(args, 'retrain', False))

    if force or not model_path.exists():
        if not model_path.exists():
            print(f"[INFO] Model not found at {model_path}. Training now.")
        elif getattr(args, 'retrain', False):
            print(f"[INFO] --retrain: forcing model retraining at {model_path}.")

        # Keras splits off args.val_split internally for EarlyStopping
        # monitoring; final generalisation is measured by the toy experiments
        # downstream.
        flat_hists_nn = flat_hists[..., np.newaxis]

        batch = args.batch_size if args.batch_size is not None else len(flat_hists)
        print(f"[INFO] Training model: max {args.epochs} epochs, batch={batch}, "
              f"patience={args.patience}, val_split={args.val_split}")

        model = create_nn_model(nbins)
        model = train_nn_model(
            model, flat_hists_nn, binary_labels,
            epochs=args.epochs, batch_size=batch,
            validation_split=args.val_split, patience=args.patience,
        )
        model.save(str(model_path))
        print(f"[INFO] Model saved to {model_path}")

        loss_tr, acc_tr = evaluate_nn_model(model, flat_hists_nn, binary_labels)
        print(f"  Train loss={loss_tr:.4f}, acc={acc_tr:.4f}")
    else:
        print(f"[INFO] Loading model from {model_path}")
        model = load_nn_model(model_path)

    return model


# ---------------------------------------------------------------------------
# Toy-experiment loop
# ---------------------------------------------------------------------------

def _llh_fit_one(task):
    """Worker for run_ordering_loop — two LLH fits on one toy histogram.

    Module-level so it is picklable for multiprocessing.Pool workers.
    Returns the best-fit Δm²₃₂ in eV² (signed: + for NO, − for IO).
    """
    hist, fixed_NO, ig_NO, fixed_IO, ig_IO = task
    fp_NO, _, llh_NO = osc_fit(hist, fixed_NO, ig_NO, poisson_likelihood)
    fp_IO, _, llh_IO = osc_fit(hist, fixed_IO, ig_IO, poisson_likelihood)
    return fp_NO[0] if llh_NO <= llh_IO else fp_IO[0]


def run_ordering_loop(specs, nn_model, nbins, nsamp,
                      fixed_NO, ig_NO, fixed_IO, ig_IO,
                      n_workers=None):
    """Run *nsamp* Poisson toy experiments for a given set of Asimov spectra.

    Parameters
    ----------
    specs : list of 4 ndarrays
        Asimov (expected) event spectra returned by createSpec.
    nn_model : keras.Model
    nbins : int
    nsamp : int
    fixed_NO, ig_NO : list, dict
        fixed_params and initial_guess for the NO LLH fit.
    fixed_IO, ig_IO : list, dict
        fixed_params and initial_guess for the IO LLH fit.
    n_workers : int or None
        Number of worker processes used for the LLH fits.  ``None`` (default)
        uses ``os.cpu_count()``; pass ``1`` for serial execution.

    Returns
    -------
    nn_scores : list of float
        NN classifier output for each toy experiment (in [0, 1]).
    llh_best_dm2 : list of float
        Best-fit Δm²₃₂ (×10⁻³ eV²) for each toy experiment.
        Positive → LLH prefers NO; negative → LLH prefers IO.
    """
    # Generate all nsamp Poisson histograms up-front.  Both a per-toy list
    # of 4-channel arrays (for the LLH fits) and a flat (nsamp, nbins, 1)
    # tensor (for the batched NN forward pass) are kept.
    all_hists_4ch = [[np.random.poisson(s) for s in specs] for _ in range(nsamp)]
    x_batch = np.stack(
        [np.concatenate(h) for h in all_hists_4ch]
    ).reshape(nsamp, nbins, 1)

    # Single batched forward pass through the NN.
    nn_scores = nn_model(x_batch, training=False).numpy().ravel().tolist()

    # Per-toy LLH fits, parallelised across n_workers processes.  imap
    # preserves task ordering so llh_best_dm2[i] still matches nn_scores[i].
    n_workers = n_workers or os.cpu_count()
    tasks = [(h, fixed_NO, ig_NO, fixed_IO, ig_IO) for h in all_hists_4ch]

    def _consume(result_iter):
        out = []
        for samp, best_dm2 in enumerate(result_iter):
            out.append(best_dm2 * 1e3)   # convert to ×10⁻³ eV²
            if (samp + 1) % 50 == 0 or samp == 0:
                print(f"  [{samp+1:4d}/{nsamp}] NN={nn_scores[samp]:.3f}  "
                      f"Δm²₃₂={best_dm2*1e3:.4f}")
        return out

    if n_workers > 1:
        print(f"  [INFO] Running LLH fits across {n_workers} workers")
        with mp.Pool(n_workers) as pool:
            llh_best_dm2 = _consume(
                pool.imap(_llh_fit_one, tasks, chunksize=16)
            )
    else:
        llh_best_dm2 = _consume(_llh_fit_one(t) for t in tasks)

    return nn_scores, llh_best_dm2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args    = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)

    no_file = out_dir / "NO_output.pkl"
    io_file = out_dir / "IO_output.pkl"

    if (not args.force and not args.retrain and not args.rerun_toys
            and no_file.exists() and io_file.exists()):
        # ── Results already on disk — skip all computation ─────────────────
        print("[INFO] Found existing results — skipping model loading and toy experiments.")
        with no_file.open("rb") as f:
            nn_scores_NO, llh_dm2_NO = load(f)
        with io_file.open("rb") as f:
            nn_scores_IO, llh_dm2_IO = load(f)
    else:
        # ── Full pipeline ─────────────────────────────────────────────────
        if args.model is None:
            sys.exit(
                "[ERROR] --model is required when results files are not present.\n"
                "        Provide --model <path> or use --force to regenerate everything."
            )
        if args.force:
            args.train = True   # tells load_or_* helpers to regenerate

        # Step 1: Training data and model
        flat_hists, binary_labels, nbins = load_or_generate_training_data(args, out_dir)
        nn_model = load_or_train_model(args, flat_hists, binary_labels, nbins)

        # Step 2: Toy experiments
        fixed_NO = [S23SQ,  DELTA_CP, None]
        ig_NO    = {'Dmsq32':  2.4e-3}
        fixed_IO = [S23SQ, -DELTA_CP, None]
        ig_IO    = {'Dmsq32': -2.5e-3}

        print(f"\n[INFO] Running {args.nsamp} NO-true toy experiments...")
        specs_NO, _, _ = createSpec(s23sq=S23SQ, delta=DELTA_CP,  Dmsq32=DM2_NO)
        nn_scores_NO, llh_dm2_NO = run_ordering_loop(
            specs_NO, nn_model, nbins, args.nsamp, fixed_NO, ig_NO, fixed_IO, ig_IO,
            n_workers=args.n_workers,
        )

        print(f"\n[INFO] Running {args.nsamp} IO-true toy experiments...")
        specs_IO, _, _ = createSpec(s23sq=S23SQ, delta=-DELTA_CP, Dmsq32=DM2_IO)
        nn_scores_IO, llh_dm2_IO = run_ordering_loop(
            specs_IO, nn_model, nbins, args.nsamp, fixed_NO, ig_NO, fixed_IO, ig_IO,
            n_workers=args.n_workers,
        )

        with no_file.open("wb") as f:
            dump((nn_scores_NO, llh_dm2_NO), f)
        with io_file.open("wb") as f:
            dump((nn_scores_IO, llh_dm2_IO), f)
        print(f"\n[INFO] Results saved to {out_dir}")

    # ── Step 3: Print summary (no plotting) ───────────────────────────────
    nn_thr  = 0.5
    llh_tpr = np.mean(np.array(llh_dm2_NO) > 0)
    llh_fpr = np.mean(np.array(llh_dm2_IO) > 0)
    nn_tpr  = np.mean(np.array(nn_scores_NO) > nn_thr)
    nn_fpr  = np.mean(np.array(nn_scores_IO) > nn_thr)

    print("\n" + "=" * 50)
    print("  Mass-ordering discrimination summary")
    print("=" * 50)
    print(f"  LLH               TPR = {llh_tpr:.4f}   FPR = {llh_fpr:.4f}")
    print(f"  NN (thr = 0.5)    TPR = {nn_tpr:.4f}   FPR = {nn_fpr:.4f}")
    print("=" * 50)
    print("\n[Done] Run make_figures.py to produce all paper figures.")


if __name__ == "__main__":
    main()
