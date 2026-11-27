"""Reproduce Figure 3 from archived runs or by retraining the estimators."""
import argparse
from pathlib import Path

import numpy as np

from simulation.report import load_runs, plot_figure3

ROOT = Path(__file__).resolve().parent
GRID = np.linspace(0.1, 0.9, 9)
SAMPLE_SIZE = 50_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["archive", "train"], default="archive")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--pool", type=Path, help="Synthetic .npz with aligned X, A and Y arrays")
    parser.add_argument("--seed", type=int, default=1, help="Sampling and training seed")
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:INDEX")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--epochs-dr", type=int, default=200)
    parser.add_argument("--epochs-cnf", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--nfr-widths", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--cnf-width", type=int, default=32)
    parser.add_argument("--density-floor", type=float, default=0.01)
    args = parser.parse_args()
    for name in ("reps", "epochs_dr", "epochs_cnf", "batch_size", "cnf_width"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 2 <= args.folds <= SAMPLE_SIZE // 2 or args.lr <= 0 or args.density_floor <= 0 or min(args.nfr_widths) < 1:
        parser.error("Check the number of folds, network widths, learning rate, and density floor.")
    return args


def train(args):
    import torch
    from simulation.data import generate, save_pool, truth
    from simulation.fit import fit, unit_quantiles

    pool = args.pool or args.output / "simulation_pool.npz"
    if not pool.exists():
        print("Generating the synthetic data pool.", flush=True)
        save_pool(pool, *generate(seed=args.data_seed))
    with np.load(pool, allow_pickle=False) as bundle:
        x, a, y = (bundle[key] for key in ("X", "A", "Y"))
    if (x.ndim != 2 or x.shape[1] != 10 or a.shape != (len(x),)
            or y.shape != (len(x), 100) or len(x) < SAMPLE_SIZE):
        raise ValueError("Provide at least 50,000 aligned units with 10 covariates and 100 repeated outcomes.")
    if not all(np.isfinite(arr).all() for arr in (x, a, y)):
        raise ValueError("Simulation arrays must be finite.")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    target = truth(x, a, GRID)
    idx = np.random.RandomState(args.seed).choice(len(a), SAMPLE_SIZE, replace=False)
    sample_x, sample_a = x[idx], a[idx]
    quantiles = unit_quantiles(y[idx], GRID)
    torch.manual_seed(args.seed)
    out = args.output / "trained"
    runs = []
    for r in range(args.reps):
        print(f"Replication {r + 1}/{args.reps} on {args.device}", flush=True)
        estimates = fit(sample_x, sample_a, quantiles, GRID, args, out / f"run_{r:03d}")
        runs.append(np.vstack((target, estimates)))
    runs = np.stack(runs)
    np.savez_compressed(out / "figure3_runs.npz", runs=runs)
    plot_figure3(runs, out)


def main():
    args = parse_args()
    if args.mode == "archive":
        runs = load_runs(ROOT / "reference" / "figure3_runs.npz")
        plot_figure3(runs, args.output / "archive")
    else:
        train(args)


if __name__ == "__main__":
    main()
