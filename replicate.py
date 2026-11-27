"""Rebuild the simulation results or train the models from scratch."""
import argparse
from pathlib import Path

import numpy as np

from simulation.report import SENSITIVITY, SIZES, summarize

ROOT = Path(__file__).resolve().parent
GRID = np.linspace(0.1, 0.9, 9)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["archive", "generate", "train"], default="archive")
    parser.add_argument("--preset", choices=["all", "table1", "smoke"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--pool", type=Path, help="Synthetic .npz with aligned X, A and Y arrays")
    parser.add_argument("--seed", type=int, default=1, help="Sampling and training seed")
    parser.add_argument("--data-seed", type=int, default=1, help="Data generation seed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:INDEX")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--reps", type=int, help="Number of replications (default 100, or 1 in smoke)")
    parser.add_argument("--observations", type=int, default=100, help="Repeated measurements per unit")
    parser.add_argument("--epochs-dr", type=int)
    parser.add_argument("--epochs-cnf", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--nfr-widths", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--cnf-width", type=int, default=32)
    parser.add_argument("--kernel", choices=["epanechnikov", "gaussian", "triweight"],
                        default="epanechnikov")
    parser.add_argument("--bandwidth", choices=["legacy", "pilot", "amise"], default="legacy")
    parser.add_argument("--density-floor", type=float, default=0.01)
    args = parser.parse_args()
    smoke = args.preset == "smoke"
    for name, fallback in (("reps", 1 if smoke else 100), ("epochs_dr", 3 if smoke else 200),
                           ("epochs_cnf", 3 if smoke else 50)):
        if getattr(args, name) is None:
            setattr(args, name, fallback)
    for name in ("reps", "epochs_dr", "epochs_cnf", "batch_size", "cnf_width", "observations"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.folds < 2 or args.lr <= 0 or args.density_floor <= 0 or min(args.nfr_widths) < 1:
        parser.error("Use at least two folds and positive widths, learning rate, and density floor.")
    return args


def make_pool(path, seed):
    from simulation.data import generate, save_pool

    print("Generating the 100,000-unit simulation pool.", flush=True)
    save_pool(path, *generate(seed=seed))


def training_cases(preset):
    if preset == "smoke":
        return {"smoke": (320, 1)}
    if preset == "table1":
        return {"n50000": (50000, 1)}
    cases = {f"n{n}": (n, 1) for n in SIZES}
    cases.update({name: (10000, factor) for name, factor in SENSITIVITY.items()})
    return cases


def train(args, pool):
    import torch
    from simulation.data import DOSES, generate, truth
    from simulation.fit import fit, unit_quantiles

    cases = training_cases(args.preset)
    smoke = args.preset == "smoke"
    if smoke and args.pool is None:
        x, a, y = generate(n=320, observations=max(100, args.observations), seed=args.data_seed)
    else:
        if not pool.exists():
            make_pool(pool, args.data_seed)
        with np.load(pool, allow_pickle=False) as bundle:
            x, a, y = (bundle[key] for key in ("X", "A", "Y"))
    n_min, n_max = min(n for n, _ in cases.values()), max(n for n, _ in cases.values())
    if (x.ndim != 2 or x.shape[1] != 10 or a.shape != (len(x),)
            or y.ndim != 2 or len(y) != len(x) or len(x) < n_max):
        raise ValueError(f"Provide X, A and Y for at least {n_max} aligned units with ten covariates.")
    if y.shape[1] < args.observations or args.folds > n_min // 2:
        raise ValueError("Check the available repeated measurements and number of folds.")
    if not all(np.isfinite(arr).all() for arr in (x, a, y)):
        raise ValueError("Simulation arrays must be finite.")
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ref = truth(x, a, GRID)
    out = args.output / "trained"
    out.mkdir(parents=True, exist_ok=True)
    for name, (n, multiplier) in cases.items():
        idx = np.random.RandomState(args.seed).choice(len(a), n, replace=False)
        cases_x, cases_a = x[idx], a[idx]
        quantiles = unit_quantiles(y[idx, :args.observations], GRID)
        torch.manual_seed(args.seed)
        runs = []
        for r in range(args.reps):
            print(f"{name}: replication {r + 1}/{args.reps} on {args.device}", flush=True)
            estimates = fit(cases_x, cases_a, quantiles, DOSES, GRID, args,
                            out / name / f"run_{r:03d}", multiplier)
            runs.append(np.concatenate([ref[:, None, :], estimates], axis=1))
        np.savez_compressed(out / f"{name}.npz", runs=np.stack(runs))
    if not smoke:
        summarize(out, args.output / "trained_summary", table_only=args.preset == "table1")


def main():
    args = parse_args()
    pool = args.pool or args.output / "simulation_pool.npz"
    if args.mode == "archive":
        summarize(ROOT / "reference", args.output / "archive")
    elif args.mode == "generate":
        make_pool(pool, args.data_seed)
    else:
        train(args, pool)


if __name__ == "__main__":
    main()
