"""One command for the simulation figures and synthetic application."""
import argparse
import json
from pathlib import Path

import numpy as np

from report import SENSITIVITY, SIZES, plt, summarize

HERE = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["quick", "archive", "generate", "train", "example", "fit"],
                        default="quick")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--pool", type=Path, help="Synthetic X/A/Y NPZ; created if absent in train mode")
    parser.add_argument("--seed", type=int, default=1, help="Sampling, training and diagnostic seed")
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:INDEX")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs-dr", type=int)
    parser.add_argument("--epochs-cnf", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--nfr-widths", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--cnf-width", type=int, default=32)
    parser.add_argument("--kernel", choices=["epanechnikov", "gaussian", "triweight"], default="epanechnikov")
    parser.add_argument("--bandwidth", choices=["pilot", "amise", "legacy"])
    parser.add_argument("--density-floor", type=float, default=0.01)
    parser.add_argument("--quantiles", type=float, nargs="+")
    parser.add_argument("--doses", type=float, nargs="+")
    parser.add_argument("--units", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--preset", choices=["all", "table1"], default="all")
    parser.add_argument("--within-sensitivity", action="store_true", help="Refit after halving per-unit counts")
    parser.add_argument("--bootstrap-draws", type=int, default=499)
    args = parser.parse_args()
    for name in ("reps", "batch_size", "bootstrap_draws", "cnf_width"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_','-')} must be positive")
    if args.folds < 2 or any(w < 1 for w in args.nfr_widths) or not args.density_floor > 0:
        parser.error("Use at least two folds and positive network widths and density floor")
    if args.mode == "fit" and any(v is None for v in (args.units, args.outcomes, args.doses)):
        parser.error("fit requires --units, --outcomes and --doses")
    return args


def configure(args):
    import torch
    full = args.mode == "train"
    args.device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.epochs_dr = (200 if full else 12) if args.epochs_dr is None else args.epochs_dr
    args.epochs_cnf = (50 if full else 4) if args.epochs_cnf is None else args.epochs_cnf
    if min(args.epochs_dr, args.epochs_cnf) < 1:
        raise ValueError("Epoch caps must be positive.")
    args.bandwidth = args.bandwidth or ("legacy" if full else "pilot")
    print(f"Using {args.device}, NFR {args.epochs_dr} epochs, CNF up to {args.epochs_cnf} epochs.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def synthetic_application(args, out):
    import pandas as pd
    from simulation_data import generate
    rng = np.random.default_rng(args.data_seed)
    x, dose, raw = generate(320, 64, args.data_seed)
    credit = 10000 + 5000*dose
    units = rng.permutation(len(x))
    counts = rng.integers(24, 65, size=len(x))
    repeated = [np.maximum(0, 12+110*raw[i, :counts[i]]) for i in units]
    out.mkdir(parents=True, exist_ok=True)
    units_df = pd.DataFrame(x[units], columns=[f"x{k}" for k in range(x.shape[1])])
    units_df.insert(0, "dose", credit[units])
    units_df.insert(0, "unit_id", units)
    units_df.to_csv(out / "units.csv", index=False)
    pd.DataFrame({"unit_id": np.repeat(units, counts[units]),
                  "outcome": np.concatenate(repeated)}).to_csv(out / "outcomes.csv", index=False)
    return x[units], credit[units], repeated


def plot_application(df, path, example):
    levels = df["quantile"]
    chosen = levels.unique()
    cols = 3
    rows = (len(chosen) + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 2.8*rows),
                             layout="constrained", squeeze=False, sharex=True)
    for ax, q in zip(axes.flat, chosen):
        p = df.loc[df["quantile"] == q].sort_values("dose")
        a, est, lo, hi = (p[c].to_numpy() for c in ["dose", "estimate", "point_lower", "point_upper"])
        ax.plot(a, est, ".-")
        ax.fill_between(a, lo, hi, alpha=.2)
        ax.set_title(f"Quantile {q:g}")
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    fig.supxlabel("Assigned dose")
    fig.supylabel("Estimated outcome")
    if example:
        fig.suptitle("Synthetic example of the Figure 6 workflow")
    fig.savefig(path)
    plt.close(fig)


def application(args, grid, out, example):
    import pandas as pd
    import torch
    from fit import bands, fit, read_long
    if example:
        data = synthetic_application(args, out / "data")
        default_doses = np.arange(6000, 14500, 1000)
    else:
        data = read_long(args.units, args.outcomes)
        default_doses = args.doses
    doses = np.array(args.doses if args.doses is not None else default_doses)
    scores = fit(*data, doses, grid, args, out)
    df = bands(scores, doses, grid, out, args.seed, args.bootstrap_draws)
    plot_application(df, out / ("figure6_synthetic.pdf" if example else "dose_response.pdf"), example)
    if example or args.within_sensitivity:
        x, a, ys = data
        rng = np.random.default_rng(args.seed)
        half = [rng.choice(y, max(1, len(y)//2), replace=False) for y in ys]
        reduced = out / "half_observations"
        torch.manual_seed(args.seed)
        second = fit(x, a, half, doses, grid, args, reduced)
        reduced_mean = second[:, 2].mean(axis=1).ravel()
        pd.DataFrame({"dose": df["dose"], "quantile": df["quantile"],
                      "estimate_full": df["estimate"], "estimate_half": reduced_mean,
                      "change": reduced_mean-df["estimate"].to_numpy()}).to_csv(
            out / "within_unit_sensitivity.csv", index=False)
    metadata = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
    metadata.update(quantiles=grid.tolist(), doses=doses.tolist())
    (out / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_pool(path, seed):
    from simulation_data import generate, save_pool
    print("Generating the 100,000-unit synthetic pool.")
    save_pool(path, *generate(seed=seed))
    print(f"Saved {path}")


def train_suite(args, grid):
    import torch
    from fit import fit, unit_quantiles
    from simulation_data import DOSES, truth
    pool = args.pool or args.output / "simulation_pool.npz"
    if not pool.is_file():
        make_pool(pool, args.data_seed)
    with np.load(pool, allow_pickle=False) as bundle:
        x, a, y = (bundle[k] for k in ("X", "A", "Y"))
    if len(a) < (50_000 if args.preset == "table1" else max(SIZES)):
        raise ValueError("The selected simulation suite requires a larger synthetic pool.")
    if not np.allclose(grid, np.linspace(.1,.9,9)) or args.doses is not None:
        raise ValueError("The paper simulation uses the fixed nine-quantile and three-dose grids.")
    ref = truth(x, a, grid)
    # A given sample size uses the same seed and draw order across bandwidths.
    settings = {"n50000": (50000, 1)}
    if args.preset == "all":
        settings.update({f"n{size}": (size, 1) for size in SIZES})
        settings.update({key: (10000, scale) for key, scale in SENSITIVITY.items()})
    root = args.output / "trained"
    root.mkdir(parents=True, exist_ok=True)
    for label, (n, scale) in settings.items():
        idx = np.random.RandomState(args.seed).choice(len(a), n, replace=False)
        torch.manual_seed(args.seed)
        yy = y[idx]
        q = unit_quantiles(yy, grid, legacy=True)
        runs = []
        for r in range(args.reps):
            print(f"{label}: replication {r+1}/{args.reps}", flush=True)
            scores = fit(x[idx], a[idx], yy, DOSES, grid, args, root / label / f"run_{r:03d}",
                         bandwidth_multiplier=scale, diagnostic_multipliers=(1,),
                         legacy=True, quantiles=q)
            runs.append(np.concatenate([ref[:, None, :], scores.mean(axis=2)], axis=1))
        np.savez_compressed(root / f"{label}.npz", runs=np.stack(runs))
    summarize(root, args.output / "trained_summary", table_only=args.preset == "table1")


def main():
    args = arguments()
    if args.mode in {"quick", "archive"}:
        summarize(HERE / "reference", args.output / "archive")
        if args.mode == "archive":
            return
    if args.mode == "generate":
        make_pool(args.pool or args.output / "simulation_pool.npz", args.data_seed)
        return
    configure(args)
    grid = np.asarray(args.quantiles if args.quantiles is not None else np.linspace(.1,.9,9))
    if args.mode == "train":
        train_suite(args, grid)
    else:
        example = args.mode != "fit"
        application(args, grid, args.output / ("example" if example else "application"), example)


if __name__ == "__main__":
    main()
