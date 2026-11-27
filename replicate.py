"""Reproduce Figure 3 from archived results or from a new training run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
QUANTILES = np.arange(1, 10) / 10
METHODS = ('Dist-DR', 'Dist-IPW', 'Dist-DML')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['archive', 'generate', 'train'], default='archive')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data-seed', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--sample-size', type=int, default=50000)
    parser.add_argument('--observations', type=int, default=100)
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--reps', type=int, default=100)
    parser.add_argument('--epochs-dr', type=int, default=200)
    parser.add_argument('--epochs-cnf', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--nfr-widths', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--cnf-width', type=int, default=32)
    parser.add_argument('--density-floor', type=float, default=0.01)
    parser.add_argument('--bandwidth-multiplier', type=float, default=1.0)
    args = parser.parse_args()
    if args.bandwidth_multiplier <= 0:
        parser.error('--bandwidth-multiplier must be positive.')
    return args


def load_runs(path):
    with np.load(path, allow_pickle=False) as data:
        return data['runs']


def plot_figure3(runs, out):
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, layout='constrained')
    for j, (method, ax) in enumerate(zip(METHODS, axes), start=1):
        curve = runs[:, j]
        lower, upper = np.quantile(curve, [0.025, 0.975], axis=0)
        ax.fill_between(QUANTILES, lower, upper, alpha=0.2, label='Central 95% of runs')
        ax.plot(QUANTILES, curve.mean(axis=0), label='Estimate')
        ax.plot(QUANTILES, runs[:, 0].mean(axis=0), '--', label='Truth')
        ax.set(title=method, xlabel='Quantile index')
    axes[0].set_ylabel('Distributional outcome')
    axes[-1].legend(fontsize='small')
    fig.savefig(out / 'figure3.pdf')
    fig.savefig(out / 'figure3.png', dpi=135)
    plt.close(fig)


def make_pool(args):
    from simulation.data import POOL_SIZE, generate, save_pool

    n = POOL_SIZE if args.sample_size == 50000 else args.sample_size
    name = 'simulation_pool.npz' if n == POOL_SIZE else f'simulation_pool_{n}.npz'
    pool = args.pool or args.output / name
    if args.mode == 'generate' or not pool.exists():
        print(f'Generating {n:,} units.', flush=True)
        save_pool(pool, *generate(n=n, seed=args.data_seed))
    return pool


def train(args, pool):
    import torch
    from simulation.data import truth
    from simulation.estimation import fit, unit_quantiles

    with np.load(pool, allow_pickle=False) as data:
        X, D, Y = data['X'], data['A'], data['Y']
    if not 1 <= args.observations <= Y.shape[1]:
        raise ValueError('--observations must be within the available outcomes per unit.')

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quantiles = np.linspace(0.1, 0.9, 9)
    target = truth(X, D, quantiles)
    idx = np.random.RandomState(args.seed).choice(len(D), args.sample_size, replace=False)
    X, D = X[idx], D[idx]
    Y_inv = unit_quantiles(Y[idx, :args.observations], quantiles)
    torch.manual_seed(args.seed)
    out = args.output / 'trained'
    runs = []
    for r in range(args.reps):
        print(f'Replication {r + 1}/{args.reps} on {args.device}', flush=True)
        estimates = fit(X, D, Y_inv, quantiles, args, out / f'run_{r:03d}')
        runs.append(np.vstack((target, estimates)))
    runs = np.stack(runs)
    np.savez_compressed(out / 'figure3_runs.npz', runs=runs)
    plot_figure3(runs, out)


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'archive':
        runs = load_runs(ROOT / 'reference' / 'figure3_runs.npz')
        plot_figure3(runs, args.output / 'archive')
    else:
        pool = make_pool(args)
        if args.mode == 'train':
            train(args, pool)
