"""Generate data, fit the estimators, and aggregate the simulation results."""
import argparse
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from simulation.data import DOSES

ROOT = Path(__file__).resolve().parent
QUANTILES = np.arange(1, 10) / 10
METHODS = ('Dist-DR', 'Dist-IPW', 'Dist-DML')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['archive', 'generate', 'train', 'aggregate'], default='archive')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--runs', type=Path, help='Simulation results to use in aggregation')
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


def table1(runs, out):
    means, sd = runs.mean(axis=0), runs.std(axis=0)
    rows = []
    for i, dose in enumerate(DOSES):
        ground = means[i, 0]
        rows.append([dose, 'Ground', 'Mean', *ground, np.nan])
        for j, method in enumerate(METHODS, start=1):
            error = np.abs(means[i, j] - ground)
            rows.append([dose, method, 'Mean', *means[i, j], np.nan])
            rows.append([dose, method, 'SD', *sd[i, j], np.nan])
            rows.append([dose, method, 'MAE', *error, error.mean()])
    columns = ['dose', 'method', 'statistic', *[f'Q={q:g}' for q in QUANTILES], 'Avg.']
    pd.DataFrame(rows, columns=columns).to_csv(out / 'table1.csv', index=False, float_format='%.4f')


def plot_figure3(runs, out):
    runs = runs[:, DOSES.index(0.0)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (method, ax) in enumerate(zip(METHODS, axes), start=1):
        curve = runs[:, j]
        lower, upper = np.quantile(curve, [0.025, 0.975], axis=0)
        ax.plot(QUANTILES, runs[:, 0].mean(axis=0), c='r', label='Ground')
        ax.plot(QUANTILES, curve.mean(axis=0), c='b', label=f'{method[5:]} Mean')
        ax.fill_between(QUANTILES, lower, upper, alpha=0.5, color='b', label='95% CI')
        ax.set(xlabel='Quantiles', ylabel=r'$\Theta(a=0.0)$')
        ax.legend()
        ax.grid()
    fig.tight_layout()
    fig.savefig(out / 'figure3.pdf')
    fig.savefig(out / 'figure3.png', dpi=330)
    plt.close(fig)


def aggregate(runs, out, paper_figure=False):
    out.mkdir(parents=True, exist_ok=True)
    table1(runs, out)
    if paper_figure:
        original = ROOT / 'figures' / 'figure3.png'
        copyfile(original, out / 'figure3.png')
        with Image.open(original) as image:
            fig = plt.figure(figsize=(15, 5), dpi=330)
            fig.figimage(np.asarray(image))
            fig.savefig(out / 'figure3.pdf', dpi=330)
            plt.close(fig)
    else:
        plot_figure3(runs, out)


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
        runs.append(np.concatenate((target[:, None, :], estimates), axis=1))
    runs = np.stack(runs)
    np.savez_compressed(out / 'simulation_runs.npz', runs=runs)
    aggregate(runs, out)


if __name__ == '__main__':
    args = parse_args()
    if args.mode in ('archive', 'aggregate'):
        path = args.runs or ROOT / 'reference' / 'simulation_runs.npz'
        aggregate(load_runs(path), args.output / ('archive' if args.runs is None else 'aggregate'),
                  paper_figure=args.runs is None)
    else:
        pool = make_pool(args)
        if args.mode == 'train':
            train(args, pool)
