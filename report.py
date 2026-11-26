"""Aggregate simulation replications into the paper table and diagnostic figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOSES = [-0.5, 0.0, 0.5]
METHODS = ["Truth", "Dist-DR", "Dist-IPW", "Dist-DML"]
QUANTILES = np.arange(1, 10) / 10
SIZES = [1000, 5000, 10000, 30000, 50000, 80000, 100000]
SENSITIVITY = {"h_over_6": 1/6, "h_over_4": 1/4, "h_over_2": 1/2,
               "n10000": 1, "h_times_2": 2, "h_times_4": 4, "h_times_6": 6}


def load_runs(path):
    with np.load(path, allow_pickle=False) as bundle:
        return bundle["runs"]


def table1(runs, out):
    means = runs.mean(axis=0)
    sd = runs.std(axis=0)
    rows, dr_rows = [], []
    for i, dose in enumerate(DOSES):
        for j, method in enumerate(METHODS):
            values = means[i, j]
            rows.append([dose, method, "Mean", *values, np.nan])
            if j:
                off = np.abs(values - means[i, 0])
                rows.append([dose, method, "SD", *sd[i, j], np.nan])
                rows.append([dose, method, "Abs. error of mean", *off, off.mean()])
                run_mae = np.abs(runs[:, i, j] - runs[:, i, 0]).mean(axis=1)
                dr_rows.append([dose, method, run_mae.mean(), run_mae.std()])
    pd.DataFrame(rows, columns=["dose", "method", "statistic", *QUANTILES, "Average"]).to_csv(
        out / "table1.csv", index=False, float_format="%.4f")
    pd.DataFrame(dr_rows, columns=["dose", "method", "mean_run_MAE", "sd_run_MAE"]).to_csv(
        out / "table1_run_mae.csv", index=False)


def figure3(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, layout="constrained")
    for j, ax in enumerate(axes, start=1):
        curve = runs[:, 1, j, :]
        lower, upper = np.quantile(curve, [0.025, 0.975], axis=0)
        ax.fill_between(QUANTILES, lower, upper, alpha=0.2, label="2.5–97.5% of runs")
        ax.plot(QUANTILES, curve.mean(axis=0), label="Estimate")
        ax.plot(QUANTILES, runs[:, 1, 0, :].mean(axis=0), "--", label="Truth")
        ax.set(title=METHODS[j], xlabel="Quantile index")
    axes[0].set_ylabel("Distributional outcome")
    axes[-1].legend(fontsize="small")
    fig.savefig(out / "figure3.pdf")
    plt.close(fig)


def sensitivity(groups, name, xlabel, out):
    rows = []
    for setting, runs in groups.items():
        mae = np.abs(runs[:, :, -1, :] - runs[:, :, 0, :]).mean(axis=-1)
        for j, dose in enumerate(DOSES):
            rows.append([setting, dose, mae[:, j].mean(), mae[:, j].std(), len(mae)])
    df = pd.DataFrame(rows, columns=["setting", "dose", "mean_MAE", "sd_MAE", "runs"])
    df.to_csv(out / (name + ".csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for dose, ax in zip(DOSES, axes):
        panel = df[df.dose == dose]
        x, m, s = (panel[c].to_numpy() for c in ["setting", "mean_MAE", "sd_MAE"])
        ax.plot(x, m, ".-")
        ax.fill_between(x, np.maximum(0, m-s), m+s, alpha=0.2)
        ax.set(xscale="log", title=f"Dose {dose:g}", xlabel=xlabel)
    axes[0].set_ylabel("Mean run-level MAE")
    fig.savefig(out / (name + ".pdf"))
    plt.close(fig)


def summarize(reference, out, table_only=False):
    out.mkdir(parents=True, exist_ok=True)
    main = load_runs(reference / "n50000.npz")
    table1(main, out)
    figure3(main, out)
    if not table_only:
        sensitivity({n: load_runs(reference / f"n{n}.npz") for n in SIZES},
                    "figure4", "Number of units", out)
        sensitivity({ratio: load_runs(reference / f"{label}.npz")
                     for label, ratio in SENSITIVITY.items()},
                    "figure5", "Bandwidth multiplier", out)
