"""Plot the quantile-function estimates and run-to-run variation in Figure 3."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ("Dist-DR", "Dist-IPW", "Dist-DML")
QUANTILES = np.arange(1, 10) / 10


def load_runs(path):
    with np.load(path, allow_pickle=False) as bundle:
        return bundle["runs"]


def plot_figure3(runs, out):
    if runs.ndim != 3 or len(runs) == 0 or runs.shape[1:] != (4, 9):
        raise ValueError("Expected runs by four curves (truth and three estimators) by nine quantiles.")
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True, layout="constrained")
    for j, (method, ax) in enumerate(zip(METHODS, axes), start=1):
        curve = runs[:, j]
        lower, upper = np.quantile(curve, [0.025, 0.975], axis=0)
        ax.fill_between(QUANTILES, lower, upper, alpha=0.2, label="Central 95% of runs")
        ax.plot(QUANTILES, curve.mean(axis=0), label="Estimate")
        ax.plot(QUANTILES, runs[:, 0].mean(axis=0), "--", label="Truth")
        ax.set(title=method, xlabel="Quantile index")
    axes[0].set_ylabel("Distributional outcome")
    axes[-1].legend(fontsize="small")
    fig.savefig(out / "figure3.pdf")
    fig.savefig(out / "figure3.png", dpi=135)
    plt.close(fig)
