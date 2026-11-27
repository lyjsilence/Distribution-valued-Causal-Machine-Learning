"""Cross-fitted simulations with fold-level overlap and training diagnostics."""
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, TensorDataset

from .bandwidth import kernel, select
from .models import Flow_IPW, NFR
from .training import train_density, train_regression


def unit_quantiles(outcomes, grid):
    """Convert repeated scalar outcomes to the paper's empirical quantile grid."""
    grid_index = np.searchsorted(np.linspace(0, 1, outcomes.shape[1]), grid, side="right")
    return np.sort(outcomes, axis=1)[:, np.minimum(grid_index, outcomes.shape[1] - 1)]


def paired_loader(z, q, batch_size, shuffle=False):
    return DataLoader(TensorDataset(torch.as_tensor(z), torch.as_tensor(q)),
                      batch_size=batch_size, shuffle=shuffle)


def fit(x, a, q, doses, grid, config, out, multiplier=1):
    device = torch.device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    xa = np.column_stack((x, a))
    scores = np.empty((len(doses), 3, len(a), len(grid)))
    diagnostics = []
    p_x = x.shape[1]
    network = SimpleNamespace(num_cov=p_x, input_dim=1, cond_dim=p_x, hidden_dim=config.cnf_width,
                              rademacher=False, flow_time=1.0, train_T=True, solver="dopri5",
                              num_blocks=1, batch_norm=True, kinetic_energy=0.1, jacobian_norm2=0.1)
    for fold, (train, test) in enumerate(KFold(n_splits=config.folds).split(x), start=1):
        folder = out / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "split_indices.npz", train=train, score=test)
        counterfactuals = [np.column_stack((x[test], np.full(len(test), d))) for d in doses]
        with (folder / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            nfr = NFR(p_x + 1, p_x + 1, t=grid, dim_hidden=config.nfr_widths, device=device).to(device)
            mean = train_regression(
                nfr, paired_loader(xa[train], q[train], config.batch_size, shuffle=True),
                paired_loader(xa[test], q[test], len(test)),
                [paired_loader(z, q[test], len(test)) for z in counterfactuals],
                device, config, folder / "nfr.pt")
            flow = Flow_IPW(network, device=device).to(device)
            raw_density = train_density(
                flow, DataLoader(xa[train], batch_size=config.batch_size, shuffle=True),
                DataLoader(xa[test], batch_size=len(test)),
                [DataLoader(z, batch_size=len(test)) for z in counterfactuals],
                device, config, folder / "cnf.pt")
        for j, (dose, pred, raw) in enumerate(zip(doses, mean, raw_density)):
            if not np.isfinite(pred).all() or not np.isfinite(raw).all():
                raise RuntimeError(f"Non-finite predictions in fold {fold} at dose {dose}.")
            density = np.maximum(raw, config.density_floor)
            h = select(a[test], dose, density, q[test], pred, config.kernel, config.bandwidth)
            for factor in sorted({0.5, 1, 2, multiplier}):
                w = kernel(config.kernel, h * factor, dose, a[test, None]) / density
                mass = w.sum()
                dml = pred + w * (q[test] - pred)
                diagnostics.append([fold, dose, factor, h * factor, len(test), np.count_nonzero(w),
                                    np.quantile(raw, 0.05), np.median(raw),
                                    np.mean(raw < config.density_floor),
                                    mass**2 / np.square(w).sum() if mass else 0,
                                    w.max() / mass if mass else np.nan, *dml.mean(axis=0)])
                if factor == multiplier:
                    scores[j, 0, test] = pred
                    scores[j, 1, test] = w * q[test]
                    scores[j, 2, test] = dml
        print(f"Completed fold {fold}/{config.folds}", flush=True)
    columns = ["fold", "dose", "bandwidth_multiplier", "bandwidth", "n", "nonzero_weights",
               "raw_density_p05", "raw_density_median", "fraction_density_floored", "weight_ess",
               "max_weight_share", *[f"dml_q{t:g}" for t in grid]]
    pd.DataFrame(diagnostics, columns=columns).to_csv(out / "diagnostics.csv", index=False)
    return scores.mean(axis=2)
