"""Fit the distributional dose-response using the archived nuisance architectures."""
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
import torch

from models import Flow_IPW, NFR
import optimal_h
from training import train_IPW, train_regression
from utils import sim_dataset_reg


def unit_quantiles(outcomes, grid, legacy=False):
    rows = []
    for obs in outcomes:
        obs = np.sort(np.asarray(obs, dtype=float))
        if not len(obs) or not np.isfinite(obs).all():
            raise ValueError("Every unit must have at least one finite outcome.")
        if legacy:
            idx = np.searchsorted(np.linspace(0, 1, len(obs)), grid, side="right")
            rows.append(obs[np.minimum(idx, len(obs)-1)])
        else:
            rows.append(np.quantile(obs, grid, method="inverted_cdf"))
    return np.stack(rows)


def read_long(units_file, outcomes_file):
    units, obs = pd.read_csv(units_file), pd.read_csv(outcomes_file)
    if units.unit_id.isna().any() or obs.unit_id.isna().any() or units.unit_id.duplicated().any():
        raise ValueError("unit_id must be complete, with one row per unit in the units CSV.")
    groups = obs.groupby("unit_id").outcome.apply(list)
    if set(units.unit_id) != set(groups.index):
        raise ValueError("The two CSVs must contain the same set of unit IDs.")
    x = units.drop(columns=["unit_id", "dose"]).to_numpy(float)
    a = units.dose.to_numpy(float)
    return x, a, [np.asarray(groups.loc[i]) for i in units.unit_id]


def choose_h(a, dose, p, q, m, kernel, rule):
    pilot = np.std(a) * len(a)**(-0.2)
    if pilot <= 0:
        raise ValueError("A scoring fold has no dose variation and cannot support a local estimate.")
    if rule == "pilot":
        return pilot
    if rule == "legacy":
        h = optimal_h.optbandwidth_surface(dose, p, m, a[:, None], q, kernel)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            h = optimal_h.ex_surface(pilot, dose, q, m, p, kernel, a[:, None], len(a))
    return h if np.isfinite(h) and h > 0 else pilot


def weights(a, dose, h, kernel, p):
    return optimal_h.kernel(kernel, h, dose, a[:, None]) / p


def fit(x, a, outcomes, doses, grid, config, out, bandwidth_multiplier=1,
        diagnostic_multipliers=(0.5, 1, 2), legacy=False, quantiles=None):
    q = unit_quantiles(outcomes, grid, legacy=legacy) if quantiles is None else quantiles
    x, a, doses, grid = map(np.asarray, (x, a, doses, grid))
    if not np.isfinite(x).all() or not np.isfinite(a).all() or not np.isfinite(doses).all():
        raise ValueError("Covariates, doses, and target doses must be finite.")
    if x.ndim != 2 or x.shape[1] == 0 or len(x) != len(a) or len(q) != len(a):
        raise ValueError("X must have one row per unit and at least one numeric covariate.")
    if np.std(a) == 0 or len(a) < max(12, config.folds * 4):
        raise ValueError("Supply varying doses and enough units for the requested folds.")
    if (len(grid) < 8 or not np.isfinite(grid).all()
            or not (0 < grid[0] < grid[-1] < 1) or np.any(np.diff(grid) <= 0)):
        raise ValueError("The cubic spline needs at least eight strictly increasing quantiles in (0,1).")
    device = torch.device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    scores = np.empty((len(doses), 3, len(a), len(grid)))
    diagnostics = []
    splitter = KFold(n_splits=config.folds, shuffle=not legacy,
                     random_state=None if legacy else config.seed)
    for k, (outer, test) in enumerate(splitter.split(x), start=1):
        train, val = (outer, test) if legacy else train_test_split(
            outer, test_size=0.2, random_state=config.seed+k)
        if legacy:
            tx, ta, eval_doses = x, a, doses
        else:
            mu, sd = x[train].mean(axis=0), x[train].std(axis=0)
            tx = (x-mu) / np.where(sd > 0, sd, 1)
            dm, ds = a[train].mean(), a[train].std()
            if ds == 0:
                raise ValueError("At least one training fold has no variation in dose.")
            ta, eval_doses = (a-dm)/ds, (doses-dm)/ds

        xa = np.column_stack((tx, ta))
        at_dose = [np.column_stack((tx[test], np.full(len(test), d))) for d in eval_doses]
        fold_path = Path(out) / f"fold_{k}"
        (fold_path / "Simulation" / f"Size={len(a)}").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_path / "split_indices.npz", train=train, validation=val, score=test)
        np.savez_compressed(fold_path / "preprocessing.npz",
                            x_mean=np.zeros(x.shape[1]) if legacy else mu,
                            x_scale=np.ones(x.shape[1]) if legacy else np.where(sd > 0, sd, 1),
                            dose_mean=0 if legacy else dm, dose_scale=1 if legacy else ds)
        p_x = x.shape[1]
        args = SimpleNamespace(
            results_save_path=str(fold_path), sample_size=len(a), lr=config.lr,
            weight_decay=.001, epochs_DR=config.epochs_dr, epochs_IPW=config.epochs_cnf,
            num_cov=p_x, num_obs=0, input_dim=1, cond_dim=p_x, hidden_dim=config.cnf_width,
            rademacher=False, flow_time=1.0, train_T=True, solver="dopri5", num_blocks=1,
            batch_norm=True, kinetic_energy=.1, jacobian_norm2=.1, directional_penalty=None,
            select_best=not legacy, log_batches=False)
        batch = config.batch_size
        with (fold_path / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            nfr = NFR(p_x+1, p_x+1, t=grid, dim_hidden=config.nfr_widths, device=device).to(device)
            mean = train_regression(
                args, nfr, DataLoader(sim_dataset_reg(xa[train], q[train]), batch_size=batch, shuffle=True),
                DataLoader(sim_dataset_reg(xa[val], q[val]), batch_size=len(val)),
                [DataLoader(sim_dataset_reg(z, q[test]), batch_size=len(z)) for z in at_dose], device)
            flow = Flow_IPW(args, device=device).to(device)
            raw_p = train_IPW(
                args, flow, DataLoader(xa[train], batch_size=batch, shuffle=True),
                DataLoader(xa[val], batch_size=len(val)),
                [DataLoader(z, batch_size=len(z)) for z in at_dose], device)
        for j, (dose, pred, p_raw) in enumerate(zip(eval_doses, mean, raw_p)):
            if not np.isfinite(pred).all() or not np.isfinite(p_raw).all():
                raise RuntimeError(f"Non-finite predictions in fold {k} at dose {doses[j]}.")
            p = np.maximum(p_raw, config.density_floor)
            h = choose_h(ta[test], dose, p, q[test], pred, config.kernel, config.bandwidth)
            for factor in sorted(set(diagnostic_multipliers) | {bandwidth_multiplier}):
                w = weights(ta[test], dose, h*factor, config.kernel, p)
                s = w.sum()
                estimates = (pred + w*(q[test]-pred)).mean(axis=0)
                diagnostics.append([k, doses[j], factor, h*factor, len(test), np.count_nonzero(w),
                                    np.quantile(p_raw, .05), np.median(p_raw),
                                    np.mean(p_raw < config.density_floor),
                                    s**2 / np.square(w).sum() if s else 0,
                                    w.max() / s if s else np.nan, *estimates])
                if factor == bandwidth_multiplier:
                    scores[j, 0, test] = pred
                    scores[j, 1, test] = w*q[test]
                    scores[j, 2, test] = pred + w*(q[test]-pred)
        print(f"Completed outer fold {k}/{config.folds}", flush=True)
    pd.DataFrame(diagnostics, columns=["fold", "dose", "bandwidth_multiplier", "bandwidth_fit_scale",
                                      "n", "nonzero_weights", "raw_density_p05", "raw_density_median",
                                      "fraction_density_floored", "weight_ess", "max_weight_share",
                                      *[f"dml_q{t:.4g}" for t in grid]]).to_csv(
        Path(out) / "diagnostics.csv", index=False)
    return scores


def bands(scores, doses, grid, out, seed, draws=499):
    """Report pointwise and within-dose simultaneous multiplier intervals on the grid."""
    rng = np.random.default_rng(seed)
    rows = []
    for dose, v in zip(doses, scores[:, 2]):
        center = v.mean(axis=0)
        residual = v-center
        se = residual.std(axis=0, ddof=1)/np.sqrt(len(v))
        delta = np.concatenate([rng.standard_normal((min(32, draws-k), len(v))) @ residual / len(v)
                                for k in range(0, draws, 32)], axis=0)
        lo, hi = np.quantile(delta, [.025, .975], axis=0)
        critical = np.quantile(np.max(np.abs(delta) / np.where(se > 0, se, 1), axis=1), .95)
        for t, e, s, l, h in zip(grid, center, se, lo, hi):
            rows.append([dose, t, e, s, e+l, e+h, e-critical*s, e+critical*s])
    df = pd.DataFrame(rows, columns=["dose", "quantile", "estimate", "se", "point_lower",
                                    "point_upper", "simultaneous_lower", "simultaneous_upper"])
    df.to_csv(Path(out) / "curve.csv", index=False)
    return df
