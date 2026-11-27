import numpy as np
import pandas as pd
import torch
from contextlib import redirect_stdout
from types import SimpleNamespace
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .data import DOSES, FIGURE_DOSE
from .models import Flow_IPW, NFR

# Kernel and bandwidth
def kernel(h, dose, observed):
    u = (observed - dose) / h
    value = 3 / 4 * (1 - u**2) * (np.abs(u) <= 1)
    return (1 / h) * value


def select(a, dose, p, q):
    pilot = np.std(a) * len(a)**(-0.2)
    observed = a[:, None]

    def objective(params):
        h = params[0]
        if h <= 0:
            return np.inf
        b = 2 * h
        nominal = kernel(h, dose, observed)
        wider = kernel(b, dose, observed)
        loss = 0
        for curve in q.T:
            y = curve[:, None]
            v_h = nominal * y / p
            v_b = wider * y / p
            bias = np.mean(v_b - v_h, axis=0) / (0.75 * b**2)
            loss += v_h.var() / len(a) + bias**2 * h**4
        return (loss + params[1]).item()

    h = minimize(objective, (pilot, 0), method="SLSQP",
                 bounds=((0, None), (0, 0))).x[0]
    return h if np.isfinite(h) and h > 0 else pilot

# Nuisance estimation
def train_regression(model, train, validation, evaluation, device, config, checkpoint):
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-4)
    mae = nn.L1Loss()
    saved = False
    for epoch in range(1, config.epochs_dr + 1):
        model.train()
        train_loss = []
        for x, y in train:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            loss = mae(model(x), y)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x, y = next(iter(validation))
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            loss = mae(model(x), y).item()
        print(f"Epoch {epoch} | train MAE={np.mean(train_loss):.6f} | validation MAE={loss:.6f}")
        if np.isfinite(loss):
            torch.save(model.state_dict(), checkpoint)
            saved = True
        scheduler.step(loss)
    if not saved:
        raise RuntimeError("The outcome regression produced no finite validation checkpoints.")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        curves = [model(next(iter(loader))[0].to(device=device, dtype=torch.float32)).cpu().numpy()
                  for loader in evaluation]
    torch.cuda.empty_cache()
    return curves


def train_density(model, train, validation, evaluation, device, config, checkpoint):
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-4)
    best_training = np.inf
    stalled = 0
    saved = False
    for epoch in range(1, config.epochs_cnf + 1):
        model.train()
        train_loss = []
        for z in train:
            optimizer.zero_grad()
            z = z.to(device=device, dtype=torch.float32)
            nll, penalty = model(z[:, :-1], z[:, -1])
            (nll + penalty).backward()
            optimizer.step()
            train_loss.append(nll.item())
        training = np.mean(train_loss)
        if training < best_training:
            best_training = training
            stalled = 0
        else:
            stalled += 1
        if stalled > 20:
            print("Stopped after 20 epochs without improvement in training NLL.")
            break
        model.eval()
        with torch.no_grad():
            z = next(iter(validation)).to(device=device, dtype=torch.float32)
            loss, _ = model(z[:, :-1], z[:, -1])
            loss = loss.item()
        print(f"Epoch {epoch} | train NLL={training:.6f} | validation NLL={loss:.6f}")
        if np.isfinite(loss):
            torch.save(model.state_dict(), checkpoint)
            saved = True
        scheduler.step(loss)
    if not saved:
        raise RuntimeError("The density estimator produced no finite validation checkpoints.")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    predictions = []
    with torch.no_grad():
        for loader in evaluation:
            z = next(iter(loader)).to(device=device, dtype=torch.float32)
            predictions.append(model(z[:, :-1], z[:, -1], test=True).cpu().numpy())
    torch.cuda.empty_cache()
    return predictions

# Cross-fitted estimates
def unit_quantiles(outcomes, grid):
    """Convert repeated scalar outcomes to the paper's empirical quantile grid."""
    grid_index = np.searchsorted(np.linspace(0, 1, outcomes.shape[1]), grid, side="right")
    return np.sort(outcomes, axis=1)[:, np.minimum(grid_index, outcomes.shape[1] - 1)]


def paired_loader(z, q, batch_size, shuffle=False):
    return DataLoader(TensorDataset(torch.as_tensor(z), torch.as_tensor(q)),
                      batch_size=batch_size, shuffle=shuffle)


def fit(x, a, q, grid, config, out):
    device = torch.device(config.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    xa = np.column_stack((x, a))
    scores = np.empty((3, len(a), len(grid)))
    diagnostics = []
    p_x = x.shape[1]
    network = SimpleNamespace(num_cov=p_x, input_dim=1, cond_dim=p_x, hidden_dim=config.cnf_width,
                              rademacher=False, flow_time=1.0, train_T=True, solver="dopri5",
                              num_blocks=1, batch_norm=True, kinetic_energy=0.1, jacobian_norm2=0.1)
    for fold, (train, test) in enumerate(KFold(n_splits=config.folds).split(x), start=1):
        folder = out / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "split_indices.npz", train=train, score=test)
        # Follow the original prediction order to preserve the random stream across folds.
        counterfactuals = [np.column_stack((x[test], np.full(len(test), d))) for d in DOSES]
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
        target = DOSES.index(FIGURE_DOSE)
        pred, raw = mean[target], raw_density[target]
        density = np.maximum(raw, config.density_floor)
        h = config.bandwidth_multiplier * select(a[test], FIGURE_DOSE, density, q[test])
        w = kernel(h, FIGURE_DOSE, a[test, None]) / density
        mass = w.sum()
        scores[0, test] = pred
        scores[1, test] = w * q[test]
        scores[2, test] = pred + w * (q[test] - pred)
        diagnostics.append([fold, FIGURE_DOSE, h, len(test), np.count_nonzero(w),
                            np.quantile(raw, 0.05), np.median(raw),
                            np.mean(raw < config.density_floor),
                            mass**2 / np.square(w).sum() if mass else 0,
                            w.max() / mass if mass else np.nan])
        print(f"Completed fold {fold}/{config.folds}", flush=True)
    columns = ["fold", "dose", "bandwidth", "n", "nonzero_weights",
               "raw_density_p05", "raw_density_median", "fraction_density_floored", "weight_ess",
               "max_weight_share"]
    pd.DataFrame(diagnostics, columns=columns).to_csv(out / "diagnostics.csv", index=False)
    return scores.mean(axis=1)
