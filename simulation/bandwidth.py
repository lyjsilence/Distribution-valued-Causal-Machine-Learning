"""Kernel weights and bandwidth selection for the simulation."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def kernel(name, h, dose, observed):
    u = (observed - dose) / h
    if name == "gaussian":
        value = norm.pdf(u)
    elif name == "epanechnikov":
        value = 3 / 4 * (1 - u**2) * (np.abs(u) <= 1)
    elif name == "triweight":
        value = 35 / 32 * (1 - u**2)**3 * (np.abs(u) <= 1)
    else:
        raise ValueError(f"Unknown kernel: {name}")
    return (1 / h) * value


def select(a, dose, p, q, m, name, rule):
    pilot = np.std(a) * len(a)**(-0.2)
    if pilot <= 0:
        raise ValueError("The scoring fold has no dose variation.")
    if rule == "pilot":
        return pilot
    observed = a[:, None]
    if rule == "amise":
        pilot_score = m + kernel(name, pilot, dose, observed) * (q - m) / p
        wider_score = m + kernel(name, 2 * pilot, dose, observed) * (q - m) / p
        bias = (wider_score - pilot_score).mean(axis=0) / (3 * pilot**2)
        squared_bias = np.square(bias).sum()
        if squared_bias == 0:
            return pilot
        h = (pilot * pilot_score.var(axis=0).sum() / (4 * len(a) * squared_bias))**0.2
    else:
        def objective(params):
            h = params[0]
            if h <= 0:
                return np.inf
            b = 2 * h
            nominal = kernel(name, h, dose, observed)
            wider = kernel(name, b, dose, observed)
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
