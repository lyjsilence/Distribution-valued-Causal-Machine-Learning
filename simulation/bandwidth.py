"""Epanechnikov weights and the bandwidth selector used for Figure 3."""
import numpy as np
from scipy.optimize import minimize


def kernel(h, dose, observed):
    u = (observed - dose) / h
    value = 3 / 4 * (1 - u**2) * (np.abs(u) <= 1)
    return (1 / h) * value


def select(a, dose, p, q):
    pilot = np.std(a) * len(a)**(-0.2)
    if pilot <= 0:
        raise ValueError("The scoring fold has no dose variation.")
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
