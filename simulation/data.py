"""Synthetic data for the continuous-dose simulation."""
from pathlib import Path

import numpy as np
from scipy.stats import beta

DOSES = (-0.5, 0.0, 0.5)
FIGURE_DOSE = 0.0
SEED = 1
POOL_SIZE = 100_000


def outcome_weights(x):
    w = np.exp(x[:, ::2] * x[:, 1::2])
    return w / w.sum(axis=1, keepdims=True)


def generate(n=POOL_SIZE, observations=100, seed=SEED):
    """Return aligned covariates X, continuous dose A, and repeated outcomes Y."""
    rng = np.random.RandomState(seed)
    x = np.column_stack([rng.normal(loc, 1, (n, 2)) for loc in range(-2, 3)])
    a = rng.normal(x @ np.full(10, 0.1), np.logaddexp(0, x @ np.full(10, 0.05)))
    w = outcome_weights(x)
    y = np.empty((n, observations))
    # Interleave draws by individual, matching the original simulation's draw order.
    for start in range(0, n, 2048):
        end = min(start + 2048, n)
        uniforms = np.empty((end - start, observations))
        noise = np.empty_like(uniforms)
        for i in range(end - start):
            uniforms[i] = rng.uniform(size=observations)
            noise[i] = rng.normal(0, 0.05, size=observations)
        curve = sum(w[start:end, j:j+1] * beta.ppf(uniforms, k / 10, k / 10)
                    for j, k in enumerate(range(1, 6)))
        y[start:end] = (a.mean() + np.exp(a[start:end, None])) * curve + noise
    return x, a, y


def truth(x, a, quantiles):
    """Analytic reference evaluated over the finite synthetic covariate pool."""
    curve = sum(weight * beta.ppf(quantiles, k / 10, k / 10)
                for weight, k in zip(outcome_weights(x).mean(axis=0), range(1, 6)))
    return (a.mean() + np.exp(FIGURE_DOSE)) * curve


def save_pool(path, x, a, y):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=x, A=a, Y=y)
