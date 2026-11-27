# Distribution-valued Causal Machine Learning

Simulation code for *Causal Policy Evaluation with Distribution-valued Outcomes under Continuous Interventions*.

## Requirements

Use Python 3.8 to 3.11 and run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

Training uses CUDA when available. Use `--device cpu` to train on the CPU.

## Repository structure

| File | Purpose |
| --- | --- |
| `replicate.py` | Generate data, run replications, and aggregate results |
| `simulation/data.py` | Synthetic data and true quantile curves |
| `simulation/estimation.py` | Nuisance training, cross-fitted estimation, and diagnostics |
| `simulation/models.py`, `simulation/batchnorm.py` | NFR and CNF architectures |
| `reference/simulation_runs.npz` | Archived simulation estimates |

## 1. Generate data

```bash
python replicate.py --mode generate
```

The default pool contains 100,000 synthetic units. Each has 10 covariates `X`, a continuous dose `A`, and 100 repeated outcomes `Y`. The estimator sorts each unit's outcomes and evaluates its empirical quantile curve at levels 0.1 to 0.9. The pool is saved to `outputs/simulation_pool.npz`.

## 2. Run the simulations

```bash
python replicate.py --mode train
```

The command generates or reuses the pool, draws a fixed sample of 50,000 units, and fits 100 replications. Each replication estimates the curves at doses -0.5, 0, and 0.5 using two folds. Both `--data-seed` and `--seed` default to 1.

| Setting | Default |
| --- | --- |
| Outcome regression | NFR with two 64-unit ReLU layers and cubic spline output, 200 epochs |
| Propensity density | CNF with one ODE block of width 32, a conditional Gaussian base, and the Dormand-Prince solver, up to 50 epochs |
| Kernel and bandwidth | Epanechnikov kernel, with bandwidth selected separately by fold and dose using the estimated squared bias and variance |
| Optimization | Adam with learning rate 0.003 and batch size 128. The estimated density is floored at 0.01. |

The following command runs a short example on the CPU.

```bash
python replicate.py --mode train --device cpu --sample-size 1000 --reps 1 --epochs-dr 1 --epochs-cnf 1 --output outputs/example
```

Each `run_*/diagnostics.csv` reports bandwidth, propensity density, density flooring, effective sample size, and weight concentration by fold and dose. Fold logs report training and scoring losses. To check sensitivity, rerun with `--bandwidth-multiplier 0.5` or `--observations 50`, reuse the same pool with `--pool`, and choose a separate output folder.

## 3. Aggregate results

```bash
python replicate.py
```

This command aggregates the 100 archived replications and returns **Table 1** as `table1.csv` and **Figure 3** as `figure3.png` and `figure3.pdf` in `outputs/archive/`. Table 1 uses the per-quantile absolute error of the across-run mean. The default output and the image below use the original manuscript figure.

`--mode train` aggregates the new results in `outputs/trained/`. To aggregate a saved run file separately, use `python replicate.py --mode aggregate --runs PATH`. This also allows Figure 3 to be redrawn directly from `reference/simulation_runs.npz` using the manuscript plotting settings.

![Figure 3. Ground and estimated quantile functions at dose zero for Dist-DR, Dist-IPW, and Dist-DML.](figures/figure3.png)
