# Distribution-valued Causal Machine Learning

Simulation code for Figure 3 of *Causal Policy Evaluation with Distribution-valued Outcomes under Continuous Interventions*.

## Requirements

Python 3.8 to 3.11. Install the dependencies and run the commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

Training uses CUDA when available. Use `--device cpu` for CPU training.

## Repository structure

| File | Purpose |
| --- | --- |
| `replicate.py` | Run the simulation and plot Figure 3 |
| `simulation/data.py` | Generate synthetic data and the true quantile curve |
| `simulation/estimation.py` | Train the nuisance models and compute cross-fitted estimates and diagnostics |
| `simulation/models.py`, `simulation/batchnorm.py` | NFR and CNF architectures |
| `reference/figure3_runs.npz` | 100 runs at dose zero, with shape `(100, 4, 9)` for runs, curves, and quantiles. Curve order is truth, Dist-DR, Dist-IPW, Dist-DML. |

## 1. Reproduce Figure 3

```bash
python replicate.py
```

The command plots the archived runs and saves `figure3.png` and `figure3.pdf` under `outputs/archive/`.

![Figure 3. True and estimated quantile curves at dose zero for Dist-DR, Dist-IPW, and Dist-DML.](figures/figure3.png)

Curves show the mean across runs at dose zero. Shading shows the 2.5th and 97.5th percentiles across runs.

## 2. Generate data and retrain

```bash
python replicate.py --mode train
```

This command generates a pool of 100,000 synthetic units and trains 100 replications on a fixed sample of 50,000 units. The pool contains `X` (10 covariates), `A` (continuous dose), and `Y` (100 repeated measurements per unit). The estimator sorts each unit's measurements and evaluates its empirical quantile curve at levels 0.1 to 0.9.

The command reuses an existing simulation pool. Use `--pool PATH` to select an input file or `--mode generate` to generate a pool separately. Set `--data-seed` and `--seed` for data generation and training. Both default to 1.

| Setting | Default |
| --- | --- |
| Cross-fitting | Two folds |
| Outcome regression | NFR with two ReLU hidden layers of width 64 and a cubic spline output, 200 epochs |
| Generalized propensity density | CNF with one ODE block of width 32, a conditional Gaussian base, and the Dormand-Prince solver, up to 50 epochs. The estimated density is floored at 0.01. |
| Kernel and bandwidth | Epanechnikov kernel. The selector minimizes the estimated squared bias plus variance, initialized at the fold's dose standard deviation times the fold size to the power `-1/5`. |
| Optimization | Adam with learning rate 0.003 and batch size 128 |

## 3. Worked example and diagnostics

```bash
python replicate.py --mode train --device cpu --sample-size 1000 --reps 1 --epochs-dr 1 --epochs-cnf 1 --output outputs/example
```

This runs both folds and draws Figure 3 in `outputs/example/trained/`. The training directory contains the run-level curves, fold checkpoints and loss logs. Each `run_*/diagnostics.csv` reports bandwidth, local sample count, propensity density, density flooring, effective sample size, and the largest normalized weight. The simulation logs the loss on each scoring fold and uses it in the learning-rate scheduler.

To check sensitivity, repeat the example with `--bandwidth-multiplier 0.5` or `--observations 50`. Reuse `--pool outputs/example/simulation_pool_1000.npz` and choose a separate `--output` folder. Use `python replicate.py --help` for options.
