# Distribution-valued Causal Machine Learning

Software and reproducibility materials for *Causal Policy Evaluation with Distribution-valued Outcomes under Continuous Interventions* (IJDS-2026-0219).

The observational unit contributes a continuous treatment, pre-treatment covariates, and repeated outcome measurements. The software forms a quantile-function outcome for each unit and estimates the population average of those quantile functions at each target dose. The estimator combines neural functional regression (NFR), a conditional normalizing flow (CNF) for the treatment density, and a cross-fitted kernel-localized adjustment.

## Install and run

Create and activate a Python 3.8–3.11 virtual environment. The training workflow was checked on Python 3.8.20 and PyTorch 2.3.0, including a CPU-only run of the worked example. Use a PyTorch 2.3.0 wheel that matches your OS and CUDA environment if you plan to train on a GPU.

```bash
python -m pip install -r requirements.txt
python replicate.py
```

The second command rebuilds the simulation table and figures from the public reference outputs and trains the synthetic application. It also runs the within-unit sensitivity check. Outputs are written to `outputs/archive/` and `outputs/example/`. Use `--device cpu` to force CPU execution or `--output PATH` to choose another directory. Running the example is intended as a check of the full workflow.

For the reference summaries alone, install `requirements-summary.txt` and run the archive command.

```bash
python replicate.py --mode archive
```

## Simulation reproduction

| Command or data | Contents |
| --- | --- |
| `python replicate.py --mode archive` | Rebuilds Table 1 and simulation Figures 3–5 from public synthetic run outputs. |
| `python replicate.py --mode generate` | Generates the synthetic 100,000-unit input pool with 100 repeated measurements per unit. The seed and full generator are included. |
| `python replicate.py --mode train --preset table1 --reps 1` | Fits both nuisance models from random initialization for one two-fold replication of the 50,000-unit Table 1 configuration. |
| `python replicate.py --mode train` | Fits the full 100-replication simulation suite for Table 1 and Figures 3–5, including sample-size and bandwidth experiments. A CUDA GPU is recommended for this run. |
| `reference/simulation_excerpt.npz` | 96 synthetic units from the original simulation pool. It contains aligned arrays `X` (96 × 10), `A` (96,), and `Y` (96 × 100). |
| `reference/n50000.npz` | 100 synthetic simulation runs. Array `runs` has dimensions run, dose, method, quantile. Dose order is −0.5, 0, 0.5 and method order is truth, Dist-DR, Dist-IPW, Dist-DML. |
| Other `reference/*.npz` files | 100 synthetic runs per scenario with the same axes and two methods, truth and Dist-DML. The `n*` files vary sample size. The `h_*` files vary the bandwidth at 10,000 units. |

Every saved configuration uses the quantile grid 0.1, 0.2, …, 0.9. `table1.csv` uses the paper's historical aggregation, including the absolute error of the mean across runs. `table1_run_mae.csv` also reports the mean and standard deviation of the MAE calculated separately within each run. The Figure 3 shaded regions show empirical percentiles across runs. Figures 4 and 5 use run-level MAE and its empirical standard deviation. Figure 5 compares bandwidths at a common sample size of 10,000.

Full retraining uses the supplied generator, program settings, and current software environment. Floating-point results can vary with dependency versions, device, and stochastic optimization. The archived synthetic arrays provide the numerical reference for comparison. Original per-run model checkpoints are unnecessary for the fast summary command.

## Application workflow and data access

The platform transaction records used for the article's empirical analysis are proprietary. The published Figure 6 requires authorized access to those records. The public release contains the **synthetic simulation data and synthetic run outputs**. `figure6_synthetic.pdf` demonstrates the workflow on newly generated individual-level data with variable numbers of observations. Its numerical results describe the synthetic example.

The example writes two CSVs under `outputs/example/data/`. You can inspect them and run the same interface locally with data you are authorized to use.

```bash
python replicate.py --mode fit --units units.csv --outcomes outcomes.csv --doses 6000 8000 10000 12000 14000
```

| Input | Structure |
| --- | --- |
| `units.csv` | One row per unit. Columns are `unit_id`, a continuous numeric `dose`, and one or more numeric pre-treatment covariates. |
| `outcomes.csv` | One row per repeated measurement. Columns are `unit_id` and a numeric `outcome`. All units must have at least one outcome, and both files must cover the same IDs. |
| `--doses` | Target doses in the units of the input file. Choose values supported by the observed data. |

The outcome construction sorts each unit's measurements and evaluates the inverse empirical CDF on the requested grid. Each unit contributes one curve to the estimator. The standard workflow scales covariates and doses using the outer fold's fitting subset. Results use the original dose units. The historical simulation uses the original order-statistic convention, the scoring fold for validation monitoring, the last saved checkpoint, and the legacy bandwidth rule. The worked example and `fit` reserve validation within the outer training fold and choose the checkpoint by its validation loss.

## Defaults and diagnostics

| Component | Default and guidance |
| --- | --- |
| Cross-fitting | Two outer folds with unit-level splitting. `--folds` controls their number. The application shuffles units using the training seed, and the historical simulation preserves its original split order. |
| Quantile grid | Nine points from 0.1 to 0.9. `--quantiles` accepts at least eight increasing interior points for the cubic spline. Denser tail grids need enough measurements per unit. |
| Kernel and bandwidth | Epanechnikov. The application uses a pilot of sample dose SD × fold size^(−1/5). `--bandwidth amise` uses the DML score plug-in, and `--bandwidth legacy` uses the historical simulation selector. `--kernel` also accepts `gaussian` and `triweight`. |
| NFR | Two ReLU layers of width 64, with a cubic B-spline output head. Adjust with `--nfr-widths`. The short example uses 12 epochs and full simulation uses 200. `--epochs-dr` overrides these. |
| CNF | One conditional ODE block with a width-32 network, a conditional Gaussian base, a Dormand–Prince solver, and density regularization. `--cnf-width` controls width. The example uses up to 4 epochs and full simulation up to 50. `--epochs-cnf` controls the cap. Validation NLL and stability of the estimated densities help assess longer fits. |
| Optimization and randomness | Adam, learning rate 0.003, weight decay 0.001, batch size 128. The training and synthetic data seeds default to 1 and are set separately via `--seed` and `--data-seed`. |
| Density and weights | The propensity density is floored at 0.01 on the scale used for fitting. `--density-floor` changes this value. Examine the sensitivity of conclusions when flooring is frequent. |

The fitted workflow writes `curve.csv` with the estimate and pointwise and within-dose simultaneous intervals over the finite quantile grid. These are multiplier intervals based on centered unit scores with fixed nuisance fits. They leave smoothing bias outside the interval calculation. Each fold also writes preprocessing parameters, split indices, a training log, and the NFR and CNF checkpoints.

- **Overlap and weight stability**. `diagnostics.csv` contains unfloored density summaries, the fraction floored, local nonzero weight count, effective sample size, and maximum normalized weight. Small effective samples or concentrated weights call for cautious interpretation and a better-supported dose grid.
- **Convergence**. `fold_*/training.log` contains training losses, validation MAE and NLL, and iterations. The worked example uses short budgets. Use validation loss trajectories to assess whether a longer fit is warranted for an application.
- **Bandwidth sensitivity**. The diagnostics include scores and weight statistics for half, nominal, and twice the selected bandwidth. The synthetic figure workflow and published simulation reproduction use the same kernel-localized estimator.
- **Within-unit sensitivity**. The synthetic application repeats the fit after randomly halving the measurement count in each unit. `within_unit_sensitivity.csv` reports changes across the dose and quantile grid. Use `--within-sensitivity` to run the same check on authorized local data.

The original simulation entry point, `Simulation.py`, is retained for existing users. `replicate.py` is the supported entry point for the reviewer-facing workflows.

