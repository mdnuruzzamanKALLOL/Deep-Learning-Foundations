# Hyperparameter Tuning for Deep Nets

> Status: ✅ Complete

Learning-rate schedules and hyperparameter search — implemented from scratch on a
digit-classification MLP, validated against PyTorch schedulers, and compared on equal
trial budgets.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`15_hyperparameter_tuning_for_deep_nets.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Training a neural network involves many choices that are not learned by gradient descent:
learning rate, hidden width, regularization strength, batch size, and how long to train.
**Hyperparameter tuning** is the process of finding good values for these knobs.

Topic 05 showed that optimizer choice and a **constant** learning rate interact strongly.
Topic 07 showed **early stopping** as regularization. This notebook goes further:

1. **Learning-rate schedules** — start with a larger $\\eta$ for fast progress, then decay
   so updates stay small near a minimum.
2. **Search strategies** — grid search vs random search under a fixed trial budget.

The guiding principle: tune on **validation** data, never the test set, and report how
many trials you ran.

## Mathematical Explanation

**Constant:** $\\eta_t = \\eta_0$

**Step decay:** $\\eta_t = \\eta_0 \\cdot \\gamma^{\\lfloor t / s \\rfloor}$  
Drop by factor $\\gamma$ every $s$ epochs.

**Exponential:** $\\eta_t = \\eta_0 \\cdot \\gamma^t$

**Cosine annealing** (Loshchilov & Hutter, 2017):

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)$$

**Random search** (Bergstra & Bengio, 2012): sample hyperparameters independently (often
log-uniformly for learning rate) instead of placing trials on a Cartesian grid. When a few
dimensions dominate validation error, random search explores those dimensions more densely
for the same number of trials.

## Code Implementation

All code lives in [`15_hyperparameter_tuning_for_deep_nets.ipynb`](15_hyperparameter_tuning_for_deep_nets.ipynb). It:

1. Implements `lr_at_epoch` for constant, step, exponential, and cosine schedules.
2. Trains `DigitMLP` (64 → hidden → 10) on `load_digits` with Adam (Topic 05).
3. Validates cosine schedule values against PyTorch `CosineAnnealingLR`.
4. Sweeps six **constant** learning rates and plots validation accuracy vs $\\eta$.
5. Compares schedules at $\\eta_0 = 0.1$ over 800 epochs.
6. Runs **grid** (3×3) vs **random** (9 log-uniform learning rates) search over $(\\eta, \\text{hidden})$.
7. Connects early stopping (Topic 07) as implicit tuning.

## Key Results

**PyTorch cosine validation** ($\\eta_0=0.02$, $T=600$, $\\eta_{\\min}=10^{-4}$):

| | Max diff |
|---|---|
| NumPy vs `CosineAnnealingLR` | **3.02e-05** |

**Constant LR sensitivity** (400 epochs, hidden=64):

| $\\eta$ | Val acc |
|---|---|
| 0.001 | 0.972 |
| 0.003 | 0.975 |
| **0.01** | **0.978** |
| 0.03 | 0.975 |
| 0.1 | 0.967 |
| 0.3 | 0.953 |

Peak near $\\eta=0.01$; too-large rates overshoot and hurt validation accuracy.

**Schedules at $\\eta_0=0.1$** (800 epochs — honest tie on this easy task):

| Schedule | Final val | Val @ ep 200 |
|---|---|---|
| constant | 0.967 | 0.964 |
| step | 0.967 | 0.964 |
| exponential | **0.969** | **0.969** |
| cosine | 0.964 | 0.964 |

Exponential decay edges ahead slightly; all schedules are within ~0.005 val acc. On harder
models and longer training, schedule choice matters more — here the task saturates quickly.

**9-trial hyperparameter search** (400 epochs/trial):

| Method | Best val | Best config | Unique LRs tried |
|---|---|---|---|
| Grid (3×3) | **0.981** | lr=0.003, hidden=128 | **3** |
| Random (log-uniform) | **0.981** | lr≈0.024, hidden=64 | **9** |

Both find the same best validation score, but random search tries **three times as many
distinct learning rates** — the Bergstra efficiency argument even when peak performance ties.

## Common Pitfalls

1. **Tuning on the test set.** Any metric used to pick hyperparameters becomes training
   signal; hold out a validation split (or cross-validate) and touch the test set once.
2. **Grid search in high dimensions.** A 5×5×5 grid needs 125 trials; most axes may be
   irrelevant. Random search or Bayesian optimization scale better.
3. **Learning rate too high or too low.** Our sweep drops from 0.978 at $\\eta=0.01$ to
   0.953 at $\\eta=0.3$. Topic 05 showed similar sensitivity across optimizers on moons.
4. **Assuming schedules always beat constant LR.** On saturated small tasks they can tie;
   schedules help most with long runs and aggressive $\\eta_0$.
5. **Coupling schedule with wrong optimizer settings.** Adam already adapts per-parameter
   scales; very large $\\eta_0$ with Adam can still diverge — tune schedule and base LR together.
6. **Reporting best-of-many-trials without variance.** Run multiple seeds, report mean ± std,
   and disclose the trial budget (9, 50, 500, …).

## Function Reference

| Function / Class | Purpose |
|---|---|
| `lr_at_epoch` | Closed-form LR for constant / step / exponential / cosine |
| `train_with_schedule` | Train `DigitMLP` with a schedule; return val-acc history |
| `run_trial` | Single (lr, hidden) training run |
| `hyperparameter_search` | Grid or random search with fixed trial budget |
| `DigitMLP` | 64 → hidden → 10 ReLU classifier |
| `Adam` | Optimizer (Topic 05) |

## Self-Test

1. Write the cosine annealing formula. What are $\\eta_0$, $\\eta_{\\min}$, and $T$?
2. Why sample learning rates log-uniformly in random search rather than uniformly in $[0.001, 0.1]$?
3. Our grid and random search both reach 0.981 val acc. Why might random search still be
   preferable with the same 9 trials?
4. At $\\eta=0.3$ validation accuracy falls to 0.953. What is likely happening to the
   optimization trajectory?
5. How does early stopping (Topic 07) relate to hyperparameter tuning?
6. Why did we compare cosine schedules to PyTorch instead of re-deriving Adam from scratch again?

## Further Reading

- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization* (2012)
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts* (2017)
- Topic 05 — optimizers and constant-LR robustness sweeps
- Topic 07 — early stopping as regularization
- Topic 08 — LR sensitivity with BatchNorm

---
[← Back to Deep Learning Foundations](../README.md)
