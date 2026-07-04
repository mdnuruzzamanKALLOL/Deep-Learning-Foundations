# Regularization

> Status: ✅ Complete

Dropout, L1/L2 weight penalty, early stopping, and data augmentation — implemented from
scratch where it matters, validated against PyTorch, and stress-tested on controlled
overfitting scenarios.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`07_regularization.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

A flexible neural network with enough parameters will **always** fit the training set —
including mislabeled examples and random noise — unless you deliberately stop it. Regularization
is the family of techniques that prevents this memorization and pushes the model toward
patterns that generalize.

This topic covers four complementary approaches:

1. **L1/L2 weight penalty** — add a cost for large weights directly to the loss. L2 shrinks
   all weights smoothly; L1 drives many to exactly zero (sparsity).
2. **Dropout** — randomly disable neurons during training, forcing redundant representations
   and preventing co-adaptation.
3. **Early stopping** — stop training when validation performance stops improving, before
   the model overfits noise.
4. **Data augmentation** — create synthetic training examples via label-preserving transforms,
   making exact memorization of coordinates impossible.

## Mathematical Explanation

**L2 (Ridge / weight decay)**

$$L_{\text{total}} = L_{\text{data}} + \frac{\lambda}{2}\sum_i w_i^2, \qquad
\frac{\partial L_{\text{total}}}{\partial w_i} = \frac{\partial L_{\text{data}}}{\partial w_i} + \lambda w_i$$

Penalizes large weights uniformly — prefers many small weights over a few large ones.

**L1 (Lasso)**

$$L_{\text{total}} = L_{\text{data}} + \lambda\sum_i |w_i|, \qquad
\frac{\partial L_{\text{total}}}{\partial w_i} = \frac{\partial L_{\text{data}}}{\partial w_i} + \lambda\,\text{sign}(w_i)$$

The constant-magnitude $\lambda\,\text{sign}(w_i)$ term pushes weights toward *exactly* zero,
unlike L2's proportionally shrinking gradient.

**Dropout (inverted, training time)**

For dropout probability $p$, sample mask $M \sim \text{Bernoulli}(1-p)$ and compute:
$$\tilde{h} = h \odot M / (1-p)$$

Scaling by $1/(1-p)$ keeps $\mathbb{E}[\tilde{h}]=h$, so no adjustment is needed at test time
(all units active, no scaling). At test time, dropout is disabled entirely.

**Early stopping**

Monitor validation loss $L_{\text{val}}$ during training. Save parameters whenever
$L_{\text{val}}$ improves; stop after `patience` epochs without improvement. Return the
saved checkpoint, not the final weights.

## Code Implementation

All code lives in [`07_regularization.ipynb`](07_regularization.ipynb). It:

1. Trains polynomial logistic regression (degree 6, 27 features) under no penalty, L2, and L1.
2. Implements **inverted dropout** from scratch and validates expected output scale against
   `torch.nn.functional.dropout`.
3. Trains a large MLP on **50 noisy labels** (16% flipped) comparing no regularization, L2,
   dropout, and combined L2+dropout (averaged over 3 seeds).
4. Demonstrates **early stopping** via validation loss on a label-noise overfitting scenario,
   plotting train/test accuracy and validation loss curves.
5. Shows **data augmentation** (rotation, noise, x-flip) on 20-sample `make_moons` training,
   comparing test accuracy over 3 seeds.

## Key Results

**1. L1 vs L2 on polynomial logistic regression (27 features, true boundary is linear):**

| Regime | Train acc | Test acc | $\|w\|$ | Nonzero weights ($|w|>0.01$) |
|---|---|---|---|---|
| None | 0.911 | 0.867 | 4.87 | 27 |
| L2 = 0.1 | 0.933 | 0.900 | 1.06 | 25 |
| L1 = 0.05 | 0.922 | **0.933** | 1.50 | **5** |

L1 reduces 27 features to **5 nonzero weights** — automatically discarding 22 irrelevant
polynomial terms — and achieves the best test accuracy.

**2. Inverted dropout validated:** mean output over 200 random masks = **1.0000** (ours) vs
**1.0000** (PyTorch `F.dropout`), confirming correct $1/(1-p)$ scaling.

**3. Deep MLP on 50 noisy labels** (2→128→128→128→1, 16% label flips, averaged over 3 seeds):

| Config | Noisy train | Clean train | Test | $\|W\|$ |
|---|---|---|---|---|
| None | 0.940 | 0.873 | 0.782 | 28.6 |
| L2 = 0.005 | 0.913 | 0.887 | 0.797 | 18.0 |
| Dropout p = 0.5 | 0.753 | 0.820 | 0.765 | 27.5 |
| L2 + Dropout | 0.773 | 0.813 | **0.793** | 16.8 |

The unregularized model reaches 94% on noisy labels but only 87% on clean train labels —
it memorized flipped labels. Dropout dramatically lowers noisy-train accuracy (75%) because
memorization becomes harder. L2+Dropout gives the best mean test accuracy (79.3%).

**4. Early stopping (validation loss, patience = 80):**

| Checkpoint | Test accuracy |
|---|---|
| Best val-loss epoch (210) | **0.840** |
| Full 2000 epochs | 0.790 |

Continuing to train after validation loss stopped improving **hurt** test accuracy by 5 points.

**5. Data augmentation (20 training samples, noise = 0.35, 3 seeds):**

| | Mean test accuracy |
|---|---|
| No augmentation | 0.747 |
| Rotation + noise + flip | **0.795** |

Unaugmented models hit **100% train accuracy** (perfect memorization of 20 points); augmentation
improves mean test accuracy by ~5 points.

## Common Pitfalls

1. **Forgetting to disable dropout at test time.** Training with dropout and evaluating without
   switching to `training=False` / `model.eval()` applies random noise at inference — predictions
   will differ run-to-run. Our inverted dropout needs no scaling at test time, but the mask must
   still be disabled.
2. **Treating L2 in the loss and weight decay in Adam as identical.** Topic 05 showed AdamW
   exists precisely because folding L2 into the gradient couples regularization strength to
   per-parameter gradient history. Use AdamW's decoupled `weight_decay` parameter, not
   `loss + lambda*||w||^2` with plain Adam, in modern training pipelines.
3. **Assuming dropout always helps.** On very small datasets (50 samples), dropout alone can
   *underfit* and hurt test accuracy (76.5% vs 78.2% baseline in our 3-seed average). It is
   most beneficial on larger networks and datasets where overfitting is the primary risk.
4. **Using training accuracy for early stopping.** Training accuracy monotonically increases
   (or plateaus at 100%) even as the model overfits. Always monitor **validation loss** or
   validation accuracy on a held-out set that the model never trains on.
5. **Data augmentation that changes the label.** Random rotation and noise preserve the moons
   class labels; arbitrary transforms (e.g. random horizontal flip on digit "6") may not.
   Every augmentation must be label-preserving for the specific task.
6. **Confusing regularization with fixing a bad model.** Regularization reduces overfitting
   but cannot fix an underpowered architecture, wrong loss function, or bad initialization.
   If the model underfits even without regularization, adding more will make things worse.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `train_logistic_l2`, `train_logistic_l1` | Polynomial logistic regression with L2/L1 penalty |
| `dropout_forward`, `dropout_backward` | Inverted dropout forward pass and backward mask |
| `MLPReg` | MLP with configurable L2 penalty and dropout on hidden layers |
| `MLPSimple` | Lightweight MLP for data-augmentation experiments |
| `augment_moons` | Random rotation, Gaussian noise, and optional x-flip for 2D data |

## Self-Test

1. Derive why inverted dropout scales by $1/(1-p)$ during training rather than by $(1-p)$
   at test time. What would happen to the expected output magnitude if you forgot the
   training-time scaling?
2. In the L1 polynomial experiment, why does L1 produce exactly zero weights while L2
   produces small but nonzero weights? What property of the L1 subgradient at $w_i=0$
   causes this?
3. Dropout with $p=0.5$ zeroed the example `[2,4,6,8]` to `[0,8,0,0]`. Why is the surviving
   unit's value 8 rather than 4?
4. In the early stopping experiment, why does validation *loss* (rather than accuracy) make
   a better stopping criterion on a small validation set?
5. If you have unlimited training data that already covers the full input distribution,
   will data augmentation still help? Why or why not?
6. The unregularized MLP scored 94% on noisy train labels but only 87% on clean train
   labels from the *same* 50 points. How is that possible?

## Further Reading

- Srivastava, N. et al. (2014). ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://jmlr.org/papers/v15/srivastava14a.html) — the original dropout paper.
- Tibshirani, R. (1996). ["Regression Shrinkage and Selection via the Lasso"](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) — L1 regularization and sparsity.
- Prechelt, L. (1998). ["Early Stopping — But When?"](https://doi.org/10.1007/3-540-49430-8_3) — practical guide to validation-based stopping criteria.
- Shorten, C. & Khoshgoftaar, T. (2019). ["A survey on Image Data Augmentation for Deep Learning"](https://doi.org/10.1186/s40537-019-0197-0) — augmentation strategies beyond the 2D demo here.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Chapter 7](https://www.deeplearningbook.org/contents/regularization.html) — comprehensive regularization overview.

---
[← Back to Deep Learning Foundations](../README.md)
