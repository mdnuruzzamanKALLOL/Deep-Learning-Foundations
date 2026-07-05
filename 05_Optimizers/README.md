# Optimizers

> Status: ✅ Complete

SGD, Momentum, RMSProp, Adam, AdamW — implemented from scratch, validated against PyTorch
to machine precision, and stress-tested with the three classic scenarios that motivated
each one historically.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`05_optimizers.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Every optimizer covered here was invented to fix a specific, reproducible failure mode of
the one before it:

1. **Plain SGD** oscillates badly in "ravine"-shaped loss landscapes (very different
   curvature along different directions), forcing a learning rate small enough for the
   steep direction, which makes the shallow direction painfully slow.
2. **Momentum** fixes the ravine problem by accumulating velocity in consistently-signed
   directions, effectively canceling out the oscillation.
3. **RMSProp** fixes a different problem: when different parameters have very different
   gradient *magnitudes* (not just curvature), a single learning rate can't serve both well.
   Dividing each parameter's update by its own running RMS gradient equalizes this.
4. **Adam** combines momentum and RMSProp's per-parameter normalization into one optimizer,
   with bias correction for the early steps when both running averages start at zero.
5. **AdamW** fixes a subtle bug in how weight decay had been bolted onto Adam: decoupling it
   from the adaptive per-parameter scaling restores weight decay's intended, uniform
   regularization behavior.

## Mathematical Explanation

**SGD**: $\theta \leftarrow \theta - \eta \nabla L$

**SGD + Momentum**:
$$v \leftarrow \mu v + \nabla L, \qquad \theta \leftarrow \theta - \eta v$$
Velocity $v$ is an exponentially-decaying sum of all past gradients — directions that keep
pointing the same way accumulate speed; directions that keep flipping sign cancel out.

**Nesterov Momentum**:
$$v \leftarrow \mu v + \nabla L, \qquad \theta \leftarrow \theta - \eta(\nabla L + \mu v)$$
A "look-ahead" correction: conceptually evaluates the gradient at $\theta+\mu v$ rather than
$\theta$, correcting momentum's overshoot slightly earlier.

**RMSProp**:
$$s \leftarrow \beta s + (1-\beta)(\nabla L)^2, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s}+\epsilon}\nabla L$$
$s$ is a running average of *squared* gradients per parameter. Dividing by $\sqrt{s}$ means
a parameter that has been getting large gradients takes proportionally smaller steps, and
vice versa — the effective step size self-normalizes per parameter.

**Adam**:
$$m \leftarrow \beta_1 m + (1-\beta_1)\nabla L, \qquad v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2$$
$$\hat m = \frac{m}{1-\beta_1^t}, \qquad \hat v = \frac{v}{1-\beta_2^t}, \qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat v}+\epsilon}\hat m$$
Momentum's $m$ and RMSProp's $v$ combined. Bias correction ($/(1-\beta^t)$) compensates for
$m,v$ being initialized to zero, which would otherwise bias early steps toward zero.

**AdamW**:
$$\theta \leftarrow \theta - \eta\left(\frac{\hat m}{\sqrt{\hat v}+\epsilon} + \lambda\theta\right)$$
Identical to Adam except the weight-decay term $\lambda\theta$ is added directly to the
parameter update, *not* passed through the $m,v$ moment estimates.

## Code Implementation

All code lives in [`05_optimizers.ipynb`](05_optimizers.ipynb). It:

1. Implements SGD, SGD+Momentum, SGD+Nesterov, RMSProp, Adam, and AdamW from scratch as
   small stateful classes with a `.step(params, grads)` interface.
2. Validates every optimizer against PyTorch's built-in equivalent on a toy quadratic in
   float64 — all match to machine precision.
3. Reproduces the **ravine problem**: minimizes $f(x,y)=x^2+100y^2$ from the same starting
   point with the same learning rate, comparing SGD, Momentum, and Nesterov.
4. Reproduces the **gradient-scale-disparity problem**: fits a 2-feature linear regression
   where one feature's scale is 100× the other's, comparing SGD's forced trade-off against
   Adam's per-parameter normalization.
5. Runs a **learning-rate robustness sweep**: trains a small MLP on `make_moons` with each
   optimizer across 12 learning rates spanning 6 orders of magnitude.
6. Reproduces the **AdamW paper's core result**: simulates two parameters with different
   gradient histories, then isolates weight decay's effect and compares "Adam + L2" against
   AdamW's decoupled decay.

## Key Results

**All six optimizers validated against PyTorch to machine precision** (float64, 20 steps
on $f(x,y)=x^2+10y^2$): max parameter-trajectory difference was exactly `0.0` for SGD,
Momentum, Nesterov, and RMSProp, and `~1e-15` for Adam and AdamW (floating-point roundoff).

**1. The ravine problem.** Minimizing $f(x,y)=x^2+100y^2$ from $(8,1)$ with a shared,
safely-stable learning rate ($\eta=0.003$):

| Optimizer | Steps to reach loss < $10^{-4}$ |
|---|---|
| SGD | 1111 |
| Momentum | 90 |
| Nesterov | 92 |

Momentum and Nesterov are **~12× faster** than plain SGD at the identical learning rate,
purely from accumulating velocity along the shallow, consistently-signed direction.

**2. The gradient-scale-disparity problem.** Fitting $\hat y=w_1x_1+w_2x_2$ with
$\text{scale}(x_2) = 100\times\text{scale}(x_1)$ (true $w_1=2.0$, $w_2=0.03$):

| Optimizer | Learning rate | $w_1$ (true 2.0) | $w_2$ (true 0.03) |
|---|---|---|---|
| SGD (largest stable lr) | 0.0001 | 0.115 | 0.029 |
| Adam | 0.5 (5000× larger) | 1.977 | 0.030 |

SGD's learning rate is capped by $w_2$'s large gradients, leaving $w_1$ badly underfit.
Adam, using a single learning rate **5000× larger**, recovers both weights accurately
because it normalizes each parameter's step by its own gradient history.

**3. Learning-rate robustness on an actual MLP.** Training a 2→8→1 ReLU network on
`make_moons` across 12 learning rates (0.0003 to 100):

| Optimizer | LRs achieving ≥85% accuracy (out of 12) |
|---|---|
| Momentum | 7 |
| SGD | 6 |
| Adam | 6 |
| RMSProp | 5 |

Momentum was the most learning-rate-robust optimizer on this particular task — a useful,
honest counterexample to the common assumption that adaptive methods are always the most
forgiving of a poorly-chosen learning rate. RMSProp's aggressive per-parameter
normalization made it *more* prone to overshooting once the learning rate got large.

**4. AdamW vs. "Adam + L2" (the core AdamW-paper result).** Two parameters, $w_a$ (large
task-gradient history) and $w_b$ (small task-gradient history), both starting at magnitude
1.0. After building up realistic $m,v$ history, the task gradient is switched off and only
weight decay acts:

| Optimizer | $w_a$ shrinkage | $w_b$ shrinkage |
|---|---|---|
| Adam + L2-via-gradient | 3.2% | **97.1%** |
| AdamW (decoupled) | 10.5% | 10.5% |

Under naive "Adam + L2", the regularization strength actually applied depends heavily on
each parameter's gradient history — $w_b$ gets shrunk 30× more aggressively than $w_a$ for
the *same* nominal weight-decay coefficient. AdamW applies **identical** proportional decay
to both, exactly matching what "weight decay" is supposed to mean.

## Common Pitfalls

1. **Assuming Adam is always the most learning-rate-robust choice.** Our own sweep shows
   plain Momentum matching or beating Adam's robustness on a simple MLP task. Always verify
   empirically for your specific problem rather than defaulting on reputation alone.
2. **Using "Adam + L2 regularization" (`weight_decay` folded into the gradient) and expecting
   it to behave like textbook weight decay.** It doesn't — the decay strength gets scaled by
   each parameter's own second-moment estimate, producing wildly uneven regularization. Use
   AdamW's decoupled decay instead (in PyTorch: `torch.optim.AdamW`, not
   `torch.optim.Adam(weight_decay=...)`).
3. **Forgetting Adam's bias correction.** Without the $1/(1-\beta^t)$ correction terms, the
   first several steps have artificially small effective updates because $m,v$ start at
   zero and $\beta_1,\beta_2$ are close to 1 — this can look like a "warm-up is broken" bug
   if you re-implement Adam from a paper and skip this detail.
4. **Choosing a learning rate for momentum-based methods the same way you would for plain
   SGD.** Momentum's effective per-step displacement is roughly $\eta/(1-\mu)$ once velocity
   has built up — a learning rate perfectly stable for SGD can be unstable for Momentum or
   Nesterov at the same nominal value if $\mu$ is high (0.9+).
5. **Believing adaptive methods (RMSProp/Adam) are strictly more robust to large learning
   rates.** In the sweep above, RMSProp degraded to near-random accuracy at $\eta=1$ while
   plain SGD was still near its best performance — per-parameter normalization does not
   automatically confer robustness to a poorly-scaled global learning rate.
6. **Not re-tuning the learning rate when switching optimizers.** SGD, Momentum, and
   Adam-family optimizers have different "natural" learning rate scales (SGD often wants
   0.01–1, Adam often wants 0.0001–0.01 for deep networks) — reusing the same value across
   optimizer switches is a common source of "optimizer X doesn't work" bug reports that are
   actually learning-rate mismatches.

## Function Reference

| Function/Class | Purpose |
|---|---|
| `SGD` | Plain gradient descent |
| `SGDMomentum` | Momentum, with optional Nesterov look-ahead |
| `RMSProp` | Per-parameter squared-gradient normalization |
| `Adam` | Momentum + RMSProp with bias correction |
| `AdamW` | Adam with decoupled weight decay |
| `AdamL2` | Adam with weight decay folded into the gradient (the "naive"/pre-AdamW approach), for comparison only |
| `run_traj` / `steps_to_converge` | Trajectory and convergence-speed helpers for the ravine experiment |
| `run_linreg` | Fits the two-feature linear regression used in the gradient-scale-disparity demo |
| `MLP` | Small ReLU network (2→8→1) used for the learning-rate robustness sweep |
| `run_decay_only_phase` | Isolates weight decay's effect after a warm-up phase with realistic gradient history |

## Self-Test

1. Derive why, for a quadratic $f(x)=\frac{1}{2}ax^2$, plain gradient descent requires
   $\eta < 2/a$ for stability. How does this bound explain why a shared learning rate is
   forced to be small on the steep axis of a ravine?
2. Walk through Adam's bias-correction terms at $t=1$: if $m_0=v_0=0$ and $\beta_1=0.9$,
   what is $\hat m_1$ in terms of $\nabla L_1$? Why would skipping this correction make the
   first update artificially small?
3. In the gradient-scale-disparity experiment, why couldn't we simply use two *different*
   learning rates for $w_1$ and $w_2$ under plain SGD to fix the problem manually? What
   would change if the loss surface had many more than two parameters?
4. Explain in your own words why folding weight decay into the gradient (`Adam + L2`) causes
   parameters with large gradient history to be *under*-regularized relative to parameters
   with small gradient history.
5. Nesterov's update evaluates the gradient at a "look-ahead" point. Sketch what this
   look-ahead point would be after several updates with $\mu=0.9$ if the actual gradient
   has become nearly zero (i.e. near a minimum) — does the look-ahead help or hurt here?
6. If you were handed a completely new loss landscape with unknown curvature and unknown
   per-parameter gradient scale, which optimizer would you try first, and what evidence from
   this notebook would you cite to justify (or temper) that choice?

## Further Reading

- Kingma, D. P. & Ba, J. (2015). ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) — the original Adam paper.
- Loshchilov, I. & Hutter, F. (2019). ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) — the AdamW paper; Section 2 walks through exactly the coupling bug reproduced here.
- Sutskever, I. et al. (2013). ["On the importance of initialization and momentum in deep learning"](https://proceedings.mlr.press/v28/sutskever13.html) — Nesterov momentum applied to deep network training.
- Tieleman, T. & Hinton, G. (2012). [RMSProp, Lecture 6.5, Coursera "Neural Networks for Machine Learning"](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf) — the original (unpublished) RMSProp source.
- [PyTorch optimizer documentation](https://pytorch.org/docs/stable/optim.html) — exact update-rule formulas for each optimizer, useful for cross-checking a from-scratch implementation.

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.05_Optimizers&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
