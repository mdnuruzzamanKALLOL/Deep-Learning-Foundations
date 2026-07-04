# Weight Initialization

> Status: ✅ Complete

Xavier/Glorot and He initialization — implemented from scratch, validated against PyTorch,
and stress-tested to show mechanistically why the scale of random initial weights determines
whether a deep network can train at all.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`06_weight_initialization.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Before an optimizer takes a single step or a loss function computes a single gradient, the
network's weights already determine whether training is even possible. Initialize every
weight to zero and every hidden unit in a layer remains a perfect clone of its neighbors —
symmetry is never broken, and the network's effective capacity collapses to a single unit
regardless of how many neurons you allocated. Initialize with `std=1.0` (a common legacy
default) in a deep ReLU network and activations explode exponentially with depth, producing
`inf` losses before the first useful gradient arrives.

Xavier (Glorot) and He (Kaiming) initialization solve a precise engineering problem: pick
the variance of random weights so that signal variance stays roughly constant as it
propagates forward through layers *and* as gradients propagate backward. Xavier targets
sigmoid/tanh (roughly linear near zero); He targets ReLU (which zeroes half the activations
on average, requiring a compensating factor of 2 in the variance formula).

Topic 03 showed ReLU appearing to suffer vanishing gradients under "Xavier-scale" init —
this topic explains that was an initialization mismatch, not a ReLU defect, and quantifies
exactly how large the mismatch penalty is.

## Mathematical Explanation

For a layer with input $x_j$, weight $W_{ij}$, and activation $g$:

$$\text{Var}(z_i) \approx \text{fan\_in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

**Xavier / Glorot (2010)** — balance forward and backward variance for sigmoid/tanh:
$$\text{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$$

For square layers ($\text{fan\_in}=\text{fan\_out}=n$): $\text{std}(W)=\sqrt{2/(2n)}=\sqrt{1/n}$.

**He / Kaiming (2015)** — ReLU zeroes half the activations, halving output variance; compensate:
$$\text{Var}(W) = \frac{2}{\text{fan\_in}}$$

For width $n$: $\text{std}(W)=\sqrt{2/n}$ — exactly $\sqrt{2}$ times larger than Xavier's
$\sqrt{1/n}$ for square layers.

**Biases** are almost always initialized to zero. Unlike weights, zero biases do not create
symmetry problems (each unit's bias is independent).

## Code Implementation

All code lives in [`06_weight_initialization.ipynb`](06_weight_initialization.ipynb). It:

1. Implements Xavier, He, Normal, Zero, and Constant initialization from scratch.
2. Validates Xavier and He against `nn.init.xavier_normal_` and `nn.init.kaiming_normal_`
   (accounting for PyTorch's transposed weight layout).
3. Tracks **forward activation variance** through 10 identical layers (width 256) under
   matched and mismatched init/activation pairings, plus catastrophic `std=1.0` and zero init.
4. Measures **backward gradient norms** reaching each layer under the same conditions.
5. Demonstrates **symmetry breaking failure** with all-zero weights on a 2-hidden-unit network.
6. Trains a **4-hidden-layer ReLU MLP** on `make_moons` under four init schemes, comparing
   convergence speed (epochs to 90% accuracy) over 5 random seeds for Xavier vs He.

## Key Results

**PyTorch validation** (shape `(128, 64)` in our `(fan_in, fan_out)` convention):

| Scheme | Our std | PyTorch std (on $W^T$) |
|---|---|---|
| Xavier | 0.100770 | 0.101456 |
| He | 0.123417 | 0.125049 |

For square layers of width 256: Xavier std $=0.0625$, He std $=0.0884$, Normal std $=1.0$
(**16× larger than Xavier**).

**1. Forward variance through 10 layers (width 256, input var $\approx 1$):**

| Scheme | Layer-0 var | Layer-10 var | Ratio L10/L0 |
|---|---|---|---|
| Xavier + tanh (matched) | 0.996 | 0.052 | 0.052 |
| He + ReLU (matched) | 0.996 | 0.166 | 0.166 |
| Xavier + ReLU (mismatch) | 0.996 | 0.0002 | **$1.6\times 10^{-4}$** |
| Normal std=1.0 + ReLU | 0.996 | $1.95\times 10^{20}$ | **explosion** |
| Zeros + ReLU | 0.996 | 0.000 | 0 |

Xavier+ReLU collapses activations to near zero by layer 10; Normal std=1.0 explodes to
$10^{20}$ — both make deep training impossible before optimization even starts.

**2. Backward gradient norms (layer 1 vs layer 10):**

| Scheme | Layer-1 norm | Layer-10 norm | L1/L10 |
|---|---|---|---|
| Xavier + tanh (matched) | 5.27 | 14.77 | 0.36 |
| He + ReLU (matched) | 38.89 | 18.31 | 2.12 |
| Xavier + ReLU (mismatch) | 1.22 | 12.95 | **0.094** |

Under Xavier+ReLU mismatch, the gradient reaching layer 1 is **~10× smaller** than at
layer 10 — early layers receive almost no learning signal.

**3. Zero initialization breaks symmetry permanently.** After 100 gradient steps from
all-zero weights and biases on a 2-hidden-unit network: $W_1$ remains all zeros, both hidden
units are bitwise identical clones, effective capacity $=1$ unit.

**4. Deep MLP training on `make_moons`** (architecture 2→64→64→64→64→1, ReLU, lr=0.05):

| Init | Epochs to 90% acc | Final train acc | Final test acc |
|---|---|---|---|
| Zeros | Never | 0.538 | 0.387 |
| Normal std=1.0 | Diverges | — | — |
| Xavier (mismatch for ReLU) | 298 | 0.982 | 1.000 |
| He (matched for ReLU) | 192 | 0.991 | 1.000 |

Multi-seed average (5 seeds): Xavier **279 epochs**, He **219 epochs** (~**27% faster** with
matched initialization). Zeros never learns; std=1.0 diverges entirely on this depth.

## Common Pitfalls

1. **Initializing all weights to zero.** This is not "a slow start" — it permanently caps
   effective network capacity at one hidden unit per layer because symmetry is never broken.
   Always use random initialization for weights (biases can be zero).
2. **Using Xavier init with ReLU networks.** Xavier's variance formula assumes activations
   are roughly zero-centered and linear near the origin — true for tanh, false for ReLU
   (which is one-sided). The result is activations and gradients that vanish with depth
   (Section 3–4). Use He init for ReLU/Leaky ReLU; use Xavier for sigmoid/tanh.
3. **Using He init with sigmoid/tanh.** He's $\sqrt{2}$ factor is too large for saturating
   activations, pushing initial pre-activations into saturation regions where gradients are
   tiny. Match the init scheme to the activation function.
4. **Assuming `std=1.0` is a safe default.** For a layer of width 256, std=1.0 is **16×
   larger** than Xavier's recommended scale — deep ReLU nets explode immediately (Section 3).
   Framework defaults (PyTorch's `nn.Linear` uses Kaiming uniform by default since ~2017)
   exist for a reason; don't override them without understanding the scale.
5. **Confusing an activation-function problem with an initialization-scale mismatch.** Topic
   03's "ReLU vanishing gradient" under Xavier-scale init was entirely an initialization
   artifact — switching to He init restored stable gradient flow across 20 layers without
   changing the activation function at all.
6. **Ignoring fan_in vs fan_out in the formula.** Xavier uses $(\text{fan\_in}+\text{fan\_out})$
   in the denominator; He uses only fan_in. For very rectangular layers (e.g. a wide
   embedding projection), the two formulas give noticeably different scales — always pass
   the correct fan counts, not just the layer width.

## Function Reference

| Function | Purpose |
|---|---|
| `init_xavier(shape, rng)` | Xavier/Glorot normal init: std $=\sqrt{2/(\text{fan\_in}+\text{fan\_out})}$ |
| `init_he(shape, rng)` | He/Kaiming normal init: std $=\sqrt{2/\text{fan\_in}}$ |
| `init_normal(std)` | Factory for Gaussian init with arbitrary std |
| `init_zeros` | All-zero init (for demonstrating symmetry failure) |
| `init_constant(val)` | Constant init (useful for controlled experiments) |
| `forward_variance(...)` | Tracks mean activation variance through $L$ identical layers |
| `backward_grad_norms(...)` | Measures gradient norm reaching each layer on one backward pass |
| `DeepMLP` | Configurable-depth ReLU MLP for training comparisons on `make_moons` |

## Self-Test

1. For a square layer of width $n=256$, compute Xavier's and He's recommended $\text{std}(W)$
   by hand. How do they relate to each other?
2. ReLU zeroes approximately half the pre-activations. Explain intuitively why He's formula
   has a factor of 2 in the numerator that Xavier's does not.
3. Why can biases be safely initialized to zero while weights cannot? What would happen if
   you initialized all biases to the same nonzero constant but kept weights random?
4. In Section 3, Xavier+ReLU's layer-10 variance ratio is $1.6\times 10^{-4}$. If you added
   a 20th layer with the same setup, would you expect the ratio to be closer to 0 or to 1?
   Why?
5. Section 4 shows Xavier+ReLU gives layer-1 gradients ~10× smaller than layer-10. How would
   this affect which layers learn fastest during early training?
6. PyTorch's `nn.Linear` default initialization changed from uniform $[-1/\sqrt{\text{fan\_in}},\ +1/\sqrt{\text{fan\_in}}]$ to Kaiming uniform. What activation function was the old default implicitly assuming, and why was the change made?

## Further Reading

- Glorot, X. & Bengio, Y. (2010). ["Understanding the difficulty of training deep feedforward neural networks"](http://proceedings.mlr.press/v9/glorot10a.html) — the original Xavier initialization paper.
- He, K. et al. (2015). ["Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"](https://arxiv.org/abs/1502.01852) — introduces He/Kaiming initialization for ReLU networks.
- [PyTorch nn.init documentation](https://pytorch.org/docs/stable/nn.init.html) — exact formulas for `xavier_normal_`, `kaiming_normal_`, and current `Linear` defaults.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Chapter 8.5](https://www.deeplearningbook.org/contents/mlp.html) — guidelines for weight initialization and the exploding/vanishing signal problem.

---
[← Back to Deep Learning Foundations](../README.md)
