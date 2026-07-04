# Vanishing/Exploding Gradients & Gradient Clipping

> Status: ✅ Complete

Why deep networks are hard to train, implemented from scratch: we deliberately induce
vanishing and exploding gradients in deep MLPs, validate gradient clipping against
PyTorch, and show clipping rescuing training runs that otherwise diverge to `NaN`.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`16_vanishingexploding_gradients_gradient_clipping.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Backpropagation computes gradients by the chain rule, multiplying together one factor per
layer (or per timestep, for recurrent nets). When those factors are consistently **less
than 1**, the product shrinks geometrically with depth — the **vanishing gradient
problem**: early layers receive a gradient so small they effectively stop learning. When
the factors are consistently **greater than 1**, the product grows geometrically instead —
the **exploding gradient problem**: weight updates become enormous, activations overflow,
and the loss becomes `NaN` within a few steps.

Topic 06 showed this same mechanism from the **initialization** angle: mismatched
weight scale relative to the activation function makes forward-pass activations vanish or
explode with depth. Topic 10 showed it from the **time** angle: vanilla RNNs vanish across
long sequences because the same $W_{hh}$ is multiplied repeatedly during BPTT. This topic
studies the **depth** axis directly and adds the standard practical fix for the exploding
side: **gradient clipping**, which caps the gradient's magnitude before the optimizer step
without requiring you to re-derive a perfect initialization scheme.

## Mathematical Explanation

For an $L$-layer network, the gradient reaching layer $i$ involves a product of
$(L - i)$ terms:

$$\frac{\partial \mathcal{L}}{\partial h_i} = \frac{\partial \mathcal{L}}{\partial h_L}
\prod_{k=i+1}^{L} W_k^\top \odot g'(z_k)$$

If $\|W_k^\top g'(z_k)\| < 1$ on average, the product shrinks exponentially in $(L - i)$
— **vanishing**. If $\|W_k^\top g'(z_k)\| > 1$ on average, it grows exponentially —
**exploding**. Saturating activations (sigmoid, tanh) bound $g'(z) \in [0, 1]$ or
$[0, 0.25]$, which caps how large the product can get but not how *small* — vanishing is
the dominant risk. Non-saturating activations (ReLU) have $g'(z) \in \{0, 1\}$ with no
upper bound *from the activation itself*, so an overly large weight scale can drive
unbounded growth with depth.

**Gradient clipping by global norm** rescales the entire gradient vector (concatenated
across all parameters) if its norm exceeds a threshold, preserving direction:

$$g \leftarrow g \cdot \min\!\left(1,\ \frac{c}{\|g\|_2}\right)$$

**Gradient clipping by value** clips each element independently:

$$g_i \leftarrow \text{clip}(g_i,\ -c,\ c)$$

Value clipping is simpler but does not preserve the gradient's direction — different
parameters can be clipped by different relative amounts, effectively distorting the
descent direction, unlike norm clipping which scales everything uniformly.

## Code Implementation

All code lives in [`16_vanishingexploding_gradients_gradient_clipping.ipynb`](16_vanishingexploding_gradients_gradient_clipping.ipynb). It:

1. Implements a depth-configurable `DeepMLP` (tanh/sigmoid/ReLU) with full backprop and
   per-layer gradient-norm tracking.
2. Induces **vanishing gradients**: sigmoid activations, weight scale $0.5\times$ the usual
   $1/\sqrt{\text{fan\_in}}$, across depths 5–80.
3. Induces **exploding gradients**: ReLU activations, weight scale above the He-optimal
   value, sweeping both weight scale and depth.
4. Implements `clip_grad_norm` and `clip_grad_value` from scratch.
5. Validates `clip_grad_norm` against `torch.nn.utils.clip_grad_norm_`.
6. Trains the exploding-gradient configuration at four learning rates with no clipping,
   norm clipping, and value clipping — reporting accuracy or divergence.

## Key Results

**Vanishing gradients** (sigmoid, `weight_scale=0.5`, gradient norm after one backward pass):

| Depth | Layer-0 norm | Last-layer norm | Ratio (L0 / Llast) |
|---|---|---|---|
| 5 | 5.47e-05 | 1.35e-02 | 4.06e-03 |
| 10 | 8.29e-10 | 3.65e-01 | 2.27e-09 |
| 20 | 1.11e-18 | 3.16e-01 | 3.53e-18 |
| 40 | 7.72e-37 | 9.40e-02 | 8.21e-36 |
| 60 | 3.50e-55 | 2.83e-01 | 1.24e-54 |
| **80** | **5.48e-74** | **3.35e-01** | **1.63e-73** |

At depth 80, the input layer's gradient is **~73 orders of magnitude** smaller than the
last hidden layer's — it receives no usable learning signal at all.

**Exploding gradients** (ReLU, weight-scale sweep at depth 20):

| `weight_scale` | Loss | Total grad norm |
|---|---|---|
| 1.4 | 0.697 | 5.78e-01 |
| 2.0 | 13.40 | 1.17e+03 |
| 3.0 | 13.82 | 2.58e+06 |
| **4.0** | 13.82 | **6.11e+08** |

Going from `weight_scale=1.4` to `4.0` — roughly a **3× change** — grows the total
gradient norm by **9 orders of magnitude**.

**Exploding gradients vs depth** (ReLU, `weight_scale=2.0`):

| Depth | Total grad norm |
|---|---|
| 10 | 3.04e+01 |
| 15 | 1.56e+02 |
| 20 | 1.17e+03 |
| 25 | 6.82e+03 |
| **30** | **7.73e+03** |

**PyTorch validation** (random gradients, `max_norm=1.0`):

| | Value |
|---|---|
| Total norm (numpy / pytorch) | 200.770030 / 200.770030 |
| Max diff after clipping (weights / biases) | 1.39e-17 / 1.39e-17 |

**Does clipping rescue divergent training?** (depth 20, ReLU, `weight_scale=2.0`, 150 epochs):

| lr | No clipping | Norm clip (max_norm=5.0) | Value clip (clip_val=0.5) |
|---|---|---|---|
| 0.001 | acc=0.880 | acc=0.860 | acc=0.910 |
| 0.005 | **DIVERGED** | acc=0.865 | acc=0.940 |
| 0.01 | **DIVERGED** | acc=0.915 | acc=0.915 |
| 0.02 | **DIVERGED** | acc=0.925 | acc=0.940 |

Without clipping, every learning rate $\geq 0.005$ diverges to `NaN` within a handful of
steps. Both clipping strategies rescue **every** learning rate tested, reaching
86–94% training accuracy instead of a crashed run.

## Common Pitfalls

1. **Treating vanishing and exploding gradients as unrelated problems.** Both come from
   the same multiplicative chain-rule mechanism (Topics 06, 10) — the only difference is
   whether the per-layer factor is consistently below or above 1.
2. **Expecting gradient clipping to fix vanishing gradients.** Clipping only caps large
   gradients from above; it does nothing for gradients that are already near zero. Use
   better initialization (Topic 06), normalization (Topic 08), gated architectures
   (Topic 11), or residual/skip connections instead.
3. **Clipping per-tensor instead of globally.** Clipping each weight matrix's gradient
   independently (rather than computing one norm over *all* parameters) changes the
   relative scaling between layers and is not equivalent to the standard recipe used by
   `torch.nn.utils.clip_grad_norm_`.
4. **Choosing a clip threshold with no regard to typical gradient scale.** Section 6 used
   `max_norm=5.0`; the too-aggressive `max_norm=1.0` (initial guess) noticeably hurt
   convergence at low learning rates because it clipped even well-behaved gradients.
   Inspect typical unclipped gradient norms before picking a threshold.
5. **Assuming ReLU is immune to instability because it "fixes vanishing gradients."**
   ReLU indeed removes the *saturating* half of the vanishing-gradient story, but its
   unbounded derivative on the active side makes it *more* prone to exploding gradients
   under a too-large weight scale (Section 3) — the opposite failure mode.
6. **Using value clipping and expecting the same trajectory as norm clipping.** Value
   clipping changes the gradient's *direction* (different components can be clipped by
   different relative amounts); norm clipping only changes its *magnitude*. They can both
   rescue divergence but will generally reach different final weights.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `DeepMLP` | Depth-configurable MLP (tanh/sigmoid/ReLU) with per-layer gradient tracking |
| `global_grad_norm` | $\ell_2$ norm of the gradient concatenated across all parameters |
| `clip_grad_norm` | Rescales all gradients by a single factor if global norm exceeds `max_norm` |
| `clip_grad_value` | Element-wise clip of every gradient to $[-c, c]$ |
| `train_run` | Trains `DeepMLP`, returns `(loss, accuracy)` or `None` on divergence |

## Self-Test

1. Write the chain-rule product for the gradient reaching layer $i$ in an $L$-layer
   network. Under what condition on the per-layer factors does it vanish vs explode?
2. Why does sigmoid's bounded derivative ($\leq 0.25$) make **vanishing** the dominant
   risk, while ReLU's unbounded derivative on the active side makes **exploding** more
   likely under a too-large weight scale?
3. At depth 80 with `weight_scale=0.5` (sigmoid), the layer-0 gradient is ~$10^{-74}$.
   Would increasing the learning rate fix this? Why or why not?
4. Our norm-clipping validation shows a max element-wise diff of ~1.39e-17 vs PyTorch.
   Is this difference numerically meaningful, or expected floating-point noise?
5. Section 6 shows norm clipping with `max_norm=5.0` outperforms `max_norm=1.0` at low
   learning rates. Why can a clip threshold that is *too small* hurt training even when
   gradients aren't exploding?
6. Name two fixes for vanishing gradients that do **not** involve gradient clipping, and
   explain why clipping alone cannot solve that problem.

## Further Reading

- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). ["On the difficulty of training recurrent
  neural networks"](https://arxiv.org/abs/1211.5063) — the paper that introduced gradient
  norm clipping.
- Bengio, Y. et al. (1994). ["Learning Long-Term Dependencies with Gradient Descent is
  Difficult"](https://ieeexplore.ieee.org/document/6796450) — foundational analysis (see
  also Topic 10).
- [PyTorch `torch.nn.utils.clip_grad_norm_` documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- Topic 06 — Weight Initialization (the forward-pass side of this same mechanism)
- Topic 10 — RNN Basics (vanishing gradients across time instead of depth)
- Topic 08 — Batch/Layer Normalization (an alternative, complementary fix)

---
[← Back to Deep Learning Foundations](../README.md)
