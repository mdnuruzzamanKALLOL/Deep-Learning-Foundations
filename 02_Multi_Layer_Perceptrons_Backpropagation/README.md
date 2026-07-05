# Multi-Layer Perceptrons & Backpropagation

> Status: ✅ Complete

The core algorithm every deep network is trained with

## Concept & Intuition

Topic 01 ended on a genuine failure: a single perceptron cannot learn XOR, because it can only draw one straight line through feature space. A **Multi-Layer Perceptron (MLP)** fixes this by stacking layers of perceptron-like units with a nonlinearity in between: the hidden layer first bends the input space, then the output layer draws a single line through the *bent* space — which can look like an arbitrarily curved boundary back in the original input space. Given enough hidden units, an MLP can approximate essentially any continuous function (the Universal Approximation Theorem, Cybenko 1989 / Hornik 1991).

The catch is training it. A single perceptron's error-driven rule has no way to assign blame to a hidden unit that never sees the true label directly. **Backpropagation** (Rumelhart, Hinton & Williams, 1986) solves this credit-assignment problem using the chain rule to propagate the output error backward through every layer, computing the exact gradient of the loss with respect to every weight — the algorithm every deep network, from this 2-layer MLP to a modern Transformer, is still trained with today.

## Mathematical Explanation

**Forward pass**, layer $l=1,\dots,L$: $\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$, $\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$, with $\mathbf{a}^{[0]}=\mathbf{x}$.

**Loss**: mean squared error $\mathcal{L} = \frac{1}{n}\sum_i \frac{1}{m}\|\mathbf{a}^{[L]}_i - \mathbf{y}_i\|^2$ over $n$ examples, $m$ outputs.

**Backward pass.** Define the error signal $\boldsymbol{\delta}^{[l]} = \partial\mathcal{L}/\partial\mathbf{z}^{[l]}$. At the output layer:

$$\boldsymbol{\delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \odot g'(\mathbf{z}^{[L]})$$

propagated backward through each preceding layer via:

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1]\top}\boldsymbol{\delta}^{[l+1]}\right) \odot g'(\mathbf{z}^{[l]})$$

with weight gradients falling out directly:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]}\mathbf{a}^{[l-1]\top}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]}$$

This is the multivariate chain rule applied systematically — turning an otherwise combinatorially expensive gradient computation into one forward pass plus one backward pass.

## Code Implementation (`02_multi_layer_perceptrons_backpropagation.ipynb`)

### 1. Gradient checking

The from-scratch analytical gradients are compared against numerical finite-difference gradients ($\epsilon=10^{-5}$) on a 2-4-1 network:

| Parameter | Max abs diff | Max relative error |
|---|---|---|
| Layer 0 weights | 3.43e-12 | 2.80e-08 |
| Layer 1 weights | 1.80e-12 | 2.24e-11 |
| Layer 0 bias | 3.70e-12 | 1.98e-09 |
| Layer 1 bias | 4.79e-12 | 3.86e-11 |

All well below the $10^{-7}$ threshold conventionally used to trust a backprop implementation.

### 2. Validation against PyTorch's autograd

The identical network (same weights, same input) is built in PyTorch (float64) and its `.backward()` gradients compared directly: **loss matches to 0.00e+00, and every gradient matches to within $10^{-18}$** — pure floating-point noise. The from-scratch backward pass computes exactly what a production autodiff engine computes.

### 3. XOR, solved — the redemption from Topic 01

A 2-4-1 MLP trained for 5,000 epochs reaches **100% accuracy** on XOR (loss 0.00068), with predictions [0.028, 0.976, 0.974, 0.026] rounding perfectly to the true labels [0, 1, 1, 0] — the exact problem a single perceptron could never converge on.

### 4. How much capacity does XOR actually need?

Each hidden-layer size tested across 10 random initializations:

| Hidden units | Mean accuracy | Notes |
|---|---|---|
| 1 | 0.70 | Capped regardless of seed — fundamentally lacks the capacity |
| 2 | 0.95 | 9/10 seeds reach 100%; **1 seed stuck at 50%** |
| 3 | 1.00 | Every seed succeeds |
| 4 | 1.00 | Every seed succeeds |
| 8 | 1.00 | Every seed succeeds |

**1 hidden unit cannot represent XOR at all**, capping at ~70% no matter how long it trains. **2 hidden units is the theoretical minimum that *can* represent XOR**, but one initialization (seed 8) genuinely fails: its loss goes from 0.1258 at epoch 5,000 to only 0.1252 at epoch 20,000 — a confirmed plateau, not slow ongoing convergence — with two of the four predictions permanently stuck at exactly 0.50. **3+ hidden units solve it reliably every time tested**, since extra capacity provides redundant paths around this kind of bad local minimum.

### 5. Real dataset: `make_moons`

Two interleaving crescents — not linearly separable by construction, a real-data analog of XOR:

| Model | Train accuracy | Test accuracy |
|---|---|---|
| Single perceptron | 0.8762 | 0.8667 |
| MLP (8 hidden units) | 0.9905 | 0.9556 |

The perceptron is mathematically capped at whatever a single straight line can achieve through two interleaving crescents; the MLP bends its decision boundary to actually follow their shape, closing most of the gap a linear model cannot close.

## Common Pitfalls

1. **Capacity and optimization are two separate problems.** 1 hidden unit fails because it *cannot represent* XOR (a capacity limit); 2 hidden units can represent it but sometimes *fails to find* the solution (an optimization/local-minimum problem) — these look identical (both give <100% accuracy) but need completely different fixes.
2. **A stuck loss plateau can look like slow convergence if you don't check it directly.** The seed-8 failure only becomes obviously a genuine local minimum once you compare the loss at widely separated epochs (5,000 vs. 20,000) — a short training run alone could be mistaken for "still improving."
3. **Always gradient-check a new backprop implementation before trusting it for real training.** A subtly wrong derivative (e.g., a missing term in the chain rule) can still let a network sort of learn, making the bug easy to miss without an explicit numerical check.
4. **Sigmoid activations were used here for historical/pedagogical continuity with Topic 01**, but they have real, serious drawbacks (vanishing gradients, non-zero-centered outputs) that Topic 03 covers — don't take this notebook's choice of activation as a general recommendation.
5. **Extra hidden units aren't "wasted" even when they exceed the minimum needed to represent a function.** Section 4 showed 3+ units solve XOR reliably specifically *because* of the redundancy, not despite it — overparameterization can be a legitimate defense against bad local minima.
6. **A single random seed is not evidence of a general result.** The seed-8 local-minimum failure would have been invisible testing only 1-2 seeds; the capacity/reliability conclusions in Section 4 only hold because 10 different initializations were tried per configuration.

## Function Reference

| Function | Purpose |
|---|---|
| `MLP` (from scratch) | Generic feedforward network: forward pass, backward pass, and a full gradient-descent training step |
| `gradient_check` (from scratch) | Numerical finite-difference validation of every analytical gradient |
| PyTorch `.backward()` | Reference autodiff gradients used for the machine-precision cross-check |
| `sklearn.linear_model.Perceptron` | Linear baseline used to show the real capacity gap on `make_moons` |
| `sklearn.datasets.make_moons` | Real, genuinely non-linearly-separable 2D dataset |

## Self-Test

1. Why does backpropagation's $\boldsymbol{\delta}^{[l]}$ term need to be multiplied by $g'(\mathbf{z}^{[l]})$ at every layer, not just the output layer?
2. What would you expect to see in a gradient check if the backward pass had a sign error in the weight update? What if it had the correct sign but a missing factor of 2?
3. Why does 1 hidden unit fail to represent XOR at all, while 2 hidden units can represent it (even though one particular initialization still failed to find that representation)?
4. What distinguishes a genuine local-minimum plateau (Section 4's seed-8 case) from a network that is merely converging slowly? How would you check the difference in your own training runs?
5. Why does adding more hidden units than the strict minimum needed tend to make training more reliable, even though it doesn't change what the network is theoretically capable of representing?
6. In the `make_moons` comparison, why is the MLP's decision boundary curved instead of another straight line, given that the output layer is still just a linear-plus-sigmoid unit?

## Notebook

[`02_multi_layer_perceptrons_backpropagation.ipynb`](./02_multi_layer_perceptrons_backpropagation.ipynb)

## Further Reading

- Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). *Learning Representations by Back-Propagating Errors.* Nature, 323, 533-536. (The paper that popularized backpropagation.)
- Cybenko, G. (1989). *Approximation by Superpositions of a Sigmoidal Function.* Mathematics of Control, Signals and Systems, 2(4), 303-314. (Universal Approximation Theorem.)
- Hornik, K. (1991). *Approximation Capabilities of Multilayer Feedforward Networks.* Neural Networks, 4(2), 251-257.
- Blum, E.K. (1989). *Approximation of Boolean Functions by Sigmoidal Networks: Part I: XOR and Other Two-Variable Functions.* Neural Computation, 1(4), 532-540. (Analyzes exactly the XOR local-minimum behavior seen in Section 4.)
- [CS231n: Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.02_Multi_Layer_Perceptrons_Backpropagation&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
