# Activation Functions

> Status: ✅ Complete

Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish

## Concept & Intuition

Topic 02 built a working MLP using sigmoid without ever asking whether sigmoid was a good choice. It wasn't. Without a nonlinearity between layers, stacking layers is pointless: $\mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x}+\mathbf{b}^{[1]})+\mathbf{b}^{[2]}$ algebraically collapses to a single linear layer — the nonlinearity $g$ is what gives depth any representational power at all. But not all nonlinearities behave alike during training. Sigmoid and Tanh saturate for large $|z|$, causing the **vanishing gradient** problem. ReLU fixes this on the positive side with a constant derivative of 1, but pays for it with the **dying ReLU** problem: a unit that ever lands in the negative, zero-derivative region gets zero gradient forever. GELU and Swish, used in most modern Transformers, are smooth compromises that keep ReLU's benefits while avoiding its hard zero cutoff.

## Mathematical Explanation

| Activation | $g(z)$ | $g'(z)$ |
|---|---|---|
| Sigmoid | $\dfrac{1}{1+e^{-z}}$ | $g(z)(1-g(z))$, max $0.25$ at $z=0$ |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$, max $1$ at $z=0$ |
| ReLU | $\max(0,z)$ | $1$ if $z>0$ else $0$ |
| Leaky ReLU | $z$ if $z>0$ else $\alpha z$ | $1$ if $z>0$ else $\alpha$ ($\alpha=0.01$) |
| GELU | $z\,\Phi(z)$ | $\Phi(z) + z\,\phi(z)$ |
| Swish / SiLU | $z\cdot\text{sigmoid}(\beta z)$ | $\text{sigmoid}(\beta z) + \beta z\cdot\text{sigmoid}(\beta z)(1-\text{sigmoid}(\beta z))$ |

where $\Phi,\phi$ are the standard normal CDF/PDF. Sigmoid's derivative caps at 0.25, so every layer using it can shrink a backpropagated gradient by at least 4x, compounding multiplicatively with depth — the mathematical root of Section 3's result.

## Code Implementation (`03_activation_functions.ipynb`)

### 1. Validation against PyTorch

All six activations and their derivatives match `torch.sigmoid`, `torch.tanh`, `F.relu`, `F.leaky_relu`, `F.gelu`, and `F.silu` to within $10^{-15}$ across 21 test points in $[-5,5]$ — both forward values and autograd gradients.

### 2. The vanishing gradient problem, directly measured

A 20-layer, 16-unit-wide network (Xavier-style init, $\sqrt{1/\text{width}}$, matched to sigmoid/tanh) receives one backward pass:

| Activation | Layer 1 grad norm | Layer 20 grad norm | Ratio (1/20) |
|---|---|---|---|
| Sigmoid | 7.64e-13 | 9.68e-01 | 7.89e-13 |
| Tanh | 8.98e-01 | 3.99e+00 | 2.25e-01 |
| ReLU | 4.49e-03 | 3.32e+00 | 1.35e-03 |

Sigmoid's gradient collapses **13 orders of magnitude** from output to input layer. Tanh survives far better (derivative caps at 1). ReLU also showed real decay here — but this turned out to be an initialization artifact, not a property of ReLU:

| Init scheme | Layer 1 grad | Layer 20 grad |
|---|---|---|
| ReLU, Xavier-scale ($\sqrt{1/\text{width}}$) | 4.49e-03 | 3.32e+00 |
| ReLU, He-scale ($\sqrt{2/\text{width}}$) | 3.25e+00 | 3.32e+00 |

Matching the initialization scale to the activation (He init for ReLU) keeps the gradient nearly constant across all 20 layers — the "vanishing ReLU gradient" above was entirely a mismatched-initialization artifact. This directly previews Topic 06 (Weight Initialization).

### 3. The dying ReLU problem

An 8-unit hidden layer's biases are deliberately initialized to $-8$ on real `make_moons` data:

| Checkpoint | ReLU dead fraction | Leaky ReLU dead fraction |
|---|---|---|
| Epoch 0 | 1.000 | 1.000 |
| Epoch 500 | 1.000 | 1.000 |
| Epoch 1000 | 1.000 | 0.875 |
| Epoch 3000 | 1.000 | 0.875 |
| **Final train accuracy** | **0.5000** | **0.8500** |

**Every ReLU unit dies immediately and never recovers** — a dead ReLU has exactly zero gradient with respect to its incoming weights, so gradient descent has no mechanism to revive it, permanently stuck at 50% (pure guessing). **Leaky ReLU starts identically dead** by this threshold, but its small nonzero negative-region gradient lets 1 of 8 units escape by epoch 1000, reaching 85% accuracy under the identical unlucky initialization — a direct, mechanistic demonstration of why Leaky ReLU exists.

### 4. Real dataset: all six activations, head to head

Same 2-hidden-layer (16-16) architecture on `make_moons`, averaged over 5 seeds:

| Activation | Final train loss | Train acc | Test acc | Epochs to 90% train acc |
|---|---|---|---|---|
| Sigmoid | 0.06492 | 0.9150 | 0.8367 | **2165** |
| Tanh | 0.03415 | 0.9550 | 0.9017 | 16 |
| ReLU | 0.03032 | 0.9621 | 0.8950 | 172 |
| Leaky ReLU | 0.03777 | 0.9550 | 0.8833 | 152 |
| GELU | 0.03179 | 0.9593 | 0.8883 | 62 |
| Swish | 0.03521 | 0.9536 | 0.9017 | 41 |

Sigmoid is dramatically the slowest (2,165 epochs to reach 90% train accuracy — over 13x slower than the next-slowest alternative) and lands at the lowest final accuracy, a direct real-data consequence of Section 2's vanishing gradients. Every modern alternative converges in under 200 epochs with comparable, strong final accuracy.

## Common Pitfalls

1. **A gradient problem can look like an activation-function problem when it's really an initialization-scale mismatch.** Section 2 showed ReLU appearing to vanish under Xavier-style init purely because the scale wasn't matched to ReLU's actual behavior — always check whether the initialization scheme matches the activation before concluding an activation itself is the problem.
2. **Dying ReLU is permanent, not slow.** Once a ReLU unit's pre-activation is negative for every training example, its gradient is exactly zero and gradient descent literally cannot move its incoming weights — no amount of additional training epochs will revive it.
3. **"Dead" is a spectrum with Leaky ReLU, not binary.** The dead-neuron fraction for Leaky ReLU decreased over training as units escaped — a genuinely different, recoverable failure mode compared to standard ReLU's permanent death.
4. **Sigmoid's slowness in Section 4 wasn't a subtle effect — it was over an order of magnitude in training time**, on a network only 2 hidden layers deep. The effect compounds severely with more depth (Section 2).
5. **GELU and Swish require the standard normal CDF/PDF (or a sigmoid) internally**, making them more expensive to compute per-unit than ReLU — the accuracy/speed tradeoff against plain ReLU is real, not purely an accuracy upgrade for free.
6. **A network with 100% dead ReLU units can still exist without an explicit alarm.** Nothing in ordinary training (loss decreasing slowly, say) obviously signals this failure mode — checking the dead-neuron fraction directly, as done here, is the only reliable way to catch it.

## Function Reference

| Function | Purpose |
|---|---|
| `sigmoid`, `tanh`, `relu`, `leaky_relu`, `gelu`, `swish` (from scratch) | All six activations and their exact derivatives |
| `MLPAct` (from scratch) | Configurable-activation MLP with matching (Xavier/He) initialization and dead-neuron tracking |
| `forward_backward` (from scratch) | Single-pass gradient-norm-per-layer probe used for the vanishing gradient experiment |
| PyTorch (`torch.sigmoid`, `F.relu`, `F.gelu`, `F.silu`, etc.) | Reference implementations used for validation |
| `sklearn.datasets.make_moons` | Real non-linearly-separable dataset for the head-to-head comparison |

## Self-Test

1. Why does sigmoid's gradient collapse specifically by a factor related to 0.25 per layer, and where does that number come from?
2. Section 2 showed ReLU's apparent vanishing gradient was actually an initialization artifact. What experiment would you run to distinguish "the activation function is the problem" from "the initialization scale doesn't match the activation function"?
3. Why can a dead ReLU unit never be revived by further gradient descent steps, while a "dead" Leaky ReLU unit can?
4. If you saw a network training very slowly with a decreasing-but-slow loss curve, what specific diagnostic from this notebook would you check to distinguish vanishing gradients from dying ReLUs?
5. Why do GELU and Swish need the value of $z$ itself (not just its sign) inside their gating term, unlike ReLU?
6. In Section 4's real-data comparison, why did Tanh converge so much faster than Sigmoid despite both saturating for large $|z|$?

## Notebook

[`03_activation_functions.ipynb`](./03_activation_functions.ipynb)

## Further Reading

- Nair, V., Hinton, G.E. (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines.* ICML. (Popularized ReLU.)
- Maas, A.L., Hannun, A.Y., Ng, A.Y. (2013). *Rectifier Nonlinearities Improve Neural Network Acoustic Models.* ICML Workshop. (Introduces Leaky ReLU.)
- Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Ramachandran, P., Zoph, B., Le, Q.V. (2017). *Searching for Activation Functions.* arXiv:1710.05941. (Introduces Swish.)
- Glorot, X., Bengio, Y. (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks.* AISTATS. (The paper behind Xavier initialization, directly relevant to Section 2.)

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.03_Activation_Functions&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
