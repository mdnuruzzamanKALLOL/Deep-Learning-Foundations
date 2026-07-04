# Batch Normalization & Layer Normalization

> Status: ✅ Complete

BatchNorm and LayerNorm — implemented from scratch with full backward passes, validated
against PyTorch to machine precision, and stress-tested to show why normalizing activations
stabilizes deep network training.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`08_batch_normalization_layer_normalization.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

As a network trains, the distribution of activations at each layer keeps shifting — because
the weights in all previous layers are changing simultaneously. Every layer must
continuously re-adapt to inputs whose mean and variance are moving targets. This "internal
covariate shift" makes deep networks slow and sensitive to learning rate.

**Batch Normalization** fixes this by standardizing each feature's activations across the
current mini-batch, then applying a learnable scale and shift ($\gamma$, $\beta$). During
training it also accumulates running mean/variance for use at inference.

**Layer Normalization** standardizes across the *feature* dimension within each individual
sample instead of across the batch. This makes it independent of batch size — critical for
Transformers, RNNs, and any setting where batch size is 1 or varies.

## Mathematical Explanation

**BatchNorm** (for input $x$ of shape $N \times C$, normalize each feature across $N$):

$$\mu_B = \frac{1}{N}\sum_{n=1}^N x_n, \quad \sigma_B^2 = \frac{1}{N}\sum_{n=1}^N (x_n - \mu_B)^2$$
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Running stats (training): $\text{running\_mean} \leftarrow (1-m)\cdot\text{running\_mean} + m\cdot\mu_B$

**LayerNorm** (normalize each sample across $C$ features):

$$\mu_L = \frac{1}{C}\sum_{c=1}^C x_c, \quad \sigma_L^2 = \frac{1}{C}\sum_{c=1}^C (x_c - \mu_L)^2$$
$$\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Both are placed after the linear layer and **before** the nonlinearity (ReLU) in modern
architectures, though placement varies by model family.

## Code Implementation

All code lives in [`08_batch_normalization_layer_normalization.ipynb`](08_batch_normalization_layer_normalization.ipynb). It:

1. Implements `BatchNorm1d` and `LayerNorm1d` with full analytical backward passes.
2. Validates forward and backward against `nn.BatchNorm1d` and `nn.LayerNorm` in float64.
3. Tracks activation mean/std through 6 layers with no norm, BatchNorm, and LayerNorm.
4. Demonstrates BatchNorm **train vs eval mode** (batch stats vs running stats).
5. Sweeps learning rate on a 4-layer MLP — shows unnormalized training diverges at lr=2.0
   while BatchNorm and LayerNorm succeed.
6. Stress-tests a **depth-8** network at lr=2.0.
7. Compares BatchNorm vs LayerNorm at **batch size 1** and measures batch-statistic noise
   vs batch size.

## Key Results

**PyTorch validation** (32×16 tensor, float64):

| | Forward max diff | Backward dx max diff |
|---|---|---|
| BatchNorm | 8.88e-16 | 1.38e-15 |
| LayerNorm | 4.44e-16 | 3.55e-16 |

**Activation statistics through 6 layers** (width 64, random input):

| Layer | None std | BN std | LN std |
|---|---|---|---|
| 1 | 0.831 | 0.584 | 0.578 |
| 6 | 0.642 | 0.601 | 0.614 |

BatchNorm and LayerNorm keep activation standard deviations in a tighter range across depth.

**Learning rate sweep** (4-layer MLP, 800 epochs, test accuracy):

| LR | None | BatchNorm | LayerNorm |
|---|---|---|---|
| 0.05 | 1.000 | 0.987 | 0.973 |
| 1.00 | 0.987 | 0.960 | 1.000 |
| **2.00** | **0.387** (overflow) | **0.987** | **1.000** |

At lr=2.0, the unnormalized network's weights overflow; both normalization methods train stably.

**Depth-8 stress test at lr=2.0** (1000 epochs):

| Norm | Result |
|---|---|
| None | Diverged |
| BatchNorm | Test acc **0.987** |
| LayerNorm | Test acc **0.907** |

**Batch size 1** (500 steps, lr=0.1):

| Norm | Test acc | Why |
|---|---|---|
| BatchNorm | **0.387** | Batch variance = 0; normalization degenerate |
| LayerNorm | **0.627** | Normalizes across 64 features per sample |

**BatchNorm statistic noise** (std of batch-mean estimate, 500 draws):

| Batch size | Std of mean estimate |
|---|---|
| 1 | 0.1320 |
| 4 | 0.0634 |
| 64 | 0.0155 |

Smaller batches produce noisier BatchNorm statistics — another reason LayerNorm is preferred
when batch size is small or variable.

## Common Pitfalls

1. **Forgetting `model.eval()` / `training=False` for BatchNorm at inference.** BatchNorm
   uses batch statistics during training but running statistics at eval. Evaluating in train
   mode makes predictions depend on which other samples happen to be in the batch.
2. **Using BatchNorm with batch size 1.** A single sample has zero cross-sample variance;
   BatchNorm divides by $\epsilon$ only, producing arbitrary outputs (test acc 0.387 vs
   LayerNorm's 0.627 in our experiment). Use LayerNorm or GroupNorm instead.
3. **Assuming BatchNorm and LayerNorm are interchangeable.** They normalize over different
   axes and behave differently with batch size. BatchNorm is standard in CNNs with large
   batches; LayerNorm is standard in Transformers and RNNs.
4. **Placing BatchNorm after ReLU instead of before.** The original paper normalizes pre-activations
   (before the nonlinearity). Post-activation BatchNorm is used in some architectures but
   is not equivalent — the choice affects gradient flow through the normalization layer.
5. **Ignoring the running-statistics warm-up.** Early in training, running mean/variance are
   poor estimates of the true population statistics. Some frameworks use a higher momentum
   early on, or freeze running stats for the first N batches.
6. **Confusing BatchNorm with batch-wise standardization of inputs.** Normalizing raw input
   features (zero mean, unit variance preprocessing) is a separate step. BatchNorm normalizes
   *intermediate activations* inside the network, adaptively during training.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `BatchNorm1d` | Batch normalization with running stats, train/eval modes, full backward |
| `LayerNorm1d` | Layer normalization per sample, no running stats, full backward |
| `forward_stats` | Tracks activation mean/std through a deep stack with optional norm |
| `MLPNorm` | 4–8 layer MLP with configurable BatchNorm, LayerNorm, or none after each linear layer |

## Self-Test

1. For a batch of shape $(N, C)$, which axis does BatchNorm reduce over? Which axis does
   LayerNorm reduce over? What are the shapes of $\mu_B$ and $\mu_L$?
2. Why does BatchNorm need running statistics at inference but LayerNorm does not?
3. In Section 5, why does the unnormalized MLP fail at lr=2.0 while BatchNorm succeeds?
   What happens to the weight matrices when learning rate is too large without normalization?
4. With batch size 1, BatchNorm sets $\sigma_B^2 = 0$ for each feature. What does
   $\hat{x} = (x - \mu_B)/\sqrt{0 + \epsilon}$ reduce to, and why is this not useful?
5. BatchNorm has learnable $\gamma$ and $\beta$. If $\gamma = \sigma_B$ and $\beta = \mu_B$,
   the output equals the original input $x$. Why might the network want to learn this
   identity mapping for some features?
6. Transformers use LayerNorm, not BatchNorm. Given that Transformer training often uses
   small effective batch sizes (due to sequence length memory limits), explain why LayerNorm
   is the natural choice.

## Further Reading

- Ioffe, S. & Szegedy, C. (2015). ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167) — the original BatchNorm paper.
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). ["Layer Normalization"](https://arxiv.org/abs/1607.06450) — LayerNorm for batch-size-independent normalization.
- [PyTorch nn.BatchNorm1d documentation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) — running stats, momentum, and affine parameters.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 8.7.1](https://www.deeplearningbook.org/contents/optimization.html) — normalization strategies in the context of optimization.

---
[← Back to Deep Learning Foundations](../README.md)
