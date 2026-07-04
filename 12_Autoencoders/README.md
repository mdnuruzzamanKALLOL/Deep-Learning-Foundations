# Autoencoders

> Status: ✅ Complete

Basic, denoising, and variational autoencoders — implemented from scratch with full
backpropagation, validated against PyTorch to machine precision, and trained on sklearn
digits (8×8 grayscale).

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`12_autoencoders.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Supervised learning maps inputs to labels. Many real problems have **no labels** — only
raw data. An **autoencoder** learns to compress inputs into a low-dimensional **latent
code** and reconstruct them, forcing the network to discover structure in the data itself.

The architecture has two parts:
- **Encoder** $f_\theta(x)$ — maps $x \in \mathbb{R}^D$ to a code $z \in \mathbb{R}^K$
- **Decoder** $g_\phi(z)$ — maps $z$ back to $\hat{x} \in \mathbb{R}^D$

When $K < D$ (**undercomplete**), the bottleneck cannot simply copy the input; the network
must learn a compact representation. When $K \geq D$, the network can learn the identity
function unless regularized — a common failure mode.

**Denoising autoencoders** (DAE) corrupt inputs (noise, masking) but reconstruct the
**clean** original. This forces the model to capture robust structure rather than
memorizing pixel values.

**Variational autoencoders** (VAE) treat the latent code as a *distribution* rather than a
fixed point. A KL penalty toward a standard normal prior yields a smooth, generative latent
space suitable for sampling and interpolation — at the cost of reconstruction fidelity and
sometimes class separability in the latent space.

## Mathematical Explanation

**Basic autoencoder** (MSE reconstruction loss):

$$\mathcal{L}_\text{recon} = \frac{1}{ND}\|\hat{X} - X\|^2, \qquad \hat{X} = g_\phi(f_\theta(X))$$

**Backpropagation** follows the standard MLP chain rule through encoder and decoder. The
decoder output uses sigmoid (pixels in $[0,1]$); hidden layers use ReLU.

**Denoising autoencoder:**

$$\mathcal{L} = \frac{1}{ND}\| g_\phi(f_\theta(\tilde{X})) - X \|^2, \qquad \tilde{X} = \text{corrupt}(X)$$

Gradients flow through the corruption step only if it is differentiable (Gaussian noise is;
random pixel masking is not — use straight-through or backprop through the uncorrupted path
only for the encoder input).

**Variational autoencoder:**

$$q_\theta(z|x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2)), \qquad p(z) = \mathcal{N}(0, I)$$

$$\mathcal{L} = \mathcal{L}_\text{recon} + \beta \cdot D_{KL}\big(q(z|x)\,\|\,p(z)\big)$$

$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{K}\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$$

**Reparameterization trick:** $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$,
so gradients flow through $\mu$ and $\sigma$ while sampling remains stochastic.

**PCA connection:** PCA is the optimal *linear* rank-$K$ approximation (minimizing MSE).
A nonlinear autoencoder with the same bottleneck can capture curved manifolds that PCA cannot.

## Code Implementation

All code lives in [`12_autoencoders.ipynb`](12_autoencoders.ipynb). It:

1. Implements `Autoencoder`, `DenoisingAE`, and `VAE` with full forward/backward passes.
2. Validates the autoencoder against PyTorch (float64) and the VAE decoder path separately.
3. Trains on sklearn digits ($D=64$) with Adam (Topic 05), comparing AE vs PCA at $K \in \{2,8,32\}$.
4. Visualizes reconstructions at $K=2$ vs $K=32$.
5. Trains a denoising AE ($\sigma=0.4$ Gaussian noise) and compares to a plain AE on noisy test inputs.
6. Trains a 2D VAE, plots latent scatter colored by digit, and compares kNN classification
   from VAE vs AE vs PCA codes.
7. Demonstrates smooth VAE latent-space interpolation between two digits.

## Key Results

**PyTorch validation** (4×64 input, hidden 8, latent 4, float64):

| | Max diff |
|---|---|
| Forward | 1.11e-16 |
| dX | 3.33e-16 |
| dWe1 | 5.55e-16 |
| dWd2 | 4.44e-16 |
| VAE decode (z=μ) | 2.22e-16 |

**Latent bottleneck vs PCA** (hidden 128, Adam 200 epochs):

| Latent $K$ | AE test MSE | PCA test MSE | AE improvement |
|---|---|---|---|
| 2 | **0.03943** | 0.05222 | **+24.5%** |
| 8 | **0.01309** | 0.02386 | **+45.1%** |
| 32 | 0.00295 | **0.00262** | −12.8% (PCA wins) |

At small bottlenecks, the nonlinear AE clearly beats linear PCA. At $K=32$, PCA slightly
wins — it is the optimal linear subspace and needs no iterative training.

**Denoising** ($\sigma=0.4$ noise, latent 32, 200 epochs):

| Condition | MSE to clean target |
|---|---|
| Noisy input (no model) | 0.08518 |
| Basic AE on noisy input | 0.04902 |
| **Denoising AE on noisy input** | **0.02485** |

DAE improves over basic AE by **49.3%** on noisy test inputs.

**VAE** (2D latent, hidden 256, $\beta=1$, 300 epochs):

| Metric | Value |
|---|---|
| Test recon MSE | 0.07407 |
| Mean KL | 0.001 |
| kNN-5 accuracy (VAE μ) | **0.147** |
| kNN-5 accuracy (AE $z$) | **0.756** |
| kNN-5 accuracy (PCA) | **0.622** |

The VAE's KL penalty regularizes latents toward $\mathcal{N}(0,I)$, spreading classes and
lowering kNN accuracy — a deliberate trade-off for a smooth generative space. Plain AE
preserves class structure much better in 2D.

## Common Pitfalls

1. **Using an overcomplete bottleneck without regularization.** If $K \geq D$, the AE can
   learn the identity mapping and useless latents. Use $K < D$, weight decay, or denoising.
2. **Training on clean data but expecting denoising at test time.** A plain AE trained only
   on clean inputs has never seen corruption; a DAE trained on noisy inputs with clean
   targets is required for robust denoising.
3. **Comparing AE to PCA with poorly tuned training.** With naive SGD and few epochs, AE
   can look worse than PCA even when nonlinear capacity should help. Adam and sufficient
   epochs are needed for a fair comparison.
4. **Forgetting sigmoid on the decoder output for bounded pixels.** Digits are scaled to
   $[0,1]$; linear outputs can drift outside the valid range and inflate MSE.
5. **Expecting VAE latents to cluster by class like a supervised embedding.** The KL term
   pushes latents toward a standard normal, which can destroy class separability (our VAE
   kNN accuracy is 0.147 vs 0.756 for a plain AE at the same 2D width).
6. **Ignoring the $\beta$ trade-off in VAEs.** Higher $\beta$ increases KL weight, improving
   latent regularization and generation but often hurting reconstruction and downstream
   separability. Tune $\beta$ for your goal (generation vs compression).

## Function Reference

| Function / Class | Purpose |
|---|---|
| `Autoencoder` | MLP encoder (ReLU→linear) + decoder (ReLU→sigmoid) with full backprop |
| `DenoisingAE` | Autoencoder trained on Gaussian-corrupted inputs, clean targets |
| `VAE` | Variational AE with reparameterization trick and KL penalty |
| `Adam` | Adam optimizer (from Topic 05) for training |
| `train_autoencoder` | Mini-batch training loop for AE / DAE / VAE |
| `mse` | Mean squared error helper |

## Self-Test

1. Write the basic autoencoder loss function. Why does an undercomplete bottleneck ($K < D$)
   force compression?
2. PCA at rank $K$ is the optimal *linear* reconstruction. Why does our nonlinear AE beat
   PCA at $K=2$ and $K=8$ but not at $K=32$?
3. In a denoising autoencoder, we forward corrupted $\tilde{X}$ but compute loss against
   clean $X$. What would happen if we computed loss against $\tilde{X}$ instead?
4. Write the VAE KL divergence term for diagonal Gaussian $q(z|x)$ and prior $\mathcal{N}(0,I)$.
   What happens to $\mu$ and $\sigma$ if $\beta \to 0$? If $\beta \to \infty$?
5. Our VAE achieves kNN accuracy 0.147 on 2D latents while a plain AE achieves 0.756. Is the
   VAE "worse"? What is it optimizing for instead?
6. Why do we validate weight gradients against PyTorch with a $\times N$ scaling factor
   when our loss uses batch mean but PyTorch's test uses sum?

## Further Reading

- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 14.5](https://www.deeplearningbook.org/contents/autoencoders.html) — autoencoders and denoising.
- Vincent, P. et al. (2008). ["Extracting and Composing Robust Features with Denoising Autoencoders"](https://www.cs.toronto.edu/~hinton/absps/dae.pdf) — original DAE paper.
- Kingma, D. & Welling, M. (2014). ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) — VAE and reparameterization trick.
- Higgins, I. et al. (2017). ["beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](https://openreview.net/forum?id=Sy2fzU0lg) — $\beta$-VAE trade-off.

---
[← Back to Deep Learning Foundations](../README.md)
