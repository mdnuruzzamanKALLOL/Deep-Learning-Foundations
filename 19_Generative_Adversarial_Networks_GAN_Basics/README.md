# Generative Adversarial Networks (GAN) Basics

> Status: ✅ Complete

A generator and discriminator implemented from scratch (validated against PyTorch to
machine precision), trained adversarially on a multi-modal 2D distribution and on
sklearn digits — with an honest, quantitative look at two classic GAN failure modes:
vanishing generator gradients and mode collapse.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`19_generative_adversarial_networks_gan_basics.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Topic 12's autoencoders and VAEs learn generative models by minimizing reconstruction
error against real targets. **GANs** (Goodfellow et al., 2014) take a completely
different approach: two networks compete.

- **Generator** $G(z)$ maps noise $z \sim \mathcal{N}(0,I)$ to a fake sample, trying to
  fool the discriminator.
- **Discriminator** $D(x)$ outputs $P(\text{real})$, trying to tell real data from
  $G$'s fakes.

There is no reconstruction target and no explicit density — $G$ never sees real data
directly, only the gradient signal $D$ provides through its verdict on $G$'s outputs.
This adversarial setup is elegant but famously **unstable**: two classic failure modes
are **vanishing generator gradients** (the original minimax loss saturates when $D$ is
confident) and **mode collapse** (the generator concentrates on a few "safe" outputs
that fool the current discriminator, ignoring the rest of the target distribution).

## Mathematical Explanation

**Minimax objective** (Goodfellow et al., 2014):

$$\min_G \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$

$D$ maximizes this via standard binary cross-entropy (real → 1, fake → 0). $G$
minimizes $\log(1 - D(G(z)))$ — i.e. pushes $D(G(z)) \to 1$.

**The vanishing-gradient problem:** early in training $D$ easily separates real from
fake, so $D(G(z)) \approx 0$ and $\log(1-D(G(z)))$ sits in its flat, saturated region —
the same sigmoid-saturation mechanism from Topic 16, here applied to the generator's
training signal rather than a deep stack of layers.

**Non-saturating fix:** train $G$ to instead **maximize** $\log D(G(z))$ (minimize
$-\log D(G(z))$) — the same fixed point ($D(G(z))=0.5$ at equilibrium), but a gradient
that stays large exactly when $G$ needs it most (when $D(G(z))$ is small).

$$\frac{\partial \mathcal{L}_\text{minimax}}{\partial D(G(z))} = \frac{-1}{1-D(G(z))}
\Big|_{D(G(z)) \to 0} \to -1 \quad\text{(bounded, small downstream gradient)}$$

$$\frac{\partial \mathcal{L}_\text{non-sat}}{\partial D(G(z))} = \frac{-1}{D(G(z))}
\Big|_{D(G(z)) \to 0} \to -\infty \quad\text{(large, informative gradient)}$$

## Code Implementation

All code lives in
[`19_generative_adversarial_networks_gan_basics.ipynb`](19_generative_adversarial_networks_gan_basics.ipynb). It:

1. Implements `Generator` (ReLU MLP, linear output) and `Discriminator` (LeakyReLU MLP,
   sigmoid output) with full manual backprop.
2. Validates both **independently** against equivalent PyTorch `nn.Linear`-based
   modules (float64) — no single "GAN module" exists in PyTorch, but each network alone
   is a standard MLP.
3. Reproduces Goodfellow's vanishing-gradient motivation directly: pretrains $D$ against
   a weak $G$, then computes $G$'s gradient under both loss formulations from the exact
   same network state.
4. Trains full adversarial GANs on an **8-Gaussians** ring (classic mode-collapse toy
   problem) under three D-step/G-step ratios, measuring exact mode coverage.
5. Trains the same architecture (sigmoid output) on sklearn digits restricted to a
   single class ("0") as a qualitative image-generation capstone.

## Key Results

**PyTorch validation** (Generator: 4→8→2, Discriminator: 2→8→1, float64):

| Network | Forward max diff | Input-grad max diff | Weight-grad max diff |
|---|---|---|---|
| Generator | 1.11e-16 | 1.11e-16 | 8.88e-16 |
| Discriminator | 0.0 | 2.78e-17 | 6.94e-18 |

**Vanishing generator gradient** (D pretrained to confidently reject a weak generator):

| Quantity | Value |
|---|---|
| Mean $D(\text{real})$ after pretraining | 0.9980 |
| Mean $D(\text{fake})$ after pretraining | 0.0003 |
| Generator gradient norm, minimax loss | 2.43e-04 |
| Generator gradient norm, non-saturating loss | 9.31e-02 |
| **Ratio (non-saturating / minimax)** | **382.9×** |

From the identical network state, switching loss formulations alone changes the
generator's gradient magnitude by nearly **three orders of magnitude** — a direct,
quantitative reproduction of the original GAN paper's motivation for the
non-saturating loss.

**Mode collapse on 8-Gaussians** (final mode coverage out of 8, 2000 generated samples):

| Training regime | Modes covered | Per-mode counts |
|---|---|---|
| Balanced (1 D step : 1 G step) | **8 / 8** | 195, 235, 237, 238, 360, 212, 236, 287 |
| Collapsed (1 D step : 5 G steps) | **2 / 8** | 0, 0, 0, 0, 0, 647, 1353, 0 |
| D-favored (5 D steps : 1 G step) | **8 / 8** | 237, 150, 527, 125, 191, 381, 104, 285 |

Updating the generator 5× more often than the discriminator collapses coverage from
8/8 to 2/8 — the generator repeatedly exploits whatever single mode currently fools a
comparatively stale discriminator. Giving the discriminator the training advantage
instead (5 D steps per G step, closer to the original paper's recommendation of nearly
optimizing $D$ between $G$ updates) restores full coverage and converges by step 1000
instead of step 2000. **A qualitative nuance worth reporting honestly:** the balanced
run's samples spread fairly diffusely across the whole disk (all 8 regions register
"covered" but without tight per-mode clustering), while the D-favored run's samples form
visibly tighter, denser clusters near the true centers — the mode-coverage count alone
doesn't capture this difference in fidelity.

**Qualitative digit generation** (sigmoid-output generator, 178 "0" images, 3000 steps):

Generated samples are blurrier than real digits (expected: tiny from-scratch MLP,
no convolutional structure, 178 training images) but clearly reproduce the digit "0"'s
characteristic closed-loop shape.

## Common Pitfalls

1. **Using the original minimax generator loss "as written."** It is mathematically
   elegant but saturates immediately whenever $D$ is confident — which is *most* of
   early training. Always use the non-saturating loss ($-\log D(G(z))$) in practice.
2. **Calling `forward()` on a different input between computing a probability and
   calling `backward()`.** Our `Discriminator`/`Generator` cache only their most recent
   forward pass; a stray `disc.forward(other_x)` between computing `prob_fake` and
   calling `disc.backward()` silently uses the wrong cached activations. This bug
   appeared during development and produced a nonsensical gradient ratio until fixed by
   recomputing the exact forward pass immediately before each backward call.
3. **Interpreting "loss looks stable" as "training is healthy."** Our collapsed run's
   discriminator loss (1.4–1.8) looks perfectly reasonable throughout — it does **not**
   reveal mode collapse. Always track a data-quality metric (like mode coverage, or
   visual inspection) alongside the adversarial losses.
4. **Assuming more generator updates always help the generator.** 5 G-steps-per-D-step
   is a worse configuration than balanced or D-favored training here — an
   under-trained discriminator gives an inaccurate, easily-gamed gradient signal.
5. **Treating "modes covered" as equivalent to "sample quality."** Our balanced run hits
   8/8 mode coverage with a diffuse, low-fidelity spread; the D-favored run hits the same
   8/8 with visibly tighter clusters. Report both a coverage metric and a qualitative
   (visual) check — neither alone tells the whole story.
6. **Expecting a tiny from-scratch MLP GAN to match modern image quality.** Our digit
   samples are legibly "0"-shaped but blurry. Real gains require convolutional
   generators/discriminators, much more data, and stabilization techniques (spectral
   normalization, gradient penalties) beyond this notebook's from-scratch scope.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `Generator` | Noise → 2-layer ReLU MLP → linear output (2D toy data) |
| `GeneratorSigmoid` | Same, with sigmoid output for $[0,1]$-valued pixels |
| `Discriminator` | Input → 2-layer LeakyReLU MLP → sigmoid probability |
| `Adam` | Optimizer for both networks (Topic 05) |
| `make_8gaussian_data` | 8 Gaussian modes arranged in a ring (classic mode-collapse toy task) |
| `mode_coverage` | Assigns samples to nearest true mode, counts modes above a coverage threshold |
| `train_gan` | Full adversarial training loop with configurable D-step/G-step ratio |

## Self-Test

1. Write the minimax generator loss and the non-saturating generator loss. Why do they
   share the same fixed point ($D(G(z)) = 0.5$) despite having very different gradients?
2. Our vanishing-gradient experiment shows a 382.9× gradient ratio. Under what
   discriminator confidence level ($D(\text{fake}) \to 0$ vs $D(\text{fake}) \to 0.5$)
   would you expect this ratio to shrink toward 1×? Why?
3. Explain, mechanistically, why 5 generator updates per discriminator update causes
   mode collapse but 5 discriminator updates per generator update does not.
4. Our collapsed run's discriminator loss looks unremarkable throughout training
   (Pitfall 3). What additional metric would you monitor in a real GAN training run to
   catch mode collapse without a hand-designed "true mode" ground truth?
5. The balanced and D-favored runs both achieve 8/8 mode coverage, but look visibly
   different (diffuse vs. tightly clustered). Propose a quantitative metric (beyond mode
   coverage) that would capture this fidelity difference.
6. Why must `Discriminator.backward()` be called immediately after the matching
   `forward()` call, with no intervening forward call on different data?

## Further Reading

- Goodfellow, I. et al. (2014). ["Generative Adversarial Networks"](https://arxiv.org/abs/1406.2661) — original GAN paper, minimax and non-saturating losses.
- Radford, A., Metz, L., Chintala, S. (2016). ["Unsupervised Representation Learning with Deep Convolutional GANs"](https://arxiv.org/abs/1511.06434) — DCGAN, the standard convolutional recipe.
- Metz, L. et al. (2017). ["Unrolled Generative Adversarial Networks"](https://arxiv.org/abs/1611.02163) — mode collapse and the 8-Gaussians toy problem.
- Arjovsky, M., Bottou, L. (2017). ["Towards Principled Methods for Training GANs"](https://arxiv.org/abs/1701.04862) — theoretical analysis of training instability.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 20.10.4](https://www.deeplearningbook.org/contents/generative_models.html) — GANs.

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.19_Generative_Adversarial_Networks_GAN_Basics&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
