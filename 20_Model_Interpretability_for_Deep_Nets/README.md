# Model Interpretability for Deep Nets

> Status: ✅ Complete

Grad-CAM, saliency maps, and explaining black-box predictions

## Why This Topic Matters

Every topic before this one asked "how do we make the network fit the data?" This one asks
the question that actually matters once a network *is* deployed: **why did it make this
specific prediction?** A CNN that hits 97.6% test accuracy (the same network from
[Topic 09 — CNN Basics](../09_CNN_Basics/)) is not automatically trustworthy — it could be
right for the wrong reasons (e.g. keying off a watermark, a background texture, or a
dataset artifact rather than the object itself). Interpretability methods try to expose
*what the network is actually looking at*, so we can catch failures like that before they
ship.

This topic also closes an important loop: [Adebayo et al. (2018), "Sanity Checks for
Saliency Maps"](https://arxiv.org/abs/1810.03292) showed that some popular interpretability
methods produce explanations that barely change even when the network's weights are
**completely randomized** — meaning they were never really explaining the trained model at
all, just doing a form of edge detection on the input. We reproduce that exact finding from
scratch, at toy scale, with our own numbers.

## Concept & Intuition

Four families of methods answer "why did the model predict class $c$ for image $x$?" in
different ways:

| Method | Idea | Cost |
|---|---|---|
| **Vanilla gradient saliency** | How much does the output change if I nudge each input pixel? | 1 backward pass |
| **Guided backprop** | Like vanilla gradients, but only propagate positive signal through ReLUs | 1 backward pass |
| **Grad-CAM** | Which spatial regions of the *last conv layer* mattered most, weighted by their gradient importance? | 1 backward pass |
| **Occlusion sensitivity** | Literally block out each patch of the image and see how much the predicted probability drops | $O(HW)$ forward passes |

The first three are **gradient-based** — cheap, but only ever tell you about an infinitesimal
neighborhood of $x$ (a first-order Taylor approximation of the network). The last is
**perturbation-based** — expensive, but it directly measures the actual causal effect of
removing information, with no linearization assumption.

## Mathematical Explanation

### Vanilla gradient saliency

For input $x$ and the (pre-softmax) score $y^c(x)$ of class $c$:

$$S(x) = \left|\frac{\partial y^c}{\partial x}\right|$$

This is exactly one backward pass with the "loss" set to $y^c$ itself (a one-hot vector at
position $c$, not the cross-entropy loss). It is the first-order Taylor coefficient of $y^c$
around $x$: $y^c(x + \delta) \approx y^c(x) + S(x)^\top \delta$, i.e. it says which pixels the
score is most sensitive to *right now*.

### Guided backprop

Standard backprop through a ReLU only blocks gradient where the **forward** activation was
$\le 0$:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\cdot \mathbb{1}[x > 0]$$

Guided backprop ([Springenberg et al., 2014](https://arxiv.org/abs/1412.6806)) additionally
blocks *negative incoming gradient*:

$$\frac{\partial L}{\partial x}\bigg|_{\text{guided}} = \frac{\partial L}{\partial y}\cdot \mathbb{1}[x > 0]\cdot \mathbb{1}\!\left[\frac{\partial L}{\partial y} > 0\right]$$

The intuition sold in the original paper is that this only lets "evidence for the class"
flow backward, producing sharper-looking maps. As we'll show, sharper does not mean more
faithful.

### Grad-CAM

[Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391) use the gradient flowing into the
**last convolutional feature map** $A \in \mathbb{R}^{C\times H\times W}$ (rather than all
the way back to the input) to build a coarse, class-discriminative localization map. For
each channel $k$, global-average-pool the gradient to get an importance weight:

$$\alpha_k^c = \frac{1}{HW}\sum_{i,j}\frac{\partial y^c}{\partial A_{k,i,j}}$$

then take a weighted combination of the activation maps, followed by ReLU (we only care
about features that have a *positive* influence on the class):

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c\, A_k\right)$$

$L^c$ has the spatial resolution of the last conv layer — much coarser than the input — and
is normally upsampled back to input resolution for visualization.

### Occlusion sensitivity

Slide an occluding patch (gray/zero) of size $k\times k$ over the image with some stride,
and record the drop in the predicted class probability at each location:

$$M(i,j) = p^c(x) - p^c(x_{\text{occluded at }(i,j)})$$

Large $M(i,j)$ means the network's confidence collapsed when that region was hidden — direct
evidence that region was load-bearing for the prediction, with no linearization involved.

### The sanity check (Adebayo et al., 2018)

Take a trained model $f$ and an explanation method $\text{Exp}(f, x)$. Replace $f$'s weights
with **independent random values** to get $f_{\text{rand}}$ (same architecture, same input,
completely different function). Compute $\text{Exp}(f_{\text{rand}}, x)$ for the *same*
output unit and compare it to $\text{Exp}(f, x)$ via cosine similarity. If an explanation
method is actually reading off what the trained network learned, the two maps should look
essentially unrelated (similarity $\approx 0$) — the model underneath is a totally different
function now. If the maps stay similar, the method was never sensitive to the learned
weights at all.

## Implementation Notes

- The `Conv2d`/`MaxPool2d` (im2col-based) layers and `SimpleCNN` architecture are reused
  verbatim from [Topic 09 — CNN Basics](../09_CNN_Basics/) (same seeds, same
  hyperparameters), trained on `sklearn.datasets.load_digits` (8×8 grayscale, 10 classes) to
  the identical 97.6% test accuracy, so every interpretability result below is about a model
  we've already validated is working correctly — not compensating for a broken classifier.
- `SimpleCNN.backward_from_class(x, class_idx, guided=False)` backpropagates a **one-hot
  "logit" gradient** (not the cross-entropy training loss) from a chosen class down to the
  input, returning both `dx` (input-space gradient, for saliency/guided-backprop) and `da2`
  (gradient at the last conv activation, for Grad-CAM) in a single pass — reusing the exact
  same `Conv2d.backward` / `MaxPool2d.backward` machinery as training, just with a different
  starting gradient and (optionally) a gated ReLU backward for the guided variant.
- All four methods run on the *same* trained network and the *same* test images, so
  differences between them are attributable to the method, not to different models.

## Key Results

### 1. Gradient correctness

`backward_from_class`'s analytic `dx` was checked against central-difference numerical
gradients ($\epsilon=10^{-6}$) of the class logit at 20 random input pixels:

| | value |
|---|---|
| Median absolute error | 1.7×10⁻¹⁰ |
| Max absolute error | 1.8×10⁻¹ (4 of 20 pixels above 10⁻⁴) |

The outliers are not a bug: they sit exactly on a **max-pool argmax boundary**, where a
$10^{-6}$ perturbation is enough to flip which input pixel the pooling layer selects as the
max, so the true function is locally non-smooth there and finite differences disagree with
the (still-correct) analytic subgradient. This is a general pitfall of gradient-checking any
network containing max-pooling, not specific to interpretability code.

### 2. Do the four methods actually localize the digit?

For each of 60 test digits we measured what fraction of each heatmap's total (positive) mass
falls on "digit-ink" pixels (input intensity > 0.15) versus background, and compared to a
random heatmap (which should recover the base rate — digit-ink pixels are 44% of each
8×8 image):

![Localization quality bar chart](interpretability_localization_bars.png)

| Method | Mean fraction of mass on digit-ink | vs. random baseline (0.440) |
|---|---:|---|
| Random heatmap | 0.440 | — (sanity check on the metric itself) |
| Grad-CAM | 0.377 | **worse than random** |
| Vanilla saliency | 0.568 | modestly better |
| Occlusion sensitivity | 0.880 | much better |

The Grad-CAM result is a genuine, reproducible finding, not a bug: our CNN's last conv layer
(before pooling into the classifier) has only **4×4** spatial resolution on an 8×8 input.
Grad-CAM was designed for ImageNet-scale networks whose last conv map is 7×7 or 14×14 on
224×224 inputs — a much finer relative resolution. At 4×4 on an 8×8 image, each Grad-CAM
"pixel" covers a 2×2 block that is a large fraction of the whole digit, so the method
degrades to a nearly-uninformative coarse blob. **Grad-CAM's spatial resolution is bottlenecked by how many pooling layers precede it — the deeper the network, the coarser (and
often less useful) the localization**, unless you pick an earlier, higher-resolution
layer as the CAM target. Occlusion sensitivity has no such bottleneck since it operates
directly on the input, and here it dominates.

### 3. Cross-method agreement is weak — and that's an honest, informative result

| Comparison | Pearson correlation |
|---|---:|
| \|Vanilla saliency\| vs. occlusion sensitivity | 0.30 (mean over 30 digits) |
| Grad-CAM (native 4×4 res) vs. occlusion (downsampled to 4×4) | −0.14 (mean over 30 digits) |

Vanilla gradients and occlusion agree only modestly — gradients are a *local, linearized*
sensitivity measure, while occlusion measures a *large, discrete* perturbation, so they are
not expected to match exactly even when both are "correct" in their own terms. Grad-CAM's
negative correlation with occlusion, even after matching resolutions, is consistent with
finding 2 above: at this scale Grad-CAM's signal is dominated by which of the 16 output
channels happen to have a large gradient average, not by where the digit actually is.

### 4. Sample explanations

![Interpretability methods grid](interpretability_methods_grid.png)

Occlusion sensitivity (rightmost column) visibly tracks the digit's strokes. Vanilla
saliency and guided backprop are visually similar to each other — both diffuse, edge-heavy
maps that highlight the digit's outline but also plenty of background. Grad-CAM (jet
colormap) is visibly blocky — exactly the 4×4-grid artifact described above.

### 5. The sanity check: which methods actually depend on the learned weights?

We took the trained CNN, built a second network with the **same architecture but
independently re-randomized weights** (test accuracy 10.2% — chance level, confirming it
learned nothing), and recomputed each method's explanation for the *same* input images and
the *same* target class unit, then measured cosine similarity to the trained model's
explanation:

![Sanity check visual](interpretability_sanity_check.png)

| Method | Cosine similarity (trained vs. randomized model) | Sanity check |
|---|---:|---|
| Vanilla gradient saliency | −0.05 | ✅ passes — map is essentially uncorrelated once the weights are destroyed |
| Grad-CAM | 0.31 | ⚠️ partial — some sensitivity to weights, but far from zero |
| Guided backprop | 0.57 | ❌ fails — map stays substantially similar even with a completely random network |

This reproduces the headline finding of Adebayo et al. (2018) at toy scale: **guided
backprop's "sharper-looking" maps are largely an artifact of the input's local structure and
the ReLU-gating pattern, not of what the specific trained weights learned.** A visually
convincing explanation is not the same thing as a faithful one — the sanity check is a cheap
way to catch the difference, and it should be run before trusting any new interpretability
method.

## Common Pitfalls

1. **Trusting a saliency map because it "looks reasonable."** Guided backprop produces
   crisp, plausible-looking maps yet fails the randomization sanity check badly — visual
   plausibility is not evidence of faithfulness.
2. **Using Grad-CAM on a layer that's too coarse.** If the target conv layer's spatial
   resolution is a small fraction of the input's, Grad-CAM's localization degrades to a
   coarse blob — check the actual feature map resolution before trusting the heatmap.
3. **Numerically gradient-checking through max-pooling and expecting a perfect match
   everywhere.** A handful of large per-pixel errors at pool-boundary pixels are expected,
   not a bug; check the *median* error, and separately confirm the outliers sit at
   `argmax`-tie locations.
4. **Comparing raw gradients to occlusion-based sensitivity and expecting near-identical
   maps.** Gradients are a first-order local approximation; occlusion measures a large,
   discrete intervention. Disagreement is expected and doesn't mean one of them is wrong.
5. **Forgetting to fix the target class/unit before randomizing weights in a sanity check.**
   The random model will generally predict a *different* class than the trained model; the
   sanity check must probe the *same* output unit in both models (we fix it to the trained
   model's prediction) or the comparison is meaningless.
6. **Un-normalized heatmap overlays.** Grad-CAM and occlusion maps have arbitrary units and
   very different scales from image to image; always min-max normalize (or at least clip to
   $\ge 0$) before comparing or overlaying, as done throughout this notebook.

## Function Reference

| Name | Description |
|---|---|
| `Conv2d`, `MaxPool2d` | im2col-based conv/pool layers, reused from Topic 09 |
| `SimpleCNN.forward_full(x)` | Forward pass caching every intermediate activation |
| `SimpleCNN.step(x, y, lr)` | One training step (cross-entropy loss + full backprop) |
| `SimpleCNN.backward_from_class(x, class_idx, guided=False)` | Backprop a one-hot class-logit gradient to the input (`dx`) and to the last conv activation (`da2`), optionally with guided-ReLU gating |
| `saliency_map(model, x, class_idx, guided=False)` | Vanilla gradient or guided-backprop saliency map |
| `grad_cam(model, x, class_idx)` | Grad-CAM heatmap, upsampled to input resolution |
| `occlusion_sensitivity(model, x, class_idx, patch, stride)` | Perturbation-based sensitivity heatmap |
| `mass_on_digit(heat, img, thresh)` | Fraction of a heatmap's mass on above-threshold ("ink") pixels |
| `randomize_model(model, seed)` | Fresh `SimpleCNN` with independent random weights, same architecture — used for the sanity check |
| `cosine_sim(a, b)` | Cosine similarity between two flattened maps |

## Self-Test Questions

1. Why does vanilla gradient saliency require only *one* backward pass, while occlusion
   sensitivity requires $O(HW/\text{stride}^2)$ forward passes?
2. If you moved the Grad-CAM target from `conv2`'s output (4×4) to `conv1`'s output (8×8,
   same resolution as the input), would you expect the localization quality (mass-on-digit
   fraction) to go up or down? Why?
3. Guided backprop's cosine similarity to itself under weight randomization was 0.57 — far
   from 1.0, but also far from vanilla gradient's −0.05. What does "far from 1.0" tell you
   that "far from 0" doesn't?
4. Why do we fix the target class to the *trained* model's prediction (not the random
   model's own prediction, and not the ground-truth label) when computing the randomized
   model's saliency map in the sanity check?
5. The numerical gradient check for `backward_from_class` had a handful of outlier pixels
   out of 20 with large error, all located near max-pool boundaries. Would switching from max pooling
   to average pooling in `SimpleCNN` eliminate this kind of outlier? Why?
6. Occlusion sensitivity used a gray/zero baseline. If the "occluded" value were instead set
   to the image's own mean pixel intensity, would you expect the heatmap to change much for
   this dataset? What property of the input would need to be true for the choice of baseline
   *not* to matter?

## Further Reading

- Simonyan, Vedaldi, Zisserman (2013), ["Deep Inside Convolutional Networks: Visualising
  Image Classification Models and Saliency Maps"](https://arxiv.org/abs/1312.6034) — vanilla
  gradient saliency.
- Springenberg et al. (2014), ["Striving for Simplicity: The All Convolutional
  Net"](https://arxiv.org/abs/1412.6806) — guided backprop.
- Selvaraju et al. (2017), ["Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization"](https://arxiv.org/abs/1610.02391).
- Zeiler & Fergus (2013), ["Visualizing and Understanding Convolutional
  Networks"](https://arxiv.org/abs/1311.2901) — occlusion sensitivity.
- Adebayo et al. (2018), ["Sanity Checks for Saliency
  Maps"](https://arxiv.org/abs/1810.03292) — the weight-randomization test reproduced here.
- Kindermans et al. (2019), ["The (Un)reliability of Saliency
  Methods"](https://arxiv.org/abs/1711.00867) — further failure modes of gradient-based
  explanations.

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.20_Model_Interpretability_for_Deep_Nets&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
