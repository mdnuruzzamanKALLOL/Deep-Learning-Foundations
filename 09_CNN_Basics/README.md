# CNN Basics

> Status: ✅ Complete

Convolution, max/average pooling, and feature maps — implemented from scratch with full
backward passes, validated against PyTorch to machine precision, and tested on sklearn
digits (8×8 grayscale).

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`09_cnn_basics.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

A fully connected layer on a 28×28 image has 784 inputs per neuron — each weight ties
one pixel to one output. Shift the image by one pixel and every input index changes;
the MLP must re-learn the pattern from scratch at each position.

A **convolution layer** applies the same small filter everywhere across the image.
Each filter detects a local pattern (edge, corner, stroke segment) regardless of where
it appears. This **parameter sharing** slashes the parameter count and builds in
translation equivariance: shift the input, and the feature map shifts the same way.

**Pooling** (max or average) downsamples feature maps, reducing spatial size and adding
tolerance to small translations and distortions within each pooling window.

Together, conv + pool + ReLU stacks form the core building block of modern vision models.

## Mathematical Explanation

**2D Convolution** (input $X \in \mathbb{R}^{N \times C_{in} \times H \times W}$,
filters $W \in \mathbb{R}^{C_{out} \times C_{in} \times k_H \times k_W}$):

$$Y[n,c,i,j] = b_c + \sum_{c',u,v} W[c,c',u,v]\; X[n,c', iS+u, jS+v]$$

(with padding applied before the sum). The same $W[c,c',:,:]$ is used at every output
location $(i,j)$.

**Output spatial size:**

$$H_{out} = \left\lfloor \frac{H + 2P - k_H}{S} \right\rfloor + 1$$

**Max pooling** over a $k \times k$ window: $Y = \max_{u,v \in \text{window}} X$.
Gradient flows only to the argmax position (all other positions get zero).

**Average pooling**: $Y = \text{mean}_{u,v \in \text{window}} X$. Gradient is split
equally ($1/k^2$) across all positions in the window.

## Code Implementation

All code lives in [`09_cnn_basics.ipynb`](09_cnn_basics.ipynb). It:

1. Implements `Conv2d`, `MaxPool2d`, and `AvgPool2d` via `im2col` / `col2im` with full
   analytical backward passes.
2. Validates forward and backward against PyTorch (`nn.Conv2d`, `nn.MaxPool2d`,
   `nn.AvgPool2d`) in float64.
3. Demonstrates output-size arithmetic and **Sobel edge-detection feature maps** on a
   digit image (no training required).
4. Compares parameter counts for shared conv filters vs position-specific dense layers.
5. Trains a tiny CNN and a flattened MLP on sklearn digits (10 classes, 8×8 images).
6. Stress-tests **translation robustness** by shifting test digits 1–2 pixels.

## Key Results

**PyTorch validation** (2×3×8×8 input, float64):

| | Forward max diff | Backward dx max diff |
|---|---|---|
| Conv2d | 0.00e+00 | 1.11e-16 |
| MaxPool2d | — | 0.00e+00 |
| AvgPool2d | — | 0.00e+00 |

**Output shapes** (8×8 input, 3×3 conv pad=1 stride=1 → 8×8; then 2×2 max-pool → 4×4).

**Parameter sharing** (first conv layer, 8 filters, 3×3 kernel, 1 input channel):

| Layer type | Parameters |
|---|---|
| Conv2d (shared filters) | **80** |
| Position-specific dense (×8 filters) | 4,608 |
| Sharing ratio | **57.6×** fewer params |

**Digit classification** (40 epochs, batch 32, train/test split 75/25):

| Model | Final test acc | Parameters |
|---|---|---|
| SimpleCNN (Conv→ReLU→Pool ×2 → linear) | **0.976** | 1,864 |
| Flattened MLP (64→64→10) | 0.973 | 4,736 |

Both reach ~97% on this easy task; the CNN does it with **2.5× fewer parameters**.

**Translation robustness** (test digits shifted by integer pixels):

| Shift | CNN | MLP |
|---|---|---|
| None | 0.976 | 0.973 |
| (+0, +1) | **0.656** | 0.411 |
| (+0, +2) | **0.404** | 0.109 |
| (+1, +1) | **0.184** | 0.147 |
| (+2, +0) | 0.122 | 0.116 |

After a 1-pixel horizontal shift, the CNN retains 66% accuracy vs the MLP's 41%. After
2 pixels, the gap is 40% vs 11%. The MLP treats pixel index as a feature identity;
shifting scrambles that mapping entirely.

## Common Pitfalls

1. **Forgetting padding when you want same spatial size.** A 3×3 conv with stride 1 and
   no padding shrinks the output by 2 pixels per side. Use `padding=1` for "same" size
   (when stride=1 and kernel is odd).
2. **Confusing PyTorch channel order.** PyTorch uses NCHW (batch, channels, height, width).
   Some frameworks and textbooks use NHWC. Transposing silently produces wrong results
   when validating against `nn.Conv2d`.
3. **Reusing the same pooling layer instance twice in one forward pass.** Each pool layer
   stores its argmax cache for backward. Calling `forward` twice on the same object
   overwrites the cache — use separate `MaxPool2d` instances (e.g. `pool1`, `pool2`).
4. **Updating weights before finishing the full backward pass.** In a manual training loop,
   compute all gradients first, then apply updates. Updating `Wfc` before computing
   `dflat = dlogits @ Wfc.T` uses the already-modified weights and corrupts conv gradients.
5. **Assuming CNNs always beat MLPs on tiny images.** On fixed 8×8 digits with no
   translation noise, a flattened MLP matches CNN accuracy. CNN advantages show up with
   larger images, deeper stacks, and spatial perturbations — not automatically on every task.
6. **Letting global RNG state affect training reproducibility.** Validation cells that call
   `np.random.randn` change the batch-shuffle sequence in later training cells. Use
   dedicated `np.random.RandomState(seed)` objects for validation and training separately.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `im2col` / `col2im` | Unroll image patches into columns for conv matmul; scatter gradients back |
| `Conv2d` | 2D convolution with He init, forward + backward (dX, dW, db) |
| `MaxPool2d` | Max pooling with argmax cache for backward |
| `AvgPool2d` | Average pooling with uniform gradient split |
| `SimpleCNN` | Two-layer conv network for 10-class digit classification |
| `MLPFlat` | Flattened baseline MLP for comparison |
| `shift_images` | Integer pixel-shift augmentation for robustness testing |

## Self-Test

1. For an 8×8 input, 3×3 convolution with `padding=1` and `stride=1`, what is the output
   spatial size? What about after a 2×2 max-pool with `stride=2`?
2. A conv layer has 8 output channels, 1 input channel, and a 3×3 kernel. How many
   learnable parameters does it have (including biases)? How many would a position-specific
   dense layer need for the same 8 feature maps on an 8×8 grid?
3. In max-pool backward, why does the gradient go only to the argmax position within each
   window? What would happen if you distributed the gradient equally (as in average pool)?
4. Our CNN uses ~1,864 parameters vs the MLP's 4,736, yet both reach ~97% test accuracy
   on unshifted digits. Under what conditions would you expect the CNN's advantage to grow?
5. After a (+0, +1) pixel shift, CNN test accuracy drops from 0.976 to 0.656 while the
   MLP drops to 0.411. Explain mechanistically why the CNN degrades more gracefully.
6. The validation cell uses `im2col` to implement convolution as a single large matrix
   multiply. What is the main memory drawback of `im2col` compared to a direct sliding-window
   loop, and how do modern frameworks mitigate it?

## Further Reading

- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Chapter 9](https://www.deeplearningbook.org/contents/convnets.html) — convolutional networks from first principles.
- CS231n Stanford, ["ConvNet Architecture Summary"](https://cs231n.github.io/convolutional-networks/) — output-size formulas, parameter counting, and receptive fields.
- [PyTorch nn.Conv2d documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — `padding`, `stride`, `dilation`, and `groups` modes.
- LeCun, Y. et al. (1998). ["Gradient-Based Learning Applied to Document Recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — the original LeNet and the case for conv layers on digit images.

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.09_CNN_Basics&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
