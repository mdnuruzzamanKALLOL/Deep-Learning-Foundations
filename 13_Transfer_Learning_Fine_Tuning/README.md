# Transfer Learning & Fine-Tuning

> Status: ✅ Complete

Reusing pretrained CNN representations for a new classification task — implemented from
scratch (Topic 09 CNN backbone), with frozen-backbone feature extraction, fine-tuning with
discriminative learning rates, and an honest comparison when the source domain does not
match the target.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`13_transfer_learning_fine_tuning.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Training a deep CNN from scratch requires large labeled datasets and significant compute.
**Transfer learning** reuses weights learned on a **source task** (where data is abundant)
to jump-start learning on a **target task** (where labels are scarce).

The standard recipe:
1. **Pretrain** a backbone (conv layers) on the source task.
2. **Replace the classification head** for the target task's number of classes.
3. Choose a training strategy:
   - **Feature extraction** — freeze the backbone, train only the new head (fast, safe with tiny data).
   - **Fine-tuning** — unfreeze the backbone with a **smaller learning rate**, train head + backbone jointly.

Low-level conv filters (edges, strokes) transfer across many vision tasks. High-level
features and the final classifier are task-specific and must be replaced or retrained.

Transfer is not guaranteed: if the source task is unrelated to the target, pretrained
weights can **hurt** performance (negative transfer).

## Mathematical Explanation

Split parameters into backbone $\theta_b$ (conv layers) and head $\theta_h$ (final linear layer).

**Feature extraction** (frozen backbone):

$$\theta_b \leftarrow \text{fixed}, \qquad \theta_h \leftarrow \theta_h - \eta \nabla_{\theta_h} \mathcal{L}$$

Gradients do not flow into $\theta_b$ (or equivalently, updates are masked to zero).

**Fine-tuning** with discriminative learning rates:

$$\theta_h \leftarrow \theta_h - \eta \nabla_{\theta_h} \mathcal{L}, \qquad
\theta_b \leftarrow \theta_b - \eta_b \nabla_{\theta_b} \mathcal{L}, \quad \eta_b \ll \eta$$

Typical ratio: $\eta_b / \eta \in [0.01, 0.2]$. A large $\eta_b$ can **destroy** useful
pretrained features (catastrophic forgetting of the source representation).

**Linear probe:** extract features $z = f_{\theta_b}(x)$ with frozen $\theta_b$, train a
linear classifier on $z$. This isolates representation quality from end-to-end optimization.

**Weight transfer:** copy $\theta_b^\text{source} \to \theta_b^\text{target}$; initialize
$\theta_h$ randomly for the new class count $C_\text{target}$.

## Code Implementation

All code lives in [`13_transfer_learning_fine_tuning.ipynb`](13_transfer_learning_fine_tuning.ipynb). It:

1. Implements `DigitCNN` (Topic 09 backbone) with `copy_backbone_from`, freeze flags, and discriminative LRs.
2. Validates forward pass against PyTorch and confirms zero conv-weight change when frozen.
3. Pretrains **Source A** (10-class all digits) and **Source B** (5-class digits 0–4 only).
4. Runs few-shot target classification on **digits 5–9** (relabeled 0–4) at $n \in \{3,5,10\}$ samples/class.
5. Compares scratch, frozen 10-class transfer, fine-tuning, and wrong-domain (0–4) transfer.
6. Reports linear-probe accuracy on random vs pretrained features.

## Key Results

**PyTorch validation** (3×1×8×8 input, float64):

| Check | Max diff |
|---|---|
| Forward vs PyTorch | 6.66e-16 |
| Conv weight change when frozen | **0.00e+00** |

**Source pretraining:**

| Model | Test accuracy |
|---|---|
| Source A (10-class all digits) | **0.971** |
| Source B (0–4 only) | **0.965** |

**Few-shot target task** (digits 5–9, 3 seeds mean):

| $n$ / class | Scratch | Frozen (10-cls) | Fine-tune (10-cls) | Frozen (0–4 only) |
|---|---|---|---|---|
| 3 | **0.771** | **0.918** | **0.927** | 0.723 |
| 5 | **0.863** | **0.936** | **0.939** | 0.790 |
| 10 | **0.918** | **0.978** | **0.964** | 0.830 |

At $n=3$/class, 10-class frozen transfer beats scratch by **+0.147** accuracy. Wrong-domain
(0–4) frozen transfer at $n=10$ **underperforms** scratch by **0.088** — negative transfer.

**Linear probe** at $n=3$/class (3 seeds):

| Features | Test accuracy |
|---|---|
| Random untrained backbone | 0.789 |
| **10-class pretrained backbone** | **0.954** |
| End-to-end scratch CNN | 0.771 |
| Frozen backbone + trained head | 0.918 |

Pretrained features alone (linear probe) outperform end-to-end scratch — the backbone
representation is the main asset being transferred.

## Common Pitfalls

1. **Fine-tuning the entire network with the same LR as the head.** A large LR on pretrained
   conv layers can erase useful filters in a few steps. Use $\eta_b \ll \eta$ (we use 0.05 vs 0.5).
2. **Assuming transfer always helps.** Source B (0–4 pretrain) for target 5–9 underperforms
   scratch at $n=10$. Source and target must be related; verify on a validation set.
3. **Forgetting to replace the classification head.** Source 10-class head weights have the
   wrong shape and semantics for a 5-class target — copy only backbone (conv) weights.
4. **Not actually freezing the backbone.** Setting `train_backbone=False` must zero out conv
   gradient updates; we verify max weight change = 0.00e+00.
5. **Evaluating with too many target labels to see transfer benefit.** With 50+ samples per
   class on 8×8 digits, scratch CNNs catch up — transfer shines in the **low-label regime**.
6. **Confusing feature extraction with fine-tuning.** Feature extraction is faster and safer
   with very few labels; fine-tuning can help when you have enough target data to adapt
   low-level filters without destroying them.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `DigitCNN` | 2-layer CNN with configurable class count, freeze flags, backbone copy |
| `copy_backbone_from` | Copy conv1/conv2 weights from a pretrained model |
| `train_backbone` / `train_head` | Control which parameters receive gradient updates |
| `extract_features` | Return flattened conv feature vector (for linear probe) |
| `backbone_snapshot` | Save conv weights to verify freezing |
| `train_model` | Mini-batch training loop with optional `lr_backbone` |
| `subsample_per_class` | Draw $n$ labeled examples per class for few-shot experiments |

## Self-Test

1. Why do we copy conv-layer weights but reinitialize the final linear head when transferring
   from 10 classes to 5?
2. In feature extraction mode, which parameters receive gradients? Write the update equations.
3. Our fine-tuning uses `lr=0.5` for the head and `lr_backbone=0.05` for conv layers. What
   would you expect if both used `lr=0.5`?
4. Source B (0–4 pretrain) gives frozen target accuracy 0.830 at $n=10$, below scratch
   (0.918). Why might pretraining on a *subset* of digit classes fail to help on the
   *complement* classes?
5. The linear probe on 10-class pretrained features (0.954) beats end-to-end scratch (0.771)
   at $n=3$/class. What does this tell you about where the gain from transfer comes from?
6. How would you extend this notebook's setup to simulate ImageNet-pretrained features without
   downloading ImageNet (conceptually — what would the source task need)?

## Further Reading

- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 15.2](https://www.deeplearningbook.org/contents/representation.html) — transfer and multi-task learning.
- Yosinski, J. et al. (2014). ["How Transferable Are Features in Deep Neural Networks?"](https://arxiv.org/abs/1411.1792) — layer-wise transferability analysis.
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — feature extraction vs fine-tuning in practice.
- Kornblith, S. et al. (2019). ["Do Better ImageNet Models Transfer Better?"](https://arxiv.org/abs/1805.08974) — correlation between source and target performance.

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.13_Transfer_Learning_Fine_Tuning&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
