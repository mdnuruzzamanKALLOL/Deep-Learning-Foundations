# Attention Mechanism Basics

> Status: ✅ Complete

Scaled dot-product attention and single-head self-attention — implemented from scratch with
full backpropagation, validated against PyTorch, and tested on a sequence classification
task where a learnable query must find a marked token.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`14_attention_mechanism_basics.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Topics 10–11 showed that vanilla RNNs struggle to carry information across many timesteps.
**Attention** offers a different mechanism: instead of compressing an entire sequence into
one fixed-size hidden vector, each query position can **directly read** from any key/value
position via a softmax-weighted sum.

Intuition:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?" (one key per input position)
- **Value (V):** "What information do I provide if selected?"

The softmax weights form a probability distribution over positions — an **alignment**
that can be inspected. This is the core mechanism behind Transformers (Topic 17): replace
recurrence with attention over all positions in parallel.

Two variants appear in the literature:
- **Bahdanau (additive) attention** — small feedforward network scores query–key pairs
- **Scaled dot-product attention** (Vaswani et al., 2017) — $QK^\top / \sqrt{d_k}$, faster
  and used in Transformers

This notebook implements scaled dot-product attention and single-head self-attention from
scratch.

## Mathematical Explanation

**Scaled dot-product attention** (batch-first $Q \in \mathbb{R}^{N \times T_q \times d_k}$):

$$\mathrm{scores} = \frac{QK^\top}{\sqrt{d_k}}, \qquad
\alpha = \mathrm{softmax}(\mathrm{scores}), \qquad
\mathrm{Attention}(Q,K,V) = \alpha V$$

**Self-attention:** the same sequence produces $Q = XW_q$, $K = XW_k$, $V = XW_v$, allowing
each position to attend to all others.

**Causal (autoregressive) mask:** set $\mathrm{scores}_{ij} = -\infty$ for $j > i$ so
position $i$ cannot attend to the future. Required for GPT-style language modeling.

**Backward pass highlights:**
- Softmax: $d\,\mathrm{scores} = \alpha \odot (d\alpha - \sum_j \alpha_j\, d\alpha_j)$
- Through scale: $dQ = d\,\mathrm{scores}\, K / \sqrt{d_k}$

## Code Implementation

All code lives in [`14_attention_mechanism_basics.ipynb`](14_attention_mechanism_basics.ipynb). It:

1. Implements `ScaledDotProductAttention` and `SelfAttention` with full backpropagation.
2. Validates forward and backward against PyTorch `scaled_dot_product_attention` (float64).
3. Demonstrates causal masking blocking future attention mass.
4. Trains an `AttentionClassifier` (learnable query + key/value projections) on a **marker
   digit task**: all positions are padding token 9 except one random position holding the
   class digit 0–8.
5. Compares attention pooling vs mean pooling vs vanilla RNN at $T \in \{10, 20, 40\}$.
6. Visualizes learned attention weights peaking at the marked position.

## Key Results

**PyTorch validation** ($N=2$, $T=5$, $d=8$, float64):

| | Max diff |
|---|---|
| Forward | 2.22e-16 |
| dQ | 3.33e-16 |
| dK | 4.44e-16 |
| dV | 2.22e-16 |

**Causal mask** (position 0 attending to future positions):

| Mask | Future mass |
|---|---|
| None | **0.066** |
| Lower-triangular | **0.000** |

**Marker-digit task** (padding token 9, one marked digit 0–8, 3 seeds mean):

| $T$ | Attention | Mean pool | Vanilla RNN | Attn weight @ marker |
|---|---|---|---|---|
| 10 | **1.000** | **1.000** | **0.189** | **0.999** |
| 20 | **1.000** | **1.000** | **0.186** | **1.000** |
| 40 | **1.000** | **1.000** | **0.124** | **1.000** |

Attention and mean pooling both reach 100% accuracy here because constant padding makes
the mean embedding a fixed offset plus a small class signal. The critical difference is
**interpretability**: attention places weight **1.0** at the marker vs a uniform **0.050**
baseline at $T=20$. The vanilla RNN stays near chance (~12–19%) — it cannot reliably
find the marked digit buried among 39 padding tokens, echoing Topic 10's long-range failure.

## Common Pitfalls

1. **Forgetting the $\sqrt{d_k}$ scale.** Without it, dot products grow with dimension,
   pushing softmax into saturation (near one-hot or uniform), killing gradient flow.
2. **Omitting the causal mask in autoregressive models.** Without masking, position $i$
   can peek at future tokens — leakage that inflates training metrics and breaks generation.
3. **Assuming attention weights are always interpretable.** Weights reflect $QK^\top$ in
   the *current* parameter space; they can be diffuse early in training or with ambiguous
   inputs (multiple equally valid positions).
4. **Confusing attention with memoryless independence.** Attention reads from all positions
   in one step, but still needs learned $W_q, W_k, W_v$ to be useful — random projections
   give near-uniform weights.
5. **Expecting mean pooling to fail whenever attention works.** With a **constant**
   distractor token, mean pooling can be linearly separable. Attention's advantage is
   direct position selection and interpretability; with **random** distractors, both can
   struggle without a clearer signal.
6. **Using full softmax attention on very long sequences without approximation.** Attention
   is $O(T^2)$ in memory and compute — the motivation for sparse, linear, and local
   attention variants in production Transformers.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `ScaledDotProductAttention` | Core $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ with optional mask |
| `SelfAttention` | Projects input to Q/K/V, applies attention, output projection $W_o$ |
| `AttentionClassifier` | Learnable query attends over sequence; classifies marked digit |
| `MeanPoolClassifier` | Baseline: mean embedding + linear head |
| `VanillaRNNClassifier` | Baseline: tanh RNN many-to-one (Topic 10 style) |
| `make_marker_data` | Synthetic task: padding token 9, one position holds class label |
| `Adam` | Optimizer for training (Topic 05) |

## Self-Test

1. Write the scaled dot-product attention formula. Why divide by $\sqrt{d_k}$?
2. In our backward pass, why is the softmax gradient $d\,\mathrm{scores} = \alpha \odot
   (d\alpha - \sum \alpha_j d\alpha_j)$ rather than just $d\alpha$?
3. Our causal mask zeroes future attention mass from 0.066 to 0.000. What would happen
   if you trained a language model *without* this mask?
4. On the marker task, mean pooling reaches 100% accuracy alongside attention. What
   specific quantity does the attention weight at the marker (1.0 vs 0.05 uniform) tell
   you that accuracy alone does not?
5. Why does the vanilla RNN stay near ~12–19% accuracy while attention reaches 100% at
   $T=40$? Connect this to Topic 10's vanishing-gradient discussion.
6. Self-attention computes $Q = XW_q$, $K = XW_k$, $V = XW_v$ from the *same* input $X$.
   How does this differ from encoder–decoder (cross) attention in seq2seq (Topic 17)?

## Further Reading

- Vaswani, A. et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Transformers and scaled dot-product attention.
- Bahdanau, D., Cho, K., Bengio, Y. (2015). ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) — additive attention for seq2seq.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 12.4](https://www.deeplearningbook.org/contents/seq_models.html) — attention in sequence models.
- [PyTorch scaled_dot_product_attention docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

---
[← Back to Deep Learning Foundations](../README.md)
