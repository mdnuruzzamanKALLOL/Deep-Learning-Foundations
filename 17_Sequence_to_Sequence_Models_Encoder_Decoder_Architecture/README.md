# Sequence-to-Sequence Models & Encoder-Decoder Architecture

> Status: ✅ Complete

A GRU-based encoder-decoder (with an optional attention decoder) implemented from
scratch with full backpropagation through **both** sub-networks, gradient-checked
numerically, and evaluated on a sequence-reversal task that forces the model to see the
entire input before producing the first output token.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`17_sequence_to_sequence_models_encoder_decoder_architecture.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Every previous recurrent topic (10, 11, 14) mapped a sequence to a **single** output
(classification) or worked on one sequence at a time. **Seq2seq** maps a sequence to
**another sequence**, possibly of different length or content — machine translation,
summarization, or (our toy task) reversing a list of digits.

Cho et al. (2014) proposed the original **encoder-decoder** architecture:
- **Encoder**: an RNN reads the input sequence and produces a final hidden state — the
  **context vector** — that must summarize *everything* needed downstream.
- **Decoder**: a second RNN, **initialized with that context vector**, generates the
  output sequence one token at a time, using **teacher forcing** during training (the
  true previous output token is fed as input, not the model's own guess).

The catch: squeezing an arbitrarily long input into one fixed-size vector is a
**bottleneck**. Bahdanau et al. (2015) fixed this with **attention** (Topic 14): instead
of relying solely on the context vector, the decoder recomputes a fresh, per-step
context by attending over *every* encoder hidden state. This notebook builds both
variants side by side and measures the difference honestly — including where it does
and doesn't help at small scale.

## Mathematical Explanation

**Encoder** (GRU, Topic 11's cell, reused unchanged):

$$h_t^{enc} = \mathrm{GRUCell}(x_t, h_{t-1}^{enc}), \qquad c = h_T^{enc}$$

**Decoder without attention** — initialized with the encoder's final state, conditioned
only on that fixed vector for the entire output sequence:

$$h_0^{dec} = c, \qquad h_t^{dec} = \mathrm{GRUCell}(y_{t-1}, h_{t-1}^{dec}), \qquad
\hat{y}_t = \mathrm{softmax}(h_t^{dec} W_{out} + b_{out})$$

**Decoder with attention** — same recurrence, but the output additionally depends on a
context vector recomputed at every step by attending over **all** encoder states
(Topic 14's mechanism, with the decoder hidden state as query):

$$\mathrm{ctx}_t = \sum_{i=1}^{T} \alpha_{t,i}\, h_i^{enc}, \qquad
\alpha_t = \mathrm{softmax}\!\left(\frac{h_t^{dec} \cdot h_i^{enc}}{\sqrt{d_k}}\right)$$

$$\hat{y}_t = \mathrm{softmax}\!\left([h_t^{dec}; \mathrm{ctx}_t]\, W_{out} + b_{out}\right)$$

**Backward pass:** the decoder's gradient at $t=0$ flows back into the encoder's final
hidden state exactly like an extra recurrent step ($dh_{T}^{enc} \mathrel{+}= dh_0^{dec}$).
When attention is enabled, every encoder state is reused as key *and* value at *every*
decoder step, so gradients from **all** $T$ decode steps accumulate into $dH^{enc}$
*before* backpropagating through the encoder's own time steps — a fan-in that a plain
encoder-decoder does not have.

**Teacher forcing vs free-running:** training always feeds the *true* previous token
($y_{t-1}$) to the decoder. At inference, the true token is unknown, so the model must
feed back its *own* prediction ($\hat{y}_{t-1}$) — a train/test mismatch known as
**exposure bias** (Bengio et al., 2015).

## Code Implementation

All code lives in
[`17_sequence_to_sequence_models_encoder_decoder_architecture.ipynb`](17_sequence_to_sequence_models_encoder_decoder_architecture.ipynb). It:

1. Implements `GRUCell` (Topic 11, reused verbatim) and `DotAttention` (Topic 14's
   mechanism, applied once per decode step).
2. Implements `Seq2Seq`, a full encoder-decoder with `use_attention` on/off, including
   the complete forward and backward pass through both sub-networks.
3. Validates gradients via **numerical gradient checking** (finite differences) since no
   single built-in PyTorch module matches this custom architecture.
4. Trains on a **digit-reversal task** ($T \in \{4, 8, 12, 16\}$) and compares
   free-running token/sequence accuracy with vs without attention.
5. Visualizes a trained attention decoder's alignment on one reversal example.
6. Quantifies **exposure bias**: teacher-forced vs free-running evaluation accuracy at
   $T=15$.

## Key Results

**Gradient check** (finite differences, $\epsilon = 10^{-5}$, 6 parameter groups):

| Configuration | Max relative error |
|---|---|
| No attention | 1.58e-05 |
| With attention | 1.10e-06 |

Both pass at the precision expected of finite differences, confirming correct
backpropagation through the encoder, the decoder, and the attention mechanism.

**Reversal task — free-running token/sequence accuracy (2-seed mean):**

| $T$ | No-attn token | No-attn seq | Attn token | Attn seq |
|---|---|---|---|---|
| 4 | 0.997 | 0.988 | 0.992 | 0.979 |
| 8 | 0.911 | 0.738 | 0.870 | 0.598 |
| 12 | 0.680 | 0.256 | **0.740** | 0.247 |
| 16 | 0.433 | 0.013 | **0.528** | 0.031 |

Honest reading: at $T=4$ both are essentially tied and near-perfect. At $T=8$, attention
is actually *slightly worse* — the bottleneck isn't binding yet, and the extra attention
parameters add optimization overhead without benefit. Only at $T=12$ and $T=16$, where a
32-dimensional vector genuinely struggles to hold the whole sequence, does attention's
**token** accuracy pull ahead. Exact-sequence-match accuracy is far noisier (one wrong
token anywhere fails the whole sequence) and shows no clean trend at this scale — we
report it rather than hiding it.

**Attention alignment** (trained at $T=10$, one example, prediction exactly correct):

```
Argmax attended encoder position per decode step: [9 8 7 6 5 4 3 2 1 1]
Expected under perfect reversal alignment (T-1-t): [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

The argmax nearly perfectly follows the theoretically correct anti-diagonal alignment —
but the raw weights stay close to **uniform** ($\approx 0.10$ across 10 positions), unlike
Topic 14's marker task where weights peaked near 1.0. Reversal needs information from
*every* position (nothing to filter out), so the decoder's own recurrent state already
carries most of what it needs; attention contributes a small but consistently correct
signal on top.

**Exposure bias** ($T=15$, hidden 48):

| Model | Teacher-forced acc | Free-running acc | Free-running seq acc |
|---|---|---|---|
| No attention | 0.974 | 0.901 | 0.654 |
| Attention | 0.964 | 0.882 | 0.568 |

Both models show the same ~7–9 point gap between cheating with ground-truth history
(teacher-forced) and realistic autoregressive decoding (free-running) — a reliable
effect that does not depend on architecture, only on the train/test procedure mismatch.

## Common Pitfalls

1. **Forgetting that the decoder's initial state IS the encoder's final state.**
   Without this connection, encoder and decoder are two unrelated RNNs — the encoder's
   entire purpose is to hand off information through $h_0^{dec} = h_T^{enc}$.
2. **Numpy's fancy-index `+=` silently drops duplicate indices.** Embedding gradients
   accumulated with `grads['embed'][ids] += dx` lose contributions when two rows in a
   batch share the same token id at the same timestep. `np.add.at(grads['embed'], ids, dx)`
   is required — this bug alone caused gradient checks to fail with ~95% relative error
   until fixed.
3. **Expecting attention to always win.** Our $T=8$ result shows attention performing
   *worse* than the plain context vector — attention isn't "free"; it adds parameters
   and an extra optimization surface that only pays off once the fixed-size bottleneck
   actually binds.
4. **Confusing sharp attention weights with useful attention.** Our reversal task's
   learned weights stay close to uniform even though the argmax alignment is correct —
   contrast with Topic 14's marker task, where sharp weights were necessary because most
   positions were irrelevant padding. Sharpness depends on the task, not the mechanism.
5. **Training only with teacher forcing and evaluating only with teacher forcing.**
   This hides exposure bias entirely. Always evaluate with free-running (autoregressive)
   decoding to see realistic inference-time performance.
6. **Attributing all encoder-state gradient flow only to the final timestep.** With
   attention, every encoder position receives gradients from *every* decoder step (as
   key AND value), not just from being the initial decoder state — the backward pass
   must accumulate across the full $T \times T$ attention grid before backpropagating
   through encoder time.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `GRUCell` | Single-step GRU (Topic 11), shared by encoder and decoder |
| `DotAttention` | Query-per-decode-step dot-product attention over all encoder states |
| `Seq2Seq` | Full encoder-decoder; `use_attention` toggles the attention decoder |
| `numerical_gradient_check` | Finite-difference validation (no matching PyTorch module exists) |
| `make_reversal_data` | Synthetic task: reverse a sequence of digits |
| `decode_free_running` | Realistic inference: feed the model's own previous prediction |
| `teacher_forced_accuracy` | Cheating eval: feed ground-truth previous token at every step |
| `train_seq2seq` | Trains one configuration and returns free-running test accuracy |

## Self-Test

1. Write the encoder-decoder handoff equation ($h_0^{dec} = ?$). Why does this make the
   context vector a bottleneck for long input sequences?
2. Our gradient check failed at ~95% relative error before a one-line fix. What was the
   bug, and why does it only manifest with duplicate token ids in the same batch/timestep?
3. At $T=8$, attention scored *lower* than the plain context vector. Give a plausible
   explanation for why more mechanism doesn't automatically mean better performance.
4. Our attention weights on the reversal task stay close to uniform (~0.10) even though
   predictions are 100% correct and the argmax alignment is exactly right. What does this
   tell you about the relationship between attention weight *sharpness* and attention
   *usefulness*?
5. Define exposure bias in your own words. Why is the ~7–9 point gap between
   teacher-forced and free-running accuracy independent of whether attention is used?
6. With attention, why must every encoder hidden state receive gradient contributions
   from *all* $T$ decoder steps, rather than just from the timestep where it "matches"
   the correct output? (Hint: consider what the *forward* pass computes at every step.)

## Further Reading

- Cho, K. et al. (2014). ["Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"](https://arxiv.org/abs/1406.1078) — original encoder-decoder architecture.
- Sutskever, I., Vinyals, O., Le, Q. (2014). ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) — seq2seq with LSTMs, input reversal trick.
- Bahdanau, D., Cho, K., Bengio, Y. (2015). ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473) — attention fixes the context bottleneck.
- Bengio, S. et al. (2015). ["Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"](https://arxiv.org/abs/1506.03099) — exposure bias and mitigation.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 10.4](https://www.deeplearningbook.org/contents/rnn.html) — encoder-decoder sequence-to-sequence architectures.

---
[← Back to Deep Learning Foundations](../README.md)
