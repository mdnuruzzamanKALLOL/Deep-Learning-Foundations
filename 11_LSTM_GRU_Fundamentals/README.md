# LSTM & GRU Fundamentals

> Status: ✅ Complete

Gated recurrent cells (LSTM and GRU) derived from scratch with full BPTT, validated against
PyTorch to machine precision, and compared to vanilla RNNs on gradient flow and long-range
dependency tasks.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`11_lstm_gru_fundamentals.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Topic 10 showed that vanilla RNNs struggle with long-range dependencies: gradients shrink
exponentially as they backprop through repeated $W_{hh}^\top$ multiplications, and delayed
recall at $T=50$ collapses to ~47% accuracy. **Gated recurrent cells** address this by
introducing learned switches that control what to remember, forget, and expose.

**LSTM** (Long Short-Term Memory) maintains two state vectors:
- **Cell state** $c_t$ — a protected "memory highway" that can carry information across
  many timesteps with minimal attenuation when the forget gate is open.
- **Hidden state** $h_t$ — the visible output, filtered by an output gate.

**GRU** (Gated Recurrent Unit) is a lighter alternative: one state $h_t$ with a reset gate
(decides how much past context to ignore when computing a candidate) and an update gate
(decides how much of the old state to keep vs. replace).

Both architectures add parameters (~3–4× a vanilla RNN) but dramatically improve gradient
flow and long-range memory — the core trade-off explored in this notebook.

## Mathematical Explanation

**LSTM** (batch-first, input $x \in \mathbb{R}^{N \times T \times D}$):

$$i_t = \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i) \quad\text{(input gate)}$$
$$f_t = \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f) \quad\text{(forget gate)}$$
$$g_t = \tanh(x_t W_{xg} + h_{t-1} W_{hg} + b_g) \quad\text{(cell candidate)}$$
$$o_t = \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o) \quad\text{(output gate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t)$$

We initialize **$b_f = 1$** so the forget gate defaults to "remember" at the start of
training — a standard practice that prevents the cell from immediately discarding memory.

**GRU**:

$$r_t = \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r) \quad\text{(reset gate)}$$
$$z_t = \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z) \quad\text{(update gate)}$$
$$n_t = \tanh\!\bigl(x_t W_{xn} + b_n + r_t \odot (h_{t-1} W_{hn})\bigr) \quad\text{(candidate)}$$
$$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$

We initialize **$b_z = 1$** so the update gate defaults to carrying the previous hidden
state. PyTorch applies the reset gate to the **hidden linear transform** $h W_{hn}$, not
to $(r \odot h) W_{hn}$ — our implementation matches that convention.

**Why gates help gradient flow:** In a vanilla RNN, $dh_{t-1} = dz_t W_{hh}^\top$ at every
step. In an LSTM, the cell gradient also flows through $dc_{t-1} = dc_t \odot f_t$. When
$f_t \approx 1$ (encouraged by $b_f = 1$), information and gradients can pass through the
cell state with far less attenuation than repeated $W_{hh}^\top$ products.

## Code Implementation

All code lives in [`11_lstm_gru_fundamentals.ipynb`](11_lstm_gru_fundamentals.ipynb). It:

1. Implements `LSTMCell`, `LSTM`, `GRUCell`, and `GRU` with full BPTT (forward + backward).
2. Validates forward and backward against `nn.LSTM` and `nn.GRU` (`batch_first=True`) in float64.
3. Compares gradient-norm profiles: vanilla RNN ($W_{hh} = 0.9I$) vs LSTM on a $T=40$ impulse task.
4. Counts parameters for vanilla RNN, LSTM, and GRU at equal hidden size.
5. Trains `SeqClassifier` (vanilla / LSTM / GRU) on **delayed recall** at $T=50$ and $T=100$
   (Topic 10's failure case), 3 seeds each.

## Key Results

**PyTorch validation** (4×6×3 input, hidden 5, float64):

| Model | Forward max diff | dx max diff |
|---|---|---|
| LSTM | 5.55e-17 | 5.55e-17 |
| GRU | 9.02e-17 | 1.67e-16 |

**Gradient flow** (impulse at $t=0$, loss on $h_T$ only, $T=40$, hidden 16):

| Model | $\|dh_0\|/\|dh_{39}\|$ | Early-step $\|dh_0\|$ |
|---|---|---|
| Vanilla RNN ($W_{hh}=0.9I$) | **0.0130** | 0.01297 |
| LSTM ($b_f=1$) | **0.334** | 0.09974 (**7.7× larger**) |

The LSTM preserves ~33% of the final-step gradient magnitude at $t=0$ vs ~1.3% for vanilla.

**Parameter count** ($D=1$, $H=32$):

| Model | Parameters | vs vanilla |
|---|---|---|
| Vanilla RNN | 1,088 | 1.0× |
| LSTM | 4,352 | **4.0×** |
| GRU | 3,264 | **3.0×** |

**Delayed recall** (hidden 32, 3 seeds, label at $t=0$, noise until $t=T-1$):

| $T$ | Epochs | lr | Vanilla mean | LSTM mean | GRU mean |
|---|---|---|---|---|---|
| 50 | 100 | 0.05 | **0.512** | **0.997** | **0.862** |
| 100 | 200 | 0.01 | **0.468** | **0.655** | **0.487** |

At $T=50$, LSTM reaches ~99.7% mean test accuracy where Topic 10's vanilla RNN averaged
~47%. GRU also improves substantially (~86%). At $T=100$, even LSTM struggles without
more epochs or hidden capacity — but still beats vanilla (~66% vs ~47%).

## Common Pitfalls

1. **Forgetting forget-gate bias initialization.** Setting $b_f = 0$ (the default in many
   frameworks if you zero-init all biases) makes $f_t \approx 0.5$ at start, so the cell
   immediately forgets half its state each step. Without $b_f = 1$, delayed recall stays near
   chance (~50%) despite correct BPTT.
2. **Wrong GRU reset-gate placement.** PyTorch computes $n = \tanh(x W_{xn} + b_n + r \odot (h W_{hn}))$,
   **not** $\tanh(x W_{xn} + b_n + (r \odot h) W_{hn})$. Swapping these breaks both forward
   values and backward gradients vs `nn.GRU`.
3. **PyTorch gate weight stacking order.** `nn.LSTM` stacks input weights as
   `[W_i; W_f; W_g; W_o]` (input, forget, cell, output). `nn.GRU` stacks as
   `[W_r; W_z; W_n]` (reset, update, new). Copying weights in the wrong order silently
   fails validation.
4. **Gate saturation despite correct architecture.** Even with gates, poor bias init or
   extreme inputs can saturate sigmoids ($f_t \to 0$ or $i_t \to 0$), recreating vanishing
   gradients. Monitor gate activations during training on hard sequence tasks.
5. **Ignoring the parameter cost.** LSTM has ~4× and GRU ~3× the parameters of a vanilla
   RNN at the same hidden size. Fair comparisons should either match hidden size (more params
   for gated) or match total parameter count (smaller hidden for gated).
6. **Expecting LSTM to solve arbitrarily long sequences out of the box.** At $T=100$ with
   hidden 32 and 200 epochs, LSTM mean accuracy is only ~66%. Longer dependencies need more
   capacity, training time, or architectural additions (Topic 16 covers gradient clipping and
   deeper remedies).

## Function Reference

| Function / Class | Purpose |
|---|---|
| `LSTMCell` | Single-step LSTM with forward/backward; $b_f = 1$ by default |
| `LSTM` | Unrolls `LSTMCell` over $T$ steps with full BPTT |
| `GRUCell` | Single-step GRU (PyTorch-aligned reset gate); $b_z = 1$ by default |
| `GRU` | Unrolls `GRUCell` over $T$ steps with full BPTT |
| `stack_lstm_weights` / `stack_gru_weights` | Pack our weights into PyTorch's stacked layout for validation |
| `VanillaRNN` | Baseline tanh RNN for gradient-flow comparison |
| `grad_profile_vanilla` / `grad_profile_lstm` | $\|\partial L / \partial h_t\|$ when loss is at final step |
| `SeqClassifier` | Many-to-one classifier wrapping vanilla / LSTM / GRU |
| `make_delay_data` | Delayed-recall dataset (label at $t=0$, noise until end) |
| `train_delay` | Trains and evaluates one model type on delayed recall |

## Self-Test

1. Write the LSTM update equations for $c_t$ and $h_t$. What role does the forget gate
   $f_t$ play in the backward pass for $dc_{t-1}$?
2. Why do we initialize $b_f = 1$ for LSTM and $b_z = 1$ for GRU? What happens to delayed
   recall if $b_f = 0$?
3. PyTorch's GRU computes $n = \tanh(x W_{xn} + b_n + r \odot (h W_{hn}))$. Why is this
   different from applying the reset gate to $h$ before the matrix multiply?
4. Our gradient profile shows LSTM early-step gradient is 7.7× larger than vanilla at
   $T=40$. Which mechanism (cell-state highway vs hidden-state recurrence) primarily explains
   this, and why?
5. At $T=50$, LSTM reaches 99.7% mean accuracy vs vanilla 51.2%. At $T=100$, LSTM drops to
   65.5%. Is this primarily a vanishing-gradient problem, a capacity problem, or insufficient
   training? How would you test each hypothesis?
6. LSTM has 4× the parameters of vanilla RNN at the same hidden size. Design a fair
   comparison where total parameter counts are approximately equal. What hidden size would
   you use for LSTM if vanilla uses $H=32$?

## Further Reading

- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 10.10](https://www.deeplearningbook.org/contents/rnn.html) — LSTM and gated RNNs.
- Hochreiter, S. & Schmidhuber, J. (1997). ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf) — original LSTM paper.
- Cho, K. et al. (2014). ["Learning Phrase Representations using RNN Encoder-Decoder"](https://arxiv.org/abs/1406.1078) — GRU introduction.
- [PyTorch nn.LSTM documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — weight stacking and `batch_first`.
- [PyTorch nn.GRU documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) — reset/update gate convention.

---
[← Back to Deep Learning Foundations](../README.md)
