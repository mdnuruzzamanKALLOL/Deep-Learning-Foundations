# RNN Basics

> Status: ✅ Complete

Vanilla recurrent networks and backpropagation through time (BPTT) — implemented from
scratch, validated against PyTorch to machine precision, and stress-tested on sequence
tasks that expose temporal structure and the vanishing-gradient problem.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`10_rnn_basics.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

A feedforward network treats each input independently. A **recurrent neural network (RNN)**
maintains a hidden state $h_t$ that is updated at every timestep, giving the network a
form of memory across time.

At each step the RNN reads an input $x_t$, combines it with the previous hidden state
$h_{t-1}$, and produces a new hidden state $h_t$. The **same weights** are reused at
every timestep — parameter sharing across time, analogous to convolution sharing weights
across space.

To train an RNN, we **unroll** it over $T$ timesteps and apply backpropagation — called
**backpropagation through time (BPTT)**. The gradient at early timesteps must flow through
many repeated multiplications by $W_{hh}^\top$ and $\tanh'(z)$. When those factors shrink
the signal, early timesteps receive vanishingly small updates — the **vanishing gradient
problem**. This is why vanilla RNNs struggle with long-range dependencies, motivating
LSTM and GRU (Topic 11).

## Mathematical Explanation

**Vanilla RNN (batch-first, input $x \in \mathbb{R}^{N \times T \times D}$):**

$$z_t = x_t W_{xh} + h_{t-1} W_{hh} + b, \qquad h_t = \tanh(z_t)$$

**Many-to-one classifier** (sequence label from final hidden state):

$$\hat{y} = \text{softmax}(h_T W_{hy} + b_y)$$

**BPTT** (loss $L$ depends on outputs at one or more timesteps): starting from
$dh_T = \partial L / \partial h_T$, for $t = T-1, \ldots, 0$:

$$dh_t \mathrel{+}= \frac{\partial L}{\partial h_t}, \quad
dz_t = dh_t \odot \tanh'(z_t), \quad
dh_{t-1} = dz_t W_{hh}^\top$$

The repeated application of $W_{hh}^\top$ across timesteps is the mechanism behind
vanishing (spectral radius $< 1$) or exploding (spectral radius $> 1$) gradients.

## Code Implementation

All code lives in [`10_rnn_basics.ipynb`](10_rnn_basics.ipynb). It:

1. Implements `VanillaRNN` with full BPTT (forward + backward through time).
2. Validates forward and backward against `nn.RNN(batch_first=True)` in float64.
3. Measures $\| \partial L / \partial h_t \|$ across timesteps when loss is at $t=T-1$.
4. Trains an `RNNClassifier` on a **temporal pattern task** (high values in first vs
   second half of sequence), compared to a flattened MLP.
5. Destroys temporal order by shuffling timesteps — both models collapse to ~chance.
6. Stress-tests **delayed recall** (label encoded at $t=0$, noise until $t=T-1$) at
   $T \in \{10, 50, 100\}$.

## Key Results

**PyTorch validation** (4×6×3 input, hidden 5, float64):

| | Max diff |
|---|---|
| Forward | 1.11e-16 |
| dx | 1.11e-16 |
| dW_xh | 1.78e-15 |
| dW_hh | 6.66e-16 |
| db | 8.88e-16 |

**Vanishing gradients** (impulse at $t=0$, loss on $h_T$ only, $T=40$):

| $W_{hh}$ | $\|dh\|$ at $t=0$ | $\|dh\|$ at $t=39$ | Ratio $t0/T$ |
|---|---|---|---|
| $0.9\,I$ | 0.01297 | 1.000 | **0.0130** |
| $0.5\,I$ | — | — | **1.78e-12** |

With $W_{hh} = 0.5I$, the gradient at $t=0$ is ~$10^{12}$ times smaller than at $t=T-1$.

**Temporal pattern classification** ($T=20$, first-half vs second-half, 60 epochs):

| Model | Final test accuracy |
|---|---|
| RNN (hidden 32) | **1.000** |
| Flattened MLP (64 hidden) | **1.000** |

Both solve this easy task; the difference appears when order is destroyed:

| Condition | RNN | MLP |
|---|---|---|
| Original order | 1.000 | 1.000 |
| **Shuffled timesteps** | **0.445** | 0.520 |

Both drop to ~chance (~0.5), confirming the label depends on *where* in the sequence
high values appear, not on marginal statistics alone.

**Delayed recall** (hidden 16, 3 seeds):

| Sequence length $T$ | Mean test accuracy |
|---|---|
| 10 | **1.000** |
| 50 | **0.472** |
| 100 | **0.538** |

At $T=50$, mean accuracy falls to ~47% (near chance) — vanilla RNNs cannot reliably
carry a bit of information across 50 noisy timesteps with only 16 hidden units. This
motivates gated architectures in Topic 11.

## Common Pitfalls

1. **Confusing PyTorch weight layout.** `nn.RNN` stores `weight_ih` as $(H, D)$ and
   `weight_hh` as $(H, H)$. Our convention is $W_{xh}$ as $(D, H)$ with `x @ W_xh`.
   Transposing silently when validating is essential.
2. **Using `batch_first=False` by default in PyTorch.** PyTorch's default input shape is
   $(T, N, D)$; we use $(N, T, D)$ throughout. Mixing conventions reshapes timesteps and
   batches incorrectly.
3. **Assuming BPTT stores all intermediate activations for free.** Full BPTT requires
   caching every $z_t$ and $h_{t-1}$ during forward — memory cost is $O(T \times H)$ per
   sample. Truncated BPTT (Topic 11+) limits backprop to the last $K$ steps to save memory.
4. **Expecting vanilla RNNs to learn very long dependencies.** Even when training accuracy
   looks fine at short $T$, increasing sequence length exposes vanishing gradients. Our
   $T=50$ delayed-recall task drops to 47% mean accuracy.
5. **Applying loss at every timestep vs only at the end.** Many-to-one classification
   backprops only through the final $h_T$. If you forget to zero `dhs[:, :T-1]`, you
   inject spurious gradients at every step.
6. **Updating weights before completing the full BPTT backward pass.** As with CNNs, compute
   all gradients (`dW_xh`, `dW_hh`, `dW_hy`) first, then apply learning-rate scaling once.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `VanillaRNN` | Batch-first tanh RNN with forward unroll and full BPTT backward |
| `RNNClassifier` | Many-to-one sequence classifier using final hidden state |
| `MLPSeq` | Flattened-sequence MLP baseline for the same tasks |
| `make_pattern_data` | Synthetic first-half vs second-half temporal pattern dataset |
| `make_delay_data` | Delayed-recall dataset (label at $t=0$, noise until end) |
| `grad_norm_profile` | Measures $\|\partial L / \partial h_t\|$ when loss is at final step |
| `shuffle_time` | Permutes timesteps within each sequence (destroys temporal order) |

## Self-Test

1. Write the vanilla RNN update equations for $z_t$ and $h_t$. What are the shapes of
   $W_{xh}$, $W_{hh}$, and $h_t$ for input dimension $D$ and hidden size $H$?
2. In BPTT, why does $dh_{t-1} = dz_t W_{hh}^\top$ appear at every backward step? What
   happens to $\|dh_t\|$ if the largest eigenvalue of $W_{hh}$ is 0.5?
3. Our PyTorch validation transposes `weight_ih` when copying weights. Why is this needed
   given PyTorch's `(H, D)` storage vs our `(D, H)` convention?
4. On the temporal pattern task, both RNN and MLP reach 100% test accuracy. After shuffling
   timesteps, both drop to ~45–52%. Why does shuffling hurt both models, and what does this
   tell you about what each model learned?
5. Delayed recall at $T=10$ succeeds (100%) but at $T=50$ fails (~47%). Is this primarily a
   **capacity** issue (hidden size 16) or a **vanishing gradient** issue? How would you
   distinguish the two experimentally?
6. Topic 11 introduces LSTM/GRU gates. Which term in the vanilla RNN backward pass do gates
   most directly address, and why?

## Further Reading

- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 10.2](https://www.deeplearningbook.org/contents/rnn.html) — recurrent networks and BPTT.
- Karpathy, A. ["The Unreasonable Effectiveness of Recurrent Neural Networks"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) — intuitive guide to sequence modeling with RNNs.
- [PyTorch nn.RNN documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — `batch_first`, `num_layers`, and weight layout.
- Bengio, Y. et al. (1994). ["Learning Long-Term Dependencies with Gradient Descent is Difficult"](https://ieeexplore.ieee.org/document/6796450) — classic analysis of vanishing gradients in recurrent nets.

---
[← Back to Deep Learning Foundations](../README.md)
