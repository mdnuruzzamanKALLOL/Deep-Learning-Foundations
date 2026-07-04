# Embedding Layers & Representation Learning

> Status: ✅ Complete

Embedding layers implemented from scratch (validated against PyTorch's `nn.Embedding`
to machine precision), then used to learn word representations purely from
co-occurrence statistics via skip-gram with negative sampling — no labels involved.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`18_embedding_layers_representation_learning.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

Every prior recurrent/attention topic (10, 11, 14, 17) quietly used an embedding table
(`self.embed = rng.randn(vocab, dim) * 0.1`, looked up by integer index) without
formalizing it. This topic makes that lookup a first-class object and asks a deeper
question: can the *weights themselves* be learned usefully **without any labels at all**?

An **embedding layer** maps a discrete index to a dense vector by gathering a row of a
weight matrix — mathematically identical to `one_hot(index) @ W`, just computed in
$O(d)$ instead of $O(Vd)$.

**Representation learning** trains those weights using only the structure of unlabeled
data. **Skip-gram** (Mikolov et al., 2013) is the classic example: predict a word's
context from the word itself. Words that appear in similar contexts converge to similar
vectors — the *distributional hypothesis* ("a word is characterized by the company it
keeps"). The resulting embeddings capture semantic structure that was never explicitly
labeled, and can be reused (like Topic 13's transfer learning) for downstream tasks with
far less labeled data than training from scratch.

## Mathematical Explanation

**Embedding layer:**

$$\mathrm{embed}(i) = \mathrm{one\_hot}(i)^\top W, \qquad \frac{\partial \mathcal{L}}{\partial W_i} = \sum_{\text{occurrences of } i} \frac{\partial \mathcal{L}}{\partial \mathrm{embed}(i)}$$

The sum in the backward pass is critical: if index $i$ appears multiple times in a
batch, every occurrence's gradient must accumulate into the same row.

**Skip-gram with negative sampling** turns the (expensive, full-vocabulary softmax)
context-prediction problem into $K+1$ independent binary classifications per target word:

$$\mathcal{L} = -\log\sigma(u_t \cdot v_c) - \sum_{k=1}^{K} \log\sigma(-u_t \cdot v_{n_k})$$

where $u_t$ is the target word's embedding (from table $U$), $v_c$ is the true context
word's embedding (from a *separate* table $V$), and $v_{n_k}$ are $K$ randomly sampled
negative (non-context) words. Maximizing $\sigma(u_t \cdot v_c)$ pulls true
target-context pairs together; minimizing $\sigma(u_t \cdot v_{n_k})$ pushes random pairs
apart. $U$ is kept as the final learned word representation.

**Gradients** (batch-averaged over $n$ pairs, $K$ negatives each):

$$\frac{\partial \mathcal{L}}{\partial u_t} = (\sigma(u_t \cdot v_c) - 1)\, v_c + \sum_k \sigma(u_t \cdot v_{n_k})\, v_{n_k}$$

## Code Implementation

All code lives in
[`18_embedding_layers_representation_learning.ipynb`](18_embedding_layers_representation_learning.ipynb). It:

1. Implements `Embedding` (row-gather forward, scatter-add backward) and validates it
   against manual one-hot matrix multiplication and PyTorch's `nn.Embedding`.
2. Implements `SkipGramNS` (skip-gram with negative sampling) with full manual backprop.
3. Builds a **synthetic categorical corpus** (4 categories × 8 words, topically coherent
   sentences) with known ground-truth structure the model is never shown.
4. Trains skip-gram and measures whether nearest neighbors in embedding space recover
   the (hidden) categories.
5. Visualizes learned vs. random embeddings via 2D PCA, colored by category.
6. Runs a **label-efficient transfer experiment**: train a linear probe on a *few*
   labeled words, test generalization to *many* held-out, never-labeled words.
7. Sweeps embedding dimensionality (2 to 32) on a harder version of the corpus.

## Key Results

**Embedding layer validation** (vocab=6, dim=4, repeated index in batch):

| Check | Max diff |
|---|---|
| Forward: lookup vs. `one_hot @ W` | 0.0 |
| Backward: scatter-add vs. `one_hot.T @ dout` | 0.0 |
| Forward vs. PyTorch indexing | 0.0 |
| Backward vs. PyTorch autograd | 0.0 |
| Forward vs. `nn.Embedding` | 0.0 |

Exact machine-precision agreement — the embedding layer's forward and backward passes
are precisely a gather and a scatter-add.

**Unsupervised category recovery** (top-3 same-category nearest-neighbor accuracy, 4
categories → chance = 0.25):

| Embedding | Accuracy |
|---|---|
| Trained (skip-gram) | **1.000** |
| Random (untrained) | 0.198 |
| Chance | 0.250 |

Every word's top-3 nearest neighbors fall in its own category — recovered **entirely**
from co-occurrence statistics, with zero category labels ever used during training.

**Label-efficient transfer** (2 of 8 words per category labeled = 8/32 total; linear
probe on frozen embeddings, 5-seed mean):

| Embedding | Labeled-word acc | Held-out-word acc |
|---|---|---|
| Skip-gram (pretrained) | 1.000 | **0.958** |
| Random (untrained) | 1.000 | 0.333 |

Both probes perfectly fit the 8 labeled words (any 8 points are nearly linearly
separable into 4 classes in 16D). The difference is **generalization**: the pretrained
embeddings' geometry already groups same-category words together, so the probe
transfers almost perfectly to 24 words it never saw a label for. The random-embedding
probe is barely above chance on those same held-out words — it memorized 8 arbitrary
points with no shared structure to generalize from.

**Dimensionality sweep** (harder corpus: more noise, fewer sentences):

| Dim | Trained acc | Random acc |
|---|---|---|
| 2 | 0.958 | 0.146 |
| 4 | 1.000 | 0.167 |
| 8 | 1.000 | 0.250 |
| 16 | 1.000 | 0.198 |
| 32 | 0.938 | 0.208 |

Unlike Topic 12's autoencoder bottleneck (a sharp trade-off curve), dimensionality
barely matters here — even 2D separates our 4 clean synthetic categories. We report
this honestly rather than force a dramatic trend that isn't there: dimensionality
trade-offs bite harder with large, ambiguous, real-world vocabularies than with a small
synthetic one.

## Common Pitfalls

1. **Using `+=` instead of `np.add.at` for embedding gradients.** Fancy-index `+=`
   silently drops contributions from repeated indices in the same batch — exactly the
   bug documented in Topic 17. Always use `np.add.at` for scatter-add gradients.
2. **Confusing the target and context embedding tables.** Skip-gram needs two separate
   tables ($U$, $V$); collapsing them into one makes every word's positive score with
   itself trivially maximal ($u_t \cdot u_t$ is always the largest possible dot product),
   degenerating training.
3. **Forgetting to sample negatives from the full vocabulary distribution.** We sample
   negatives uniformly here for simplicity; production skip-gram samples from a smoothed
   unigram distribution ($p(w)^{3/4}$) so common words don't dominate every negative batch.
4. **Assuming a perfect training-set fit means generalization.** Our random-embedding
   probe fits its 8 labeled words at 100% accuracy — identical to the pretrained probe —
   yet collapses to near-chance on unseen words. Always evaluate on held-out items the
   probe never saw, not just on the labeled training set.
5. **Expecting embedding dimensionality trade-offs to always mirror autoencoder
   bottlenecks.** Topic 12's reconstruction task created a sharp $K$-dependent trade-off;
   our small, well-separated synthetic vocabulary shows almost no dependence on
   dimension. The lesson generalizes only as far as the scale of the underlying data.
6. **Training skip-gram without enough negative samples or epochs and concluding it
   "doesn't work."** With too few negatives ($K=1$) or too few epochs, embeddings barely
   move from their random initialization — check the training loss is actually
   decreasing before evaluating downstream structure.

## Function Reference

| Function / Class | Purpose |
|---|---|
| `Embedding` | Row-gather lookup table with scatter-add backward |
| `SkipGramNS` | Skip-gram with negative sampling; separate target ($U$) / context ($V$) tables |
| `make_corpus` | Synthetic categorical corpus with topically coherent sentences |
| `make_skipgram_pairs` | Extracts (target, context) pairs within a sliding window |
| `topk_same_category_accuracy` | Fraction of top-$k$ cosine-nearest neighbors sharing a category |
| `train_linear_probe` / `eval_probe` | Linear classifier on frozen embeddings, train/test split |
| `Adam` | Optimizer for training (Topic 05) |

## Self-Test

1. Write the embedding layer's forward pass as a matrix operation. Why is a row-gather
   preferred over materializing the one-hot vector in practice?
2. Why does skip-gram use two separate embedding tables ($U$ for targets, $V$ for
   contexts) instead of one shared table?
3. Our top-3 same-category nearest-neighbor accuracy reaches 1.000 with zero category
   labels used during training. What information in the corpus made this possible?
4. Both the pretrained and random-embedding linear probes reach 100% accuracy on their
   8 labeled training words. What single number in our results reveals that only one of
   them actually learned something useful, and why?
5. If we labeled *all* 32 words instead of just 8, would you expect the accuracy gap
   between pretrained and random embeddings to grow, shrink, or vanish? Why?
6. Our dimensionality sweep shows almost no effect from 2D to 32D. Design an experiment
   (e.g., more categories, more semantic nuance, larger vocabulary) that you would
   expect to reveal a sharper dimensionality trade-off, and explain why.

## Further Reading

- Mikolov, T. et al. (2013). ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781) — original word2vec (CBOW and skip-gram).
- Mikolov, T. et al. (2013). ["Distributed Representations of Words and Phrases and their Compositionality"](https://arxiv.org/abs/1310.4546) — negative sampling.
- Pennington, J., Socher, R., Manning, C. (2014). ["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/pubs/glove.pdf) — co-occurrence-matrix-factorization alternative to skip-gram.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Section 15](https://www.deeplearningbook.org/contents/representation.html) — representation learning.
- [PyTorch nn.Embedding documentation](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

---
[← Back to Deep Learning Foundations](../README.md)
