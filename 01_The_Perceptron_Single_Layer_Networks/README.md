# The Perceptron & Single-Layer Networks

> Status: ✅ Complete

The original linear neuron and its learning rule

## Concept & Intuition

Every modern neural network — from a two-layer MLP to a 100-billion-parameter Transformer — is built out of the same basic unit: a weighted sum of inputs passed through a nonlinearity, trained by nudging weights toward reducing error. That unit, and the simplest possible version of that training procedure, is **Frank Rosenblatt's perceptron** (1958), itself inspired by the McCulloch-Pitts (1943) model of a biological neuron.

The perceptron is a **linear threshold unit**: it takes weighted inputs, sums them with a bias, and fires ($+1$) if the sum crosses a threshold, otherwise stays silent ($-1$). Geometrically, it draws a single straight line (a hyperplane in higher dimensions) through feature space. What makes it a genuine learning algorithm is the **perceptron learning rule**: whenever it misclassifies a training example, it nudges its weights directly toward that example, scaled by the true label. Rosenblatt proved this rule is guaranteed to find a perfect separator in a finite number of steps — but only if one exists, i.e. only if the two classes are **linearly separable**. That "if" turns out to be the single most consequential fact about this model, and is the focus of this notebook.

## Mathematical Explanation

**Decision function.** For input $\mathbf{x} \in \mathbb{R}^d$, weights $\mathbf{w}$, and bias $b$:

$$\hat{y} = \text{sign}(\mathbf{w}\cdot\mathbf{x} + b)$$

**Perceptron learning rule.** For each training example $(\mathbf{x}_i, y_i)$, $y_i\in\{-1,+1\}$, if misclassified ($y_i(\mathbf{w}\cdot\mathbf{x}_i+b) \le 0$):

$$\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i, \qquad b \leftarrow b + \eta\, y_i$$

Correctly classified points cause **zero update** — this is an error-driven rule, unlike gradient descent on a smooth loss.

**Perceptron Convergence Theorem (Novikoff, 1962).** If the training data is linearly separable with margin $\gamma$ (a unit vector $\mathbf{u}$, bias folded into $\mathbf{x}$, satisfies $y_i(\mathbf{u}\cdot\mathbf{x}_i) \ge \gamma > 0$ for all $i$) and every $\|\mathbf{x}_i\|\le R$, then starting from $\mathbf{w}=\mathbf{0}$, the perceptron makes **at most**

$$k \le \left(\frac{R}{\gamma}\right)^2$$

mistakes before converging — regardless of presentation order, and (as shown empirically below) regardless of the learning rate. If no such $\gamma>0$ exists, convergence is not guaranteed and the algorithm may cycle forever.

## Code Implementation (`01_the_perceptron_single_layer_networks.ipynb`)

### 1. Validation against scikit-learn

The from-scratch `Perceptron` class is tested on a moderately-overlapping synthetic 2D dataset and compared to `sklearn.linear_model.Perceptron`. Both reach **100% train accuracy**, but land on **visibly different weight vectors** ([4.618, 3.543] vs. [2.940, 2.230]) — the perceptron finds *a* valid separator, not a unique or maximum-margin one; any hyperplane consistent with the final training pass is a valid fixed point.

### 2. Decision boundary evolution

Visualizing the same synthetic dataset's decision boundary after epochs 1, 2, and 6 (full convergence, 13 total updates) shows the line visibly rotating and shifting as more of the (partially overlapping) data is seen, settling into a stable separator only once every point is correctly classified.

### 3. The fundamental limitation: AND/OR vs. XOR

| Gate | Converged? | Epochs run | Final train acc | Mistakes, last 5 epochs |
|---|---|---|---|---|
| AND | Yes | 6 | 1.00 | [2, 1, 1, 1, 0] |
| OR | Yes | 4 | 1.00 | [2, 1, 1, 0] |
| XOR | **No** | 200 (cap) | 0.50 | [4, 3, 3, 4, 4] |

AND and OR — both linearly separable boolean functions — converge in a handful of epochs. **XOR never converges**: even after 200 full epochs it is still making 3-4 mistakes out of 4 points *every single epoch*, oscillating rather than settling. This is exactly the limitation Minsky & Papert's 1969 book *Perceptrons* proved rigorously, and it directly contributed to a decade-long collapse in neural network research funding (the first "AI winter") — resolved only once multi-layer networks with hidden units (Topic 02) were shown to overcome it.

### 4. The convergence theorem, empirically verified

Using a hard-margin SVM to compute the true geometric margin $\gamma$ and radius $R$ for synthetic data at various controlled separations (bias folded into an augmented feature vector, matching the theorem's exact assumptions):

| Separation | $\gamma$ | $R$ | Actual updates | Bound $(R/\gamma)^2$ | Within bound? |
|---|---|---|---|---|---|
| 0.42 | 0.0769 | 1.6035 | 39 | 434.43 | YES |
| 0.45 | 0.1192 | 1.6345 | 19 | 187.94 | YES |
| 0.50 | 0.1897 | 1.6872 | 2 | 79.15 | YES |
| 0.55 | 0.2601 | 1.7412 | 2 | 44.81 | YES |
| 0.60 | 0.3305 | 1.7964 | 2 | 29.54 | YES |
| 0.80 | 0.6125 | 2.0268 | 2 | 10.95 | YES |
| 1.50 | 1.5993 | 2.9070 | 1 | 3.30 | YES |
| 4.00 | 5.1319 | 6.3348 | 1 | 1.52 | YES |

The actual mistake count is **within the theoretical bound in every case**, exactly as Novikoff's theorem guarantees — and the trend is directly visible: shrinking the true margin requires dramatically more updates in practice too (39 at separation 0.42 vs. just 1 at separation 4.0). The bound is also often extremely loose: at separation 0.50, up to 79 mistakes are allowed but only 2 occur. The theorem guarantees an upper limit, not a tight estimate.

### 5. Learning rate invariance

| Learning rate $\eta$ | Epochs | Updates | Train acc | $\|\mathbf{w}\|$ |
|---|---|---|---|---|
| 0.001 | 2 | 10 | 1.0000 | 0.0099 |
| 0.01 | 2 | 10 | 1.0000 | 0.0992 |
| 0.1 | 2 | 10 | 1.0000 | 0.9916 |
| 1.0 | 2 | 10 | 1.0000 | 9.9161 |
| 10.0 | 2 | 10 | 1.0000 | 99.1615 |
| 100.0 | 2 | 10 | 1.0000 | 991.6148 |

Across 5 orders of magnitude in $\eta$, the perceptron converges after **exactly the same 2 epochs and 10 updates** every time — only the final weight vector's magnitude scales linearly with $\eta$. For a zero-initialized perceptron, the learning rate has zero effect on which points get misclassified at each step, and therefore zero effect on convergence speed — a sharp, genuinely surprising contrast with gradient-descent-based training (Topics 02, 05), where learning rate is one of the most consequential hyperparameters.

### 6. Real dataset: Iris, and where linear separability actually breaks

| Comparison | Converged? | Epochs | Updates | Train acc |
|---|---|---|---|---|
| Setosa vs. Versicolor | Yes | 2 | 10 | 1.0000 |
| Setosa vs. Virginica | Yes | 2 | 8 | 1.0000 |
| Versicolor vs. Virginica | **No** | 500 (cap) | 6,518 | 0.8100 |

Using real sepal-length and petal-length measurements, Setosa is trivially separable from the other two species (100% accuracy in 2 epochs). **Versicolor and Virginica are not perfectly linearly separable** in this 2-feature space — the perceptron never converges in 500 epochs, settling at an **81% accuracy ceiling** with mistakes still occurring every epoch. This is the same fundamental limitation as XOR, discovered here in real biological measurements instead of a toy truth table.

## Common Pitfalls

1. **The perceptron gives no warning when it can't converge.** On non-separable data (XOR, Versicolor-vs-Virginica) it runs forever, oscillating, with no built-in stopping signal beyond a max-epoch cutoff you set yourself.
2. **The learned separator is not unique and not maximum-margin.** The from-scratch and sklearn implementations reached 100% accuracy with visibly different weight vectors — the perceptron stops at the *first* hyperplane consistent with the data pass, not the *best* one (that's what SVMs are for).
3. **The convergence bound $(R/\gamma)^2$ can be extremely loose.** Cases were found where the bound allowed 80+ mistakes but only 1-2 actually occurred — don't treat the bound as a tight prediction of real convergence speed.
4. **Learning rate does not affect convergence speed for a zero-initialized perceptron**, a genuinely different behavior from gradient-descent-based methods in later topics where learning rate is one of the most consequential hyperparameters — don't assume intuitions transfer directly.
5. **Linear separability is a property of the feature representation, not just the underlying problem.** Versicolor and Virginica aren't separable using (sepal length, petal length) alone, but adding or engineering more features can restore separability — exactly the motivation for hidden layers in Topic 02.
6. **A single perceptron only does binary classification.** Multi-class problems need one-vs-rest perceptrons or a genuinely different output layer — there is no native multi-class perceptron output.

## Function Reference

| Function | Purpose |
|---|---|
| `Perceptron` (from scratch) | Core error-driven perceptron with weight-history tracking |
| `perceptron_augmented` (from scratch) | Bias-folded perceptron matching the convergence theorem's exact assumptions |
| `margin_and_radius` (from scratch, uses `SVC`) | Computes the true geometric margin $\gamma$ and radius $R$ for the convergence bound |
| `sklearn.linear_model.Perceptron` | Reference implementation used for validation |
| `sklearn.svm.SVC(kernel="linear")` | Hard-margin SVM used only to compute the exact max-margin $\gamma$ |

## Self-Test

1. Why does a correctly classified training example cause zero weight update in the perceptron rule, unlike in gradient-descent-based training?
2. What does the "margin" $\gamma$ represent in the convergence theorem, and why does a smaller $\gamma$ lead to a larger mistake bound?
3. Why can no single perceptron ever learn the XOR function, no matter how long it trains or what learning rate is used?
4. In Section 4, the theoretical bound was often far looser than the actual number of mistakes. What does this imply about using the bound to predict real-world training time?
5. Why did the learning rate have literally zero effect on the number of updates needed to converge in Section 5? Would this still hold if the perceptron were initialized with random (non-zero) weights?
6. What does the Versicolor-vs-Virginica result in Section 6 suggest about how you might make that pair linearly separable without changing the underlying data collection?

## Notebook

[`01_the_perceptron_single_layer_networks.ipynb`](./01_the_perceptron_single_layer_networks.ipynb)

## Further Reading

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review, 65(6), 386-408.
- Novikoff, A.B.J. (1962). *On Convergence Proofs for Perceptrons.* Proceedings of the Symposium on Mathematical Theory of Automata.
- Minsky, M., Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.* MIT Press. (Proved the XOR limitation; widely credited with triggering the first AI winter.)
- McCulloch, W.S., Pitts, W. (1943). *A Logical Calculus of the Ideas Immanent in Nervous Activity.* Bulletin of Mathematical Biophysics, 5(4), 115-133. (The original artificial neuron model that inspired the perceptron.)
- [scikit-learn Perceptron documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

---
[← Back to Deep Learning Foundations](../README.md)
