# Loss Functions

> Status: ✅ Complete

Cross-Entropy, MSE, Hinge, and Focal Loss — implemented from scratch, validated against
PyTorch to machine precision, and stress-tested with controlled experiments that show
*mechanistically* why each loss is (or isn't) the right tool for a given task.

## Contents
- [Concept & Intuition](#concept--intuition)
- [Mathematical Explanation](#mathematical-explanation)
- [Code Implementation](#code-implementation) (`04_loss_functions.ipynb`)
- [Key Results](#key-results)
- [Common Pitfalls](#common-pitfalls)
- [Function Reference](#function-reference)
- [Self-Test](#self-test)
- [Further Reading](#further-reading)

---

## Concept & Intuition

A loss function defines what "good" means for a model — it shapes the entire optimization
landscape, decides which mistakes get punished harder, and determines whether gradient
descent is smooth or gets stuck. Picking the wrong loss for a task isn't just suboptimal,
it can silently sabotage training in ways that look like a bug elsewhere (a "dead" network
that never learns, a regression fit hijacked by a handful of bad data points, a classifier
that never learns the rare class it's supposed to catch).

This topic answers four concrete questions with numbers rather than assertions:
1. Why does using MSE with a sigmoid output for classification make learning stall when the
   model starts out confidently wrong?
2. Why is MAE robust to outliers in regression while MSE is not?
3. Why does hinge loss (SVM) produce sparse "support vectors" while cross-entropy keeps
   pushing on every point forever?
4. Why does focal loss help with class imbalance, and what does it cost you?

## Mathematical Explanation

**Mean Squared Error (MSE)**

$$L_{\text{MSE}} = \frac{1}{n}\sum_i (\hat y_i - y_i)^2, \qquad \frac{\partial L}{\partial \hat y_i} = \frac{2(\hat y_i - y_i)}{n}$$

Gradient is *linear* in the residual — large errors get proportionally large gradients,
which is exactly why outliers dominate an MSE fit.

**Mean Absolute Error (MAE)**

$$L_{\text{MAE}} = \frac{1}{n}\sum_i |\hat y_i - y_i|, \qquad \frac{\partial L}{\partial \hat y_i} = \frac{\text{sign}(\hat y_i - y_i)}{n}$$

Gradient magnitude is *constant* (just $\pm 1/n$) regardless of how large the residual is —
outliers get the same "vote" as any other misfit point. Non-differentiable at zero (in
practice this is a non-issue with subgradients).

**Binary Cross-Entropy (BCE)**

Derived from the Bernoulli negative log-likelihood ($p\in(0,1)$, $y\in\{0,1\}$):

$$L_{\text{BCE}} = -\frac{1}{n}\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$

If $p=\sigma(z)$, the chain rule gives $\partial L/\partial z = p - y$: the sigmoid's own
derivative $p(1-p)$ cancels analytically against the $1/p$, $1/(1-p)$ terms from the
log-likelihood. This is the single most important fact in this topic — it's *why*
cross-entropy doesn't saturate the way MSE does.

**Categorical Cross-Entropy (Softmax)**

For $K$ classes with one-hot target $y$ and softmax probabilities $p_k=e^{z_k}/\sum_j e^{z_j}$:

$$L_{\text{CCE}} = -\frac{1}{n}\sum_i\sum_k y_{ik}\log p_{ik}, \qquad \frac{\partial L}{\partial z_k} = \frac{p_k-y_k}{n}$$

Same clean $p-y$ form as BCE generalizes to the multi-class case.

**Hinge Loss (SVM)**

For labels $y\in\{-1,+1\}$ and raw score $f(x)$:

$$L_{\text{hinge}} = \frac{1}{n}\sum_i \max(0,\, 1-y_if(x_i))$$

Exactly zero loss *and* zero gradient once a point is correctly classified with margin
$\geq 1$. Only points inside the margin (or misclassified) contribute gradient — this is
the origin of SVM sparsity ("support vectors").

**Focal Loss** (Lin et al., 2017, *RetinaNet*)

$$L_{\text{focal}} = -\alpha_t(1-p_t)^\gamma\log(p_t), \qquad p_t=\begin{cases}p & y=1\\1-p & y=0\end{cases}$$

The $(1-p_t)^\gamma$ modulating factor shrinks toward zero as $p_t\to 1$ (i.e. as an example
becomes "easy"), redirecting gradient budget toward hard/rare examples. $\alpha_t$ separately
re-weights the rare class. Setting $\gamma=0$ recovers ordinary (weighted) cross-entropy.

## Code Implementation

All code lives in [`04_loss_functions.ipynb`](04_loss_functions.ipynb). It:

1. Implements MSE, MAE, BCE, CCE (softmax), Hinge, and Focal loss with their gradients
   from scratch in NumPy.
2. Validates every loss (and gradient) against PyTorch (`F.mse_loss`, `F.l1_loss`,
   `F.binary_cross_entropy`, `F.cross_entropy`) or finite-difference numerical gradients
   (for hinge and focal, which have no exact PyTorch built-in in this formulation) —
   all match to $10^{-10}$ or better.
3. Visualizes the loss landscapes: MSE/MAE vs. residual, BCE vs. predicted probability,
   hinge vs. logistic loss vs. margin.
4. Demonstrates the **MSE + sigmoid gradient saturation trap**: computes
   $\partial L/\partial z$ at various $z$ for both losses, then trains a single neuron from
   a deliberately bad initialization to show MSE gets stuck while BCE escapes quickly.
5. Demonstrates **MAE's robustness to outliers** by fitting a linear regression with both
   losses on data containing a cluster of extreme outliers.
6. Demonstrates **hinge loss sparsity**: trains a linear classifier with hinge and with
   logistic loss on the same data, identifies "support vectors" (margin $\leq 1$), and shows
   the gradient contribution of a confidently-correct point is exactly 0 for hinge but a
   small nonzero number for logistic.
7. Demonstrates **focal loss for class imbalance**: on a 95:5 imbalanced binary
   classification problem, trains with plain BCE and with focal loss, comparing precision/
   recall/F1, averaged over 20 random seeds for robustness.

## Key Results

**All losses validated against PyTorch / finite differences:**

| Loss | Value match | Gradient max diff |
|---|---|---|
| MSE | 2.2209999124 (both) | 5.55e-17 |
| MAE | 1.0456493705 (both) | 0.00e+00 |
| BCE | 0.9049169957 (both) | 1.11e-16 |
| CCE (softmax) | 1.3296191958 (both) | 1.39e-17 |
| Hinge | 1.060277 | 8.18e-11 (numerical check) |

**1. MSE + sigmoid saturates exactly when the model is most wrong.** True label $y=1$:

| $z$ | $p=\sigma(z)$ | $\partial L/\partial z$ (MSE) | $\partial L/\partial z$ (BCE) |
|---|---|---|---|
| -10 | 0.000045 | -0.000091 | **-0.999955** |
| -5 | 0.006693 | -0.013207 | -0.993307 |
| 0 | 0.5 | -0.25 | -0.5 |
| 10 | 0.999955 | -0.000000 | -0.000045 |

At $z=-10$ (confidently wrong), BCE's gradient is **~11,000× larger** than MSE's. In an
actual training run starting from a bad initialization ($p_0\approx 0.0025$, true label 1),
**BCE reaches $p>0.9$ in 17 steps**, while **MSE is still stuck at $p=0.053$ after 200
steps** — same data, same learning rate, same starting point. (See the training-curve plot
in the notebook.)

**2. MAE is robust to outliers, MSE is not.** A cluster of 4 outliers (8% of 50 points)
more than doubles the MSE-fitted slope (true slope 2.0):

| | slope | intercept |
|---|---|---|
| True | 2.0000 | 1.0000 |
| MSE fit (clean data) | 2.0462 | 0.7345 |
| MSE fit (with outliers) | **4.2776** | -5.4817 |
| MAE fit (with outliers) | 2.1242 | 0.3737 |

(See the fitted-line plot in the notebook — the MSE line is visibly dragged toward the
outlier cluster while the MAE line tracks the true trend.)

**3. Hinge loss creates sparsity; cross-entropy doesn't.** Both losses recover essentially
the same decision boundary (direction cosine similarity > 0.999 with the true boundary), but:
- Only **61 of 203 points (30%)** are "support vectors" (margin $\leq 1$) under hinge loss.
- For the most confidently-correct point (margin = 8.05): hinge's gradient contribution is
  **exactly 0.0**; logistic's is **7.48e-10** — nonzero, but shrinking forever, never reaching
  zero.

(See the decision-boundary plot in the notebook, with support vectors highlighted.)

**4. Focal loss trades precision for recall on imbalanced data.** On a 95:5 imbalanced binary
classification task (averaged over 20 seeds):

| Loss | Precision | Recall | F1 |
|---|---|---|---|
| BCE | 0.737 | 0.250 | 0.364 |
| Focal ($\alpha=0.75,\gamma=2$) | 0.515 | **0.488** | **0.485** |

Focal loss nearly **doubles recall** on the minority class and improves F1, at the cost of
precision — it stops the model from ignoring the rare class, catching more true positives at
the price of more false positives.

(See the precision/recall/F1 bar chart in the notebook.)

## Common Pitfalls

1. **Using MSE with a sigmoid/softmax output for classification.** The $p(1-p)$ sigmoid
   derivative embedded in the MSE gradient vanishes exactly when the model is confidently
   wrong, causing training to stall from bad initializations or after early mistakes.
   Cross-entropy's gradient cancels this term analytically. Always pair probability outputs
   with (some form of) cross-entropy, not MSE.
2. **Believing MAE is "always more robust" without checking convergence speed.** MAE's
   constant-magnitude gradient means training can be slower to fine-tune near the optimum
   (it doesn't shrink as the error shrinks, unlike MSE). In practice, robust regression often
   uses Huber loss — a smooth MSE/MAE hybrid — specifically to get MAE's tail robustness with
   MSE's near-zero smoothness.
3. **Assuming hinge and cross-entropy converge to the same classifier in general.** They
   agree here because the data is linearly separable and well-behaved; in general hinge loss
   optimizes margin, not probability, and gives no calibrated confidence estimate.
4. **Tuning focal loss's $\gamma$ and $\alpha$ blindly.** $\gamma$ too large can suppress
   gradient on *everything*, including genuinely hard examples that matter, slowing
   convergence. $\alpha$ re-weighting alone (without the $(1-p_t)^\gamma$ term, i.e.
   plain weighted cross-entropy) is a simpler and sometimes equally effective fix for
   imbalance — always compare against that baseline first.
5. **Reporting only accuracy on imbalanced data.** A classifier that always predicts the
   majority class gets 95% accuracy on a 95:5 split while catching zero minority cases.
   Precision/recall/F1 (or PR-AUC) are the metrics that actually reveal the trade-off focal
   loss is making.
6. **Forgetting that focal loss's benefit is a trade, not a free lunch.** The 20-seed average
   shows recall nearly doubling but precision dropping by ~22 points — whether this is a good
   trade depends entirely on the relative cost of false negatives vs. false positives in the
   application (e.g. worth it for cancer screening, possibly not for spam filtering).

## Function Reference

| Function | Purpose |
|---|---|
| `mse_loss`, `mse_grad` | Mean squared error and its gradient |
| `mae_loss`, `mae_grad` | Mean absolute error and its gradient |
| `bce_loss`, `bce_grad_wrt_p` | Binary cross-entropy and its gradient w.r.t. probability |
| `sigmoid`, `softmax` | Standard activation functions used to produce probabilities |
| `cce_loss` | Categorical cross-entropy for multi-class softmax outputs |
| `hinge_loss`, `hinge_grad` | SVM-style hinge loss and its (sub)gradient |
| `focal_loss`, `focal_grad_wrt_z` | Focal loss and its gradient w.r.t. pre-sigmoid logit |
| `train_mse` / `train_bce` | Single-neuron training loops used for the saturation demo |
| `fit_linear` | Linear regression fit via gradient descent under MSE or MAE |
| `train_hinge` / `train_logistic` | Linear classifier training under hinge vs. logistic loss |
| `train_bce_clf` / `train_focal_clf` | Logistic classifier training under BCE vs. focal loss |
| `make_imbalanced` | Generates a synthetic 95:5 imbalanced binary classification dataset |

## Self-Test

1. Derive $\partial L_{\text{BCE}}/\partial z$ from $L=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))$
   using the chain rule, and show explicitly why the sigmoid derivative $\sigma(z)(1-\sigma(z))$
   cancels out.
2. Why does MAE's gradient not shrink as training approaches the optimum, and what practical
   problem can this cause? How does Huber loss address it?
3. If you moved a hinge-loss support vector further from the boundary (but keeping margin
   $<1$), would the decision boundary change? What if you moved a non-support point further
   from the boundary?
4. Compute $L_{\text{focal}}$ for $p=0.99, y=1$ vs. $p=0.6, y=1$ with $\gamma=2,\alpha=0.75$.
   Which contributes more loss, and does that match the intuition of "down-weighting easy
   examples"?
5. In the imbalanced classification experiment, why does precision drop when recall improves?
   Sketch what happens to the decision boundary as $\gamma$ increases from 0.
6. Categorical cross-entropy's gradient is $p_k - y_k$ — identical in form to binary
   cross-entropy's $p-y$. Why is this not a coincidence, and what property of the
   softmax+CCE pairing (analogous to sigmoid+BCE) produces it?

## Further Reading

- Lin, T-Y. et al. (2017). ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002) — the original RetinaNet paper introducing focal loss.
- Cortes, C. & Vapnik, V. (1995). ["Support-Vector Networks"](https://doi.org/10.1007/BF00994018) — the original SVM paper introducing hinge loss.
- Goodfellow, Bengio, Courville, *Deep Learning* (2016), [Chapter 6.2](https://www.deeplearningbook.org/contents/mlp.html) — cross-entropy as the standard loss for neural network classifiers and why it avoids saturation.
- Huber, P. J. (1964). ["Robust Estimation of a Location Parameter"](https://doi.org/10.1214/aoms/1177703732) — the original Huber loss paper, blending MSE and MAE.
- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) — precision, recall, F1, and PR curves for imbalanced classification.

---
[← Back to Deep Learning Foundations](../README.md)
<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.04_Loss_Functions&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
