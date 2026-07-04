# 🧠 Statistical Machine Learning — Deep Learning Foundations

Part of the [📘 Statistical Machine Learning for Noob](https://github.com/mdnuruzzamanKALLOL) series — a from-scratch, math-first, fully-executed ML study series designed to be read on GitHub, run locally, and actually learned from (not just skimmed).

The [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) repo's README explicitly calls itself "pre-deep-learning" — this repo is the bridge. It builds a neural network from a single perceptron up through backpropagation, optimizers, CNNs, RNNs, and attention, at the fundamentals level this author's much larger [PyTorch→TensorFlow model-conversion catalog](https://github.com/mdnuruzzamanKALLOL?tab=repositories&q=PT2TF) assumes as background.

## 📑 Table of Contents

1. [Why This Repo Exists](#-why-this-repo-exists)
2. [Learning Path](#-learning-path)
3. [Topics](#-topics)
4. [Repository Structure](#-repository-structure)
5. [Prerequisites](#-prerequisites)
6. [How to Use](#-how-to-use)
7. [Datasets](#-datasets)
8. [The Full Series](#-the-full-series)
9. [Self-Test Philosophy](#-self-test-philosophy)
10. [Feedback & Corrections](#-feedback--corrections)
11. [Related](#-related)

---

## 🎯 Why This Repo Exists

Most deep learning material either treats backpropagation as a black box `.backward()` call, or jumps straight into a specific architecture (ResNet, Transformer) without deriving why the underlying training mechanics work at all. This repo derives the actual mechanics from scratch — the perceptron's learning rule, backprop's chain rule, why gradients vanish or explode, what attention actually computes — before any specific architecture is introduced, using the same fully-executed-notebook, honest-results standard as the rest of this series.

## 🗺️ Learning Path

```
01-03 The Core Training Loop         ->  perceptron, MLP, backpropagation, activation functions
        v
04-08 Making Training Actually Work  ->  loss functions, optimizers, initialization, regularization, normalization
        v
09-12 Core Architecture Families     ->  CNNs, RNNs, LSTM/GRU, autoencoders
        v
13-16 Practical & Failure-Mode Topics ->  transfer learning, attention, hyperparameter tuning, vanishing/exploding gradients
        v
17-20 Bridging to Modern Architectures -> seq2seq, embeddings, GANs, interpretability
```

Each topic explicitly says what it's setting up for — read the "Why This Topic Matters" section at the top of every README once topics are built out.

## 📚 Topics

| # | Topic | Key Concepts | Status |
|---|-------|-------------|:---:|
| 01 | [The Perceptron & Single-Layer Networks](01_The_Perceptron_Single_Layer_Networks/) | The original linear neuron and its learning rule | ✅ |
| 02 | [Multi-Layer Perceptrons & Backpropagation](02_Multi_Layer_Perceptrons_Backpropagation/) | The core algorithm every deep network is trained with | ✅ |
| 03 | [Activation Functions](03_Activation_Functions/) | Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish | ✅ |
| 04 | [Loss Functions](04_Loss_Functions/) | Cross-Entropy, MSE, Hinge, Focal Loss | 🚧 |
| 05 | [Optimizers](05_Optimizers/) | SGD, Momentum, RMSProp, Adam, AdamW | 🚧 |
| 06 | [Weight Initialization](06_Weight_Initialization/) | Xavier/Glorot and He initialization, and why it matters | 🚧 |
| 07 | [Regularization](07_Regularization/) | Dropout, L1/L2, early stopping, data augmentation | 🚧 |
| 08 | [Batch Normalization & Layer Normalization](08_Batch_Normalization_Layer_Normalization/) | Stabilizing and accelerating deep network training | 🚧 |
| 09 | [CNN Basics](09_CNN_Basics/) | Convolution, pooling, and feature maps from scratch | 🚧 |
| 10 | [RNN Basics](10_RNN_Basics/) | Vanilla recurrent networks and the vanishing gradient problem | 🚧 |
| 11 | [LSTM & GRU Fundamentals](11_LSTM_GRU_Fundamentals/) | Gated recurrent cells derived from scratch | 🚧 |
| 12 | [Autoencoders](12_Autoencoders/) | Basic, denoising, and a variational introduction | 🚧 |
| 13 | [Transfer Learning & Fine-Tuning](13_Transfer_Learning_Fine_Tuning/) | Reusing pretrained representations for a new task | 🚧 |
| 14 | [Attention Mechanism Basics](14_Attention_Mechanism_Basics/) | The mechanism behind Transformers, before the full architecture | 🚧 |
| 15 | [Hyperparameter Tuning for Deep Nets](15_Hyperparameter_Tuning_for_Deep_Nets/) | Learning rate schedules and search strategies for neural networks | 🚧 |
| 16 | [Vanishing/Exploding Gradients & Gradient Clipping](16_VanishingExploding_Gradients_Gradient_Clipping/) | Why deep networks are hard to train, and how to fix it | 🚧 |
| 17 | [Sequence-to-Sequence Models & Encoder-Decoder Architecture](17_Sequence_to_Sequence_Models_Encoder_Decoder_Architecture/) | Mapping one sequence to another | 🚧 |
| 18 | [Embedding Layers & Representation Learning](18_Embedding_Layers_Representation_Learning/) | Learned dense representations for discrete inputs | 🚧 |
| 19 | [Generative Adversarial Networks (GAN) Basics](19_Generative_Adversarial_Networks_GAN_Basics/) | Generator vs. discriminator, learned through competition | 🚧 |
| 20 | [Model Interpretability for Deep Nets](20_Model_Interpretability_for_Deep_Nets/) | Grad-CAM, saliency maps, and explaining black-box predictions | 🚧 |

**20 topics planned** — built one at a time, each with a deep-dive `README.md` (full math derivation in LaTeX, pitfalls, self-test exercises) and a fully-executed Jupyter notebook (30+ code cells), following the exact standard established in [Foundation](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation) and [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML).

## 📁 Repository Structure

```
Deep-Learning-Foundations/
├── README.md                          ← you are here
├── 01_The_Perceptron_Single_Layer_Networks/
│   └── README.md
├── 02_Multi_Layer_Perceptrons_Backpropagation/
│   └── README.md
├── 03_Activation_Functions/
│   └── README.md
├── 04_Loss_Functions/
│   └── README.md
├── 05_Optimizers/
│   └── README.md
├── 06_Weight_Initialization/
│   └── README.md
├── 07_Regularization/
│   └── README.md
├── 08_Batch_Normalization_Layer_Normalization/
│   └── README.md
├── 09_CNN_Basics/
│   └── README.md
├── 10_RNN_Basics/
│   └── README.md
├── 11_LSTM_GRU_Fundamentals/
│   └── README.md
├── 12_Autoencoders/
│   └── README.md
├── 13_Transfer_Learning_Fine_Tuning/
│   └── README.md
├── 14_Attention_Mechanism_Basics/
│   └── README.md
├── 15_Hyperparameter_Tuning_for_Deep_Nets/
│   └── README.md
├── 16_VanishingExploding_Gradients_Gradient_Clipping/
│   └── README.md
├── 17_Sequence_to_Sequence_Models_Encoder_Decoder_Architecture/
│   └── README.md
├── 18_Embedding_Layers_Representation_Learning/
│   └── README.md
├── 19_Generative_Adversarial_Networks_GAN_Basics/
│   └── README.md
└── 20_Model_Interpretability_for_Deep_Nets/
    └── README.md
```

Every topic folder will be self-contained once built: read the `README.md` for the theory, open the `.ipynb` for the hands-on implementation.

## 🧰 Prerequisites

- Python 3.9+
- PyTorch is used for the notebooks (matching this author's existing model-conversion work), but every core mechanic (backprop, an optimizer step, a convolution) is also derived and implemented in raw NumPy first.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision jupyter
```

## 🚀 How to Use

**Just reading?** Every notebook (once built) renders directly on GitHub with full output — click any `.ipynb` link in the table above.

**Running it yourself:**

```bash
git clone https://github.com/mdnuruzzamanKALLOL/Deep-Learning-Foundations.git
cd Deep-Learning-Foundations
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision jupyter
jupyter notebook
```

## 📦 Datasets

Real-data topics draw from the central **[Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets)** repo (162-entry catalog, 418 files) shared across this entire series.

## 🔭 The Full Series

| Repo | Covers | Status |
|---|---|---|
| [Foundation](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation) | Python/NumPy/Pandas, Visualization, Preprocessing, Feature Engineering, Math | ✅ Complete |
| [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) | Regression, Classification, Ensembles, Unsupervised, Model Evaluation & Tuning (29 algorithms) | ✅ Complete |
| [Statistical Inference & Hypothesis Testing](https://github.com/mdnuruzzamanKALLOL/Statistical-Inference-Hypothesis-Testing) | Probability through causal inference (20 topics) | 🚧 In progress |
| [Time Series Analysis](https://github.com/mdnuruzzamanKALLOL/Time-Series-Analysis) | Decomposition through neural forecasting (20 topics) | 🚧 In progress |
| [Deep Learning Foundations](https://github.com/mdnuruzzamanKALLOL/Deep-Learning-Foundations) | Perceptron through GANs and interpretability (20 topics) | 🚧 In progress |
| [Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets) | 418 datasets (162 cataloged + personal collection) backing the whole series | ✅ Complete |

## 📝 Self-Test Philosophy

Every topic README (once built) ends with a **Self-Test Exercises** section — deliberately *not* answered inline. The point is to predict the answer before running the corresponding notebook cell, not to read a solution.

## 💬 Feedback & Corrections

Found a bug, a math error, or a broken dataset link? Open an issue or a pull request — this series documents its own mistakes on purpose, so a caught error is a feature, not an embarrassment.

## 🔗 Related

- [Foundation →](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation)
- [Classical ML →](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML)
- [Datasets →](https://github.com/mdnuruzzamanKALLOL/Datasets)
- [Author's GitHub profile](https://github.com/mdnuruzzamanKALLOL)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.DeepLearningFoundations.root&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
