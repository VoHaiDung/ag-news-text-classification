# AG News Text Classification

**Intuition**:  
- Maximize probability that target word appears in context: &#963;(𝐮<sub>w_c</sub><sup>T</sup>𝐯<sub>w_t</sub>) &#8594; 1  
- Minimize probability that random words appear: &#963;(𝐮<sub>w_i</sub><sup>T</sup>𝐯<sub>w_t</sub>) &#8594; 0

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Introduction

### 1. Theoretical Foundations and Problem Formulation

#### 1.1 Text Classification as Supervised Learning

Text classification constitutes a fundamental supervised learning problem where the objective is to learn a mapping function from textual inputs to predefined categorical labels, optimizing for generalization to unseen instances drawn from the same underlying distribution.

**Formal Problem Definition**

Let $\mathcal{X}$ denote the space of all possible text documents, and $\mathcal{Y} = \{y_1, y_2, \ldots, y_K\}$ represent a finite set of $K$ predefined classes. We are provided with a training dataset:

$$
\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\}
$$

consisting of $N$ labeled examples, where each $x_i \in \mathcal{X}$ is a text document and $y_i \in \mathcal{Y}$ is its corresponding class label.

The learning objective is to find a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that minimizes the **expected risk** (generalization error):

$$
R(f) = \mathbb{E}_{(x,y) \sim P} [\ell(f(x), y)]
$$

where:
- $P$ is the unknown joint probability distribution over $\mathcal{X} \times \mathcal{Y}$ from which data are sampled
- $\ell: \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}^+$ is a loss function measuring prediction error
- $\mathbb{E}$ denotes the expectation over the data distribution

**Commonly used loss functions**:

- **0-1 Loss** (classification accuracy):
  $$\ell_{0-1}(f(x), y) = \mathbb{I}[f(x) \neq y] = \begin{cases} 0 & \text{if } f(x) = y \\ 1 & \text{if } f(x) \neq y \end{cases}$$

- **Cross-Entropy Loss** (for probabilistic predictions):
  $$\ell_{CE}(f(x), y) = -\log P(y \mid x; f)$$

Since the true distribution $P$ is unknown and inaccessible, we instead minimize the **empirical risk** (training error) computed on the observed dataset:

$$
R_{\text{emp}}(f) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), y_i)
$$

**The Fundamental Challenge: Overfitting**

The core tension in supervised learning is the **bias-variance-covariance decomposition**. For squared loss in regression, the expected error decomposes as:

$$
\mathbb{E}[(f(x) - y)^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

where:
- **Bias**: Error from incorrect assumptions in the learning algorithm (underfitting)
- **Variance**: Error from sensitivity to small fluctuations in training data (overfitting)
- **Irreducible Error**: Noise inherent in the problem (Bayes error)

A model may achieve zero empirical risk (perfect memorization of training data) yet exhibit high expected risk (poor generalization)—the phenomenon of **overfitting**. This occurs when:

$$
R_{\text{emp}}(f) \ll R(f)
$$

**Quantifying Generalization Gap**:

Define the **generalization gap** as:

$$
\Delta(f) = R(f) - R_{\text{emp}}(f)
$$

Statistical learning theory (Vapnik-Chervonenkis theory) provides upper bounds:

$$
R(f) \leq R_{\text{emp}}(f) + \sqrt{\frac{d \log(N/d) + \log(1/\delta)}{N}}
$$

with probability $1-\delta$, where $d$ is the VC dimension (model complexity measure). This bound reveals the trade-off:
- **High capacity** (large $d$): Can fit training data well (low $R_{\text{emp}}$) but large generalization gap
- **Low capacity** (small $d$): Tight bound but may have high $R_{\text{emp}}$ (underfitting)

**This work systematically addresses overfitting** through:
1. **Architectural constraints**: Parameter-efficient methods limiting effective capacity
2. **Regularization strategies**: Explicit penalties on model complexity
3. **Ensemble diversity**: Reducing variance through model averaging
4. **Automated monitoring**: Real-time detection of train-validation divergence
5. **Test set protection**: Rigorous protocols preventing data leakage

#### 1.2 Unique Challenges in Text Classification

Text classification poses distinctive challenges differentiating it from other supervised learning domains:

**Challenge 1: High Dimensionality and Sparsity**

Natural language exists in an extremely high-dimensional space. For a vocabulary of size $|\mathcal{V}|$ (typically 30,000-100,000 unique tokens), even the simplest **bag-of-words** representation creates a $|\mathcal{V}|$-dimensional feature vector.

**Mathematical Representation**: For document $d$ containing words $w_1, w_2, \ldots, w_m$, the bag-of-words vector is:

$$
\mathbf{x} = [c_1, c_2, \ldots, c_{|\mathcal{V}|}]^\top \in \mathbb{R}^{|\mathcal{V}|}
$$

where $c_i$ is the count of word $w_i$ in document $d$.

However, any individual document utilizes only a small fraction of the vocabulary, resulting in **sparse representations** where 95-99% of features are zero.

**Sparsity Measure**: Define document sparsity as:

$$
\text{Sparsity}(d) = 1 - \frac{|\{i : c_i > 0\}|}{|\mathcal{V}|} = 1 - \frac{|d|}{|\mathcal{V}|}
$$

where $|d|$ is the number of unique words in document $d$.

**Example**: A 100-word news article from a 50,000-word vocabulary:
- Unique words in document: ~80 (after removing duplicates)
- Sparsity: $1 - 80/50000 = 0.9984$ (99.84% zeros)

**Implications**:

1. **Curse of Dimensionality**: In high-dimensional spaces, distances become less meaningful. For random points in $\mathbb{R}^d$, the ratio of maximum to minimum distance approaches 1 as $d \rightarrow \infty$:
   $$\lim_{d \rightarrow \infty} \frac{\max_i \|\mathbf{x}_i - \mathbf{x}_0\|}{\min_i \|\mathbf{x}_i - \mathbf{x}_0\|} = 1$$

2. **Sample Complexity**: Number of samples required to learn grows exponentially with dimensionality. For uniform coverage of feature space with resolution $r$, need $O(r^d)$ samples.

3. **Computational Challenges**: Matrix operations on 50,000-dimensional vectors require specialized sparse data structures.

**Solutions**:
- **Dimensionality Reduction**: PCA, LSA project to low-dimensional subspace
- **Dense Embeddings**: Word2Vec, BERT map discrete tokens to continuous $\mathbb{R}^d$ with $d=100-1024$
- **Sparse Operations**: Efficient implementations (CSR matrices, sparse attention)

**Challenge 2: Variable-Length Sequential Structure**

Unlike fixed-size inputs in image classification (e.g., 224×224 pixels), text documents vary dramatically in length—from short social media posts (10-20 tokens) to long articles (1,000+ tokens).

**Sequence Modeling Requirements**:

1. **Handle arbitrary length**: Architecture must process sequences $\mathbf{x} = [x_1, x_2, \ldots, x_n]$ where $n$ varies

2. **Capture local patterns**: Phrases and n-grams like "not good", "very happy", "absolutely terrible"

3. **Model long-range dependencies**: Subject-verb agreement across clauses, coreference resolution (pronouns to antecedents)

**Approaches**:

**Padding/Truncation**: Standardize to fixed length $L$:
$$
\mathbf{x}' = \begin{cases}
[\mathbf{x}; \mathbf{0}_{L-n}] & \text{if } n < L \text{ (pad)} \\
\mathbf{x}_{1:L} & \text{if } n > L \text{ (truncate)}
\end{cases}
$$

*Limitation*: Padding introduces noise, truncation loses information.

**Recurrent Neural Networks**: Process sequences step-by-step with hidden state:
$$
\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t; \theta)
$$

*Limitation*: Vanishing gradients for long sequences (gradient magnitude decays as $\gamma^t$ where $\gamma < 1$).

**Attention Mechanisms**: Allow direct connections between all positions (Section 1.3).

**Challenge 3: Semantic Ambiguity and Context-Dependency**

Natural language exhibits profound ambiguity at multiple linguistic levels:

**1. Polysemy**: Words with multiple meanings depending on context

*Example*: "bank"
- Financial institution: "I deposited money at the **bank**"
- Land alongside river: "We sat by the river **bank**"

**2. Synonymy**: Different words with identical or near-identical meanings

$$\text{Synonyms}(\text{"quick"}) = \{\text{"fast"}, \text{"rapid"}, \text{"swift"}, \text{"speedy"}\}$$

Traditional models treating words as atomic units cannot recognize semantic equivalence.

**3. Compositionality**: Meaning emerges from word combinations

*Negation*: "not good" ≠ "good" (sentiment polarity flip)

*Modifier effects*: "incredibly boring" vs. "incredibly exciting" (same intensifier, opposite results)

**Mathematical Framework for Contextualized Representations**:

Traditional word embeddings assign fixed vectors:
$$w \mapsto \mathbf{v}_w \in \mathbb{R}^d$$

Contextualized embeddings compute representations dynamically:
$$
w_i \text{ in context } [w_1, \ldots, w_n] \mapsto \mathbf{h}_i = f(w_1, \ldots, w_n, i; \theta) \in \mathbb{R}^d
$$

**Example**: In BERT, "bank" receives different representations:
- $\mathbf{h}_{\text{bank}}^{\text{(financial)}} \approx \mathbf{v}_{\text{money}}, \mathbf{v}_{\text{loan}}$ (cosine similarity > 0.7)
- $\mathbf{h}_{\text{bank}}^{\text{(river)}} \approx \mathbf{v}_{\text{water}}, \mathbf{v}_{\text{shore}}$ (cosine similarity > 0.7)
- $\cos(\mathbf{h}_{\text{bank}}^{\text{(financial)}}, \mathbf{h}_{\text{bank}}^{\text{(river)}}) < 0.3$ (distinct representations)

**Challenge 4: Limited Labeled Data vs. Model Capacity**

State-of-the-art models contain hundreds of millions to billions of parameters:
- DeBERTa-v3-XLarge: 710M parameters
- Llama 2-70B: 70B parameters

while supervised datasets typically contain $10^3$ to $10^5$ labeled examples.

**Parameter-to-Sample Ratio**:

$$
\rho = \frac{\text{Model Parameters}}{\text{Training Samples}}
$$

**Critical Threshold**: When $\rho > 1$, severe overfitting risk—model has enough capacity to memorize all training data.

**Examples**:
- DeBERTa-v3-XLarge on AG News: $\rho = 710M / 120K \approx 5917$ (each parameter sees <0.0002 samples!)
- Llama 2-70B on AG News: $\rho = 70B / 120K \approx 583,333$

**Classical Statistical Learning Theory** (Vapnik) suggests sample complexity:

$$
N = O\left(\frac{d}{\epsilon^2}\right)
$$

to achieve error within $\epsilon$ of optimal, where $d$ is VC dimension (roughly proportional to parameters). For $d=710M$, would need billions of labeled samples for reliable learning!

**Modern Solutions**:

**1. Transfer Learning**: Pre-train on massive unlabeled corpus (billions of tokens), then fine-tune:

$$
\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{downstream}}(\theta; \mathcal{D}_{\text{labeled}})
$$

subject to initialization $\theta_0 = \theta_{\text{pretrained}}$

*Intuition*: Pre-training learns general language understanding (syntax, semantics, world knowledge); fine-tuning specializes to task.

**2. Parameter-Efficient Fine-Tuning (PEFT)**: Update only small subset of parameters:

$$
\theta_{\text{trainable}} \subset \theta, \quad |\theta_{\text{trainable}}| \ll |\theta|
$$

*Examples*: 
- LoRA: 0.1-1% of parameters trainable
- Adapters: 0.5-3% trainable
- Prompt tuning: 0.001-0.1% trainable

This dramatically reduces effective capacity, mitigating overfitting.

**3. Regularization**: Explicit complexity penalties:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \Omega(\theta)
$$

where $\Omega(\theta)$ is regularizer (L2 norm, dropout, etc.).

**4. Data Augmentation**: Synthetically expand training set while preserving label:

$$
|\mathcal{D}_{\text{augmented}}| = \alpha \cdot |\mathcal{D}_{\text{original}}|, \quad \alpha \in [2, 10]
$$

through back-translation, paraphrasing, controlled generation.

### 2. Evolution of Text Classification Paradigms

The field has progressed through five distinct eras, each introducing fundamental innovations in representation learning and model architectures.

#### Phase 1: Classical Machine Learning (1990s-2010)

**Core Paradigm**: Transform text into fixed-dimensional feature vectors through manual engineering, then apply traditional classifiers.

**Representation Method 1: Bag-of-Words (BoW)**

Represent document as unordered collection of word counts, completely ignoring grammar, word order, and syntax.

**Mathematical Formulation**: For document $d$ with vocabulary $\mathcal{V} = \{w_1, w_2, \ldots, w_{|\mathcal{V}|}\}$:

$$
\text{BoW}(d) = [c(w_1, d), c(w_2, d), \ldots, c(w_{|\mathcal{V}|}, d)]^\top \in \mathbb{R}^{|\mathcal{V}|}
$$

where $c(w_i, d)$ is the count of word $w_i$ in document $d$.

**Example**:
- Document: "The cat sat on the mat"
- Vocabulary: $\mathcal{V} = \{\text{the, cat, sat, on, mat, dog}\}$
- BoW vector: $[2, 1, 1, 1, 1, 0]^\top$ (word "the" appears twice)

**Normalization Variants**:

**Binary BoW** (presence/absence):
$$\text{BoW}_{\text{binary}}(d) = [\mathbb{I}[c(w_1, d) > 0], \ldots, \mathbb{I}[c(w_{|\mathcal{V}|}, d) > 0]]^\top$$

**Normalized BoW** (term frequency):
$$\text{BoW}_{\text{norm}}(d) = \left[\frac{c(w_1, d)}{|d|}, \ldots, \frac{c(w_{|\mathcal{V}|}, d)}{|d|}\right]^\top$$

where $|d| = \sum_i c(w_i, d)$ is total word count.

**Critical Limitation**: Word order is completely lost. The sentences:
- "The cat sat on the mat"
- "The mat sat on the cat"

produce **identical** BoW representations $[2, 1, 1, 1, 1, 0]^\top$, despite having completely different meanings.

**Representation Method 2: TF-IDF (Term Frequency-Inverse Document Frequency)**

Weight words by importance—frequent in this document but rare across corpus—to identify discriminative terms.

**Mathematical Formulation**: For term $t$ in document $d$ from corpus $\mathcal{C}$ of $N$ documents:

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

where:

**Term Frequency** (normalized count):
$$
\text{TF}(t, d) = \frac{c(t, d)}{\sum_{t' \in d} c(t', d)} = \frac{c(t, d)}{|d|}
$$

**Inverse Document Frequency** (logarithmic scaling):
$$
\text{IDF}(t) = \log \frac{N}{\text{DF}(t)} = \log \frac{N}{|\{d \in \mathcal{C} : t \in d\}|}
$$

where $\text{DF}(t)$ is the number of documents containing term $t$.

**Intuition Behind IDF**:

- **High IDF** ($\text{DF}(t)$ small): Term appears in few documents → discriminative power
  - Example: "photosynthesis" appears in 50 out of 10,000 documents
  - $\text{IDF}(\text{"photosynthesis"}) = \log(10000/50) = \log(200) \approx 5.3$

- **Low IDF** ($\text{DF}(t)$ large): Term appears in most documents → little discriminative power
  - Example: "the" appears in 9,950 out of 10,000 documents
  - $\text{IDF}(\text{"the"}) = \log(10000/9950) \approx 0.005$

**Complete TF-IDF Example**:

Corpus: 10,000 news articles  
Document A (500 words): "election" appears 50 times, appears in 100 documents total

$$
\begin{aligned}
\text{TF}(\text{"election"}, A) &= \frac{50}{500} = 0.1 \\
\text{IDF}(\text{"election"}) &= \log\frac{10000}{100} = \log(100) \approx 4.605 \\
\text{TF-IDF}(\text{"election"}, A) &= 0.1 \times 4.605 = 0.461
\end{aligned}
$$

**Document Vector**: Full TF-IDF representation:

$$
\mathbf{x}_d = [\text{TF-IDF}(w_1, d), \ldots, \text{TF-IDF}(w_{|\mathcal{V}|}, d)]^\top \in \mathbb{R}^{|\mathcal{V}|}
$$

**Variants**:

**Sublinear TF Scaling** (dampen effect of very frequent terms):
$$\text{TF}_{\text{log}}(t, d) = 1 + \log c(t, d)$$

**Smoothed IDF** (prevent division by zero):
$$\text{IDF}_{\text{smooth}}(t) = \log \frac{N + 1}{\text{DF}(t) + 1} + 1$$

**Classification Algorithm 1: Naive Bayes Classifier**

**Core Assumption**: Features (words) are conditionally independent given the class label—a "naive" assumption severely violated in natural language.

**Theoretical Foundation (Bayes' Theorem)**:

$$
P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \cdot P(y)}{P(\mathbf{x})}
$$

where:
- $P(y \mid \mathbf{x})$: **Posterior probability** of class $y$ given features $\mathbf{x}$
- $P(\mathbf{x} \mid y)$: **Likelihood** of observing features $\mathbf{x}$ in class $y$
- $P(y)$: **Prior probability** of class $y$
- $P(\mathbf{x})$: **Evidence** (constant for all classes)

**Naive Independence Assumption**:

For features $\mathbf{x} = [x_1, x_2, \ldots, x_n]$:

$$
P(\mathbf{x} \mid y) = P(x_1, x_2, \ldots, x_n \mid y) \stackrel{\text{naive}}{=} \prod_{i=1}^{n} P(x_i \mid y)
$$

This assumes features are independent given class label:
$$P(x_i \mid x_j, y) = P(x_i \mid y) \quad \forall i \neq j$$

**Classification Decision Rule**:

Since $P(\mathbf{x})$ is constant, maximize posterior probability:

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

Taking logarithm for numerical stability:

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} \left[ \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right]
$$

**Parameter Estimation from Training Data**:

**Prior Probability** (class frequency):
$$
P(y) = \frac{\text{Number of documents in class } y}{\text{Total number of documents}} = \frac{N_y}{N}
$$

**Likelihood** (word frequency in class):
$$
P(x_i \mid y) = \frac{\text{Count of feature } x_i \text{ in class } y}{\text{Total features in class } y} = \frac{c(x_i, y)}{\sum_{x' \in \mathcal{V}} c(x', y)}
$$

**Laplace Smoothing** (handle zero counts):

$$
P(x_i \mid y) = \frac{c(x_i, y) + \alpha}{\sum_{x' \in \mathcal{V}} [c(x', y) + \alpha]} = \frac{c(x_i, y) + \alpha}{N_y + \alpha |\mathcal{V}|}
$$

where $\alpha > 0$ is smoothing parameter (typically $\alpha = 1$).

**Concrete Example**:

Training data:
- 100 sports articles, 100 politics articles
- Word "goal" appears 50 times in sports, 5 times in politics
- Total words in sports: 10,000; in politics: 10,000

**Probability Calculations**:

$$
\begin{aligned}
P(\text{sports}) &= \frac{100}{200} = 0.5 \\
P(\text{politics}) &= \frac{100}{200} = 0.5 \\
P(\text{"goal"} \mid \text{sports}) &= \frac{50}{10000} = 0.005 \\
P(\text{"goal"} \mid \text{politics}) &= \frac{5}{10000} = 0.0005
\end{aligned}
$$

**Likelihood Ratio**:
$$\frac{P(\text{"goal"} \mid \text{sports})}{P(\text{"goal"} \mid \text{politics})} = \frac{0.005}{0.0005} = 10$$

The word "goal" is 10× more likely in sports articles—strong discriminative signal.

**Advantages**:
1. **Computational Efficiency**: Training is $O(N \cdot |\mathcal{V}|)$ (simple counting)
2. **Low Sample Complexity**: Works with small datasets (few parameters: $K \times |\mathcal{V}|$ probabilities)
3. **Interpretable**: Can inspect $P(word \mid class)$ to understand decisions
4. **Probabilistic Outputs**: Provides confidence scores, not just hard classifications

**Limitations**:
1. **Independence Assumption Violated**: Words are highly correlated in natural language
   - "New York" treated as independent "New" and "York"
   - Cannot capture phrases like "not good"
2. **No Word Order**: "dog bites man" vs. "man bites dog" have identical representation
3. **Zero-Frequency Problem**: If word never appears in class during training, $P(word \mid class) = 0$ makes entire probability zero

**Classification Algorithm 2: Support Vector Machines (SVM)**

**Core Idea**: Find the hyperplane that maximally separates classes in feature space, maximizing the **margin** (distance to nearest points).

**Binary Classification Formulation**:

For linearly separable data $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ where $y_i \in \{-1, +1\}$, find hyperplane:

$$
\mathbf{w}^\top \mathbf{x} + b = 0
$$

that satisfies:
$$
y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

**Geometric Margin**: Distance from hyperplane to nearest point:

$$
\gamma = \min_i \frac{|\ \mathbf{w}^\top \mathbf{x}_i + b|}{|\mathbf{w}|} = \frac{1}{|\mathbf{w}|}
$$

**Optimization Objective**: Maximize margin $\Leftrightarrow$ Minimize $|\mathbf{w}|$:

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2} \|\mathbf{w}\|^2 \\
\text{subject to} \quad & y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, N
\end{aligned}
$$

**Soft-Margin SVM** (allow misclassifications for non-separable data):

Introduce slack variables $\xi_i \geq 0$ measuring constraint violations:

$$
\begin{aligned}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad & \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \\
\text{subject to} \quad & y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad i = 1, \ldots, N
\end{aligned}
$$

where:
- $C > 0$: Regularization parameter balancing margin maximization vs. training error
  - Large $C$: Penalize violations heavily (small margin, low training error, high risk of overfitting)
  - Small $C$: Allow more violations (large margin, higher training error, better generalization)

**Dual Formulation** (enables kernel trick):

Using Lagrange multipliers $\alpha_i \geq 0$:

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad & \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j \\
\text{subject to} \quad & 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^N \alpha_i y_i = 0
\end{aligned}
$$

**Decision Function**:

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^N \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b\right)
$$

**Support Vectors**: Training points with $\alpha_i > 0$ (lie on margin boundary or violate it). Only these points determine the decision boundary—most training data can be discarded!

**The Kernel Trick**: Map data to higher-dimensional space where linear separation is possible, without explicitly computing the mapping.

**Kernel Function**: 
$$K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle$$

where $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^D$ maps to (possibly infinite-dimensional) feature space.

**Decision Function with Kernels**:

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^N \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

**Common Kernels**:

**Linear Kernel** (no transformation):
$$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$$

**Polynomial Kernel** (degree $d$):
$$K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^\top \mathbf{x}' + r)^d$$

**Radial Basis Function (RBF/Gaussian) Kernel**:
$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$

where $\gamma = \frac{1}{2\sigma^2}$ controls smoothness.

**Example**: Linear kernel cannot separate XOR problem:
- Points: $(0,0) \rightarrow -1$, $(0,1) \rightarrow +1$, $(1,0) \rightarrow +1$, $(1,1) \rightarrow -1$

But polynomial kernel of degree 2 maps to 3D space where linear separation exists.

**Advantages**:
1. **Effective in High Dimensions**: Text data with 50,000+ dimensions
2. **Memory Efficient**: Only store support vectors (typically 10-30% of training data)
3. **Kernel Flexibility**: Can handle non-linear decision boundaries
4. **Theoretical Guarantees**: Maximum margin reduces generalization error (VC theory)

**Limitations**:
1. **Computational Complexity**: Training is $O(N^2)$ to $O(N^3)$ (quadratic programming)
   - Prohibitive for $N > 100,000$ (modern datasets have millions)
2. **Hyperparameter Sensitivity**: Requires careful tuning of $C$, $\gamma$ (kernel parameters)
3. **No Probabilistic Interpretation**: Outputs decision values, not probabilities
   - (Platt scaling can calibrate to probabilities post-hoc)
4. **Binary Classification**: Requires one-vs-rest or one-vs-one decomposition for multi-class

**Classification Algorithm 3: Logistic Regression**

**Core Idea**: Model posterior class probability using the logistic (sigmoid) function, ensuring outputs lie in $[0, 1]$.

**Binary Logistic Regression**:

For binary classification $y \in \{0, 1\}$:

$$
P(y = 1 \mid \mathbf{x}; \mathbf{w}, b) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp(-(\mathbf{w}^\top \mathbf{x} + b))}
$$

where $\sigma(z)$ is the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

**Sigmoid Properties**:
- $\sigma(0) = 0.5$ (decision boundary)
- $\lim_{z \to \infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (convenient for gradient computation)

**Multi-Class Extension (Softmax Regression)**:

For $K$ classes, define linear score for each class:

$$
z_k = \mathbf{w}_k^\top \mathbf{x} + b_k, \quad k = 1, \ldots, K
$$

Apply **softmax function** to convert scores to probability distribution:

$$
P(y = k \mid \mathbf{x}; \mathbf{W}, \mathbf{b}) = \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)} = \frac{\exp(\mathbf{w}_k^\top \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^\top \mathbf{x} + b_j)}
$$

**Softmax Properties**:
- $\sum_{k=1}^K P(y = k \mid \mathbf{x}) = 1$ (valid probability distribution)
- $P(y = k \mid \mathbf{x}) \in (0, 1)$ (all probabilities positive)
- $\arg\max_k P(y = k \mid \mathbf{x}) = \arg\max_k z_k$ (invariant to constant shifts)

**Training: Maximum Likelihood Estimation**

**Likelihood** of observing data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$:

$$
\mathcal{L}(\mathbf{W}, \mathbf{b}) = \prod_{i=1}^N P(y_i \mid \mathbf{x}_i; \mathbf{W}, \mathbf{b})
$$

**Log-Likelihood** (easier to optimize):

$$
\log \mathcal{L}(\mathbf{W}, \mathbf{b}) = \sum_{i=1}^N \log P(y_i \mid \mathbf{x}_i; \mathbf{W}, \mathbf{b})
$$

**Negative Log-Likelihood (Cross-Entropy Loss)**:

$$
\mathcal{L}_{\text{CE}}(\mathbf{W}, \mathbf{b}) = -\sum_{i=1}^N \log P(y_i \mid \mathbf{x}_i; \mathbf{W}, \mathbf{b})
$$

For multi-class with one-hot encoding $\mathbf{y}_i = [0, \ldots, 1, \ldots, 0]$ (1 at position $y_i$):

$$
\mathcal{L}_{\text{CE}}(\mathbf{W}, \mathbf{b}) = -\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log P(y = k \mid \mathbf{x}_i; \mathbf{W}, \mathbf{b})
$$

**Regularized Objective** (prevent overfitting):

$$
\min_{\mathbf{W}, \mathbf{b}} \quad \mathcal{L}_{\text{CE}}(\mathbf{W}, \mathbf{b}) + \lambda \|\mathbf{W}\|_2^2
$$

where $\lambda > 0$ controls regularization strength (L2 penalty).

**Optimization**: Gradient descent or quasi-Newton methods (L-BFGS)

**Gradient Computation**:

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial \mathbf{w}_k} = \sum_{i=1}^N (\hat{y}_{ik} - y_{ik}) \mathbf{x}_i
$$

where $\hat{y}_{ik} = P(y = k \mid \mathbf{x}_i)$ is predicted probability.

**Advantages**:
1. **Fast Training**: Convex optimization guarantees global optimum
2. **Probabilistic Outputs**: Well-calibrated confidence scores
3. **Interpretable**: Weight $w_k^{(j)}$ shows contribution of feature $j$ to class $k$
4. **Regularization**: L1 (Lasso) induces sparsity, L2 (Ridge) prevents overfitting

**Limitations**:
1. **Linear Decision Boundaries**: Cannot model XOR-like patterns without feature engineering
2. **Feature Independence Assumption**: Like Naive Bayes, assumes features are independent
3. **Requires Feature Engineering**: Manual construction of informative features (n-grams, POS tags)

**Fundamental Limitation of All Classical Methods**:

All these approaches treat words as atomic units with fixed representations, failing to capture:

1. **Semantic Similarity**: "car" and "automobile" are treated as completely different features despite identical meaning
2. **Contextual Meaning**: "bank" receives the same representation in "financial bank" vs. "river bank"
3. **Compositional Semantics**: "not good" is represented as independent "not" and "good", losing the negation relationship

This motivated the paradigm shift to learned distributed representations in Phase 2.

#### Phase 2: Neural Embeddings and Deep Learning (2010-2017)

**Revolutionary Insight: The Distributional Hypothesis**

> "You shall know a word by the company it keeps" — J.R. Firth (1957)

Words appearing in similar contexts should have similar meanings. This principle enables learning dense vector representations from word co-occurrence patterns in large unlabeled corpora.

**Paradigm Shift**: Instead of treating words as atomic symbols with arbitrary IDs, represent them as **continuous vectors** in a learned semantic space where geometric relationships correspond to semantic relationships.

**Word2Vec: Neural Embedding Learning (Mikolov et al., 2013)**

Two complementary architectures for learning distributed word representations:

**Architecture 1: Continuous Bag-of-Words (CBOW)**

**Objective**: Predict center word from surrounding context words.

Given context window of size $c$, predict target word $w_t$ from context $\{w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}\}$.

**Model Architecture**:

1. **Input Layer**: One-hot encoded context words $\mathbf{x}_{t-c}, \ldots, \mathbf{x}_{t-1}, \mathbf{x}_{t+1}, \ldots, \mathbf{x}_{t+c} \in \{0,1\}^{|\mathcal{V}|}$

2. **Embedding Layer**: Map to dense vectors via embedding matrix $\mathbf{E} \in \mathbb{R}^{d \times |\mathcal{V}|}$:
   $$\mathbf{v}_{t+j} = \mathbf{E} \mathbf{x}_{t+j} \in \mathbb{R}^d$$

3. **Context Representation**: Average context embeddings:
   $$\mathbf{h} = \frac{1}{2c} \sum_{j \in \{-c, \ldots, -1, 1, \ldots, c\}} \mathbf{v}_{t+j}$$

4. **Output Layer**: Predict target word via softmax over vocabulary:
   $$P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{h})}$$
   
   where $\mathbf{u}_w \in \mathbb{R}^d$ is the output embedding for word $w$.

**Training Objective**: Maximize log-likelihood over corpus:

$$
\mathcal{L}_{\text{CBOW}} = \sum_{t=1}^T \log P(w_t \mid w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c})
$$

**Architecture 2: Skip-Gram**

**Objective**: Predict context words from center word (inverse of CBOW).

Given target word $w_t$, predict each context word $w_{t+j}$ independently.

**Model**: For each context position $j \in \{-c, \ldots, -1, 1, \ldots, c\}$:

$$
P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}
$$

where:
- $\mathbf{v}_{w_t} \in \mathbb{R}^d$: Input embedding of center word
- $\mathbf{u}_{w_{t+j}} \in \mathbb{R}^d$: Output embedding of context word

**Training Objective**: Maximize log-likelihood:

$$
\mathcal{L}_{\text{Skip-gram}} = \sum_{t=1}^T \sum_{j \in \{-c, \ldots, c\}, j \neq 0} \log P(w_{t+j} \mid w_t)
$$

**Computational Challenge**: Softmax denominator requires summing over entire vocabulary (50,000+ terms) for each prediction—computationally prohibitive.

**Solution 1: Hierarchical Softmax**

Replace flat softmax with binary tree structure (Huffman tree based on word frequency).

**Probability Computation**: Path from root to word $w$ with $L(w)$ nodes:

$$
P(w \mid w_t) = \prod_{i=1}^{L(w)-1} \sigma\left(\text{&#10214;} n(w, i+1) = \text{left}(n(w, i)) \text{&#10215;} \cdot \mathbf{u}_{n(w,i)}^\top \mathbf{v}_{w_t}\right)
$$

where:
- $n(w, i)$: $i$-th node on path to word $w$
- $\text{Indicator } \text{&#10214;}\cdot\text{&#10215;} : 1 \text{ if true, } -1 \text{ if false}$
- $\sigma(z) = 1/(1 + e^{-z})$: Sigmoid function

**Complexity Reduction**: $O(|\mathcal{V}|) \rightarrow O(\log |\mathcal{V}|)$ per word

**Solution 2: Negative Sampling**

Approximate softmax by discriminating target word from $k$ random "negative" samples.

**Modified Objective**: For each target word $w_t$ and context $w_c$, sample $k$ negative words $w_i \sim P_{\text{noise}}$:

$$
\mathcal{L}_{\text{NEG}} = \log \sigma(\mathbf{u}_{w_c}^\top \mathbf{v}_{w_t}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_{\text{noise}}} \left[\log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_t})\right]
$$

**Noise Distribution**: Empirically, unigram distribution raised to power 3/4 works best:

$$
P_{\text{noise}}(w) = \frac{f(w)^{3/4}}{\sum_{w' \in \mathcal{V}} f(w')^{3/4}}
$$

where $f(w)$ is word frequency.

**Intuition**:
- Maximize probability that target word appears in context: &#963;(<i>𝘶</i><sub>w<sub>c</sub></sub><sup>T</sup><i>𝘷</i><sub>w<sub>t</sub></sub>) &#8594; 1  
- Minimize probability that random words appear: &#963;(<i>𝘶</i><sub>w<sub>i</sub></sub><sup>T</sup><i>𝘷</i><sub>w<sub>t</sub></sub>) &#8594; 0

**Complexity**: $O(|\mathcal{V}|) \rightarrow O(k)$ per word, typically $k=5-20$

**Emergent Semantic Properties**

After training on billions of words (e.g., Google News corpus: 100B tokens), word vectors exhibit remarkable **linear regularities**:

**Semantic Analogies**:

$$
\mathbf{v}(\text{king}) - \mathbf{v}(\text{man}) + \mathbf{v}(\text{woman}) \approx \mathbf{v}(\text{queen})
$$

$$
\mathbf{v}(\text{Paris}) - \mathbf{v}(\text{France}) + \mathbf{v}(\text{Italy}) \approx \mathbf{v}(\text{Rome})
$$

**Syntactic Analogies**:

$$
\mathbf{v}(\text{walking}) - \mathbf{v}(\text{walk}) \approx \mathbf{v}(\text{swimming}) - \mathbf{v}(\text{swim})
$$

(verb tense transformation)

$$
\mathbf{v}(\text{slow}) - \mathbf{v}(\text{slower}) \approx \mathbf{v}(\text{fast}) - \mathbf{v}(\text{faster})
$$

(comparative form)

**Cosine Similarity Clustering**: Semantically related words have high cosine similarity:

$$
\cos(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i^\top \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}
$$

**Examples**:
- Countries: $\{\text{France}, \text{Germany}, \text{Italy}, \text{Spain}\}$ have pairwise similarity $>0.7$
- Sports: $\{\text{football}, \text{basketball}, \text{tennis}, \text{soccer}\}$ have pairwise similarity $>0.6$

**Geometric Interpretation**: Word vectors organize into coherent semantic and syntactic subspaces where:
- **Direction** encodes relationships (gender, tense, plurality)
- **Distance** measures semantic similarity

**GloVe: Global Vectors for Word Representation (Pennington et al., 2014)**

**Motivation**: Word2Vec relies on local context windows, potentially missing global corpus statistics.

**Core Idea**: Directly factorize word co-occurrence matrix to leverage global statistics.

**Co-occurrence Matrix**: Define $X_{ij}$ as number of times word $j$ appears in context of word $i$:

$$
X_{ij} = \sum_{t=1}^T \sum_{k=-c}^c \mathbb{I}[w_t = i \wedge w_{t+k} = j]
$$

**Objective**: Learn vectors such that their dot product equals log co-occurrence:

$$
\mathbf{w}_i^\top \mathbf{w}_j + b_i + b_j \approx \log X_{ij}
$$

where $b_i, b_j$ are bias terms for words $i, j$.

**Weighted Least Squares Loss**:

$$
\mathcal{L}_{\text{GloVe}} = \sum_{i,j=1}^{|\mathcal{V}|} f(X_{ij}) \left(\mathbf{w}_i^\top \mathbf{w}_j + b_i + b_j - \log X_{ij}\right)^2
$$

**Weighting Function**: Prevent common word pairs from dominating:

$$
f(x) = \begin{cases}
(x / x_{\max})^\alpha & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}
$$

Typical values: $x_{\max} = 100$, $\alpha = 0.75$.

**Intuition**:
- Rare co-occurrences ($X_{ij}$ small): Low weight (unreliable statistics)
- Very common co-occurrences ($X_{ij} > x_{\max}$): Capped weight (prevent dominance)
- Intermediate frequencies: Highest relative weight

**Advantage over Word2Vec**: Captures global corpus statistics, not just local windows. Empirically achieves better performance on word analogy tasks (75% → 80% accuracy).

**Neural Classification Architectures**

**Convolutional Neural Networks for Text (Kim, 2014)**

**Architecture Pipeline**:

1. **Embedding Layer**: Map words to dense vectors
   $$\mathbf{x}_{1:n} = [\mathbf{v}_{w_1}, \mathbf{v}_{w_2}, \ldots, \mathbf{v}_{w_n}]$$
   where each $\mathbf{v}_{w_i} \in \mathbb{R}^d$ (typically $d=300$ from pre-trained Word2Vec/GloVe)

2. **Convolutional Filters**: Detect n-gram patterns

   For filter $\mathbf{W} \in \mathbb{R}^{k \times d}$ of height $k$ (n-gram size):
   
   $$c_i = f\left(\mathbf{W} \cdot \mathbf{x}_{i:i+k-1} + b\right)$$
   
   where:
   - $\mathbf{x}_{i:i+k-1} = [\mathbf{v}_{w_i}; \mathbf{v}_{w_{i+1}}; \ldots; \mathbf{v}_{w_{i+k-1}}] \in \mathbb{R}^{kd}$ is concatenation of $k$ consecutive word vectors
   - $f$ is activation function (typically ReLU: $f(z) = \max(0, z)$)
   - $b \in \mathbb{R}$ is bias term

   This produces feature map:
   $$\mathbf{c} = [c_1, c_2, \ldots, c_{n-k+1}] \in \mathbb{R}^{n-k+1}$$

3. **Max-Pooling**: Extract most important feature from each filter
   $$\hat{c} = \max(\mathbf{c}) = \max\{c_1, c_2, \ldots, c_{n-k+1}\}$$

4. **Multiple Filter Sizes**: Use filters of different heights ($k \in \{3, 4, 5\}$) with $m$ filters per size
   $$\mathbf{z} = [\hat{c}_1^{(3)}, \ldots, \hat{c}_m^{(3)}, \hat{c}_1^{(4)}, \ldots, \hat{c}_m^{(4)}, \hat{c}_1^{(5)}, \ldots, \hat{c}_m^{(5)}] \in \mathbb{R}^{3m}$$

5. **Fully Connected Layer**: Classification
   $$\mathbf{y} = \text{softmax}(\mathbf{W}_{\text{fc}} \mathbf{z} + \mathbf{b}_{\text{fc}})$$

**Intuition**: 
- 3-gram filters detect trigrams: "not very good", "extremely happy"
- 4-gram filters detect 4-word phrases: "very easy to use"
- 5-gram filters detect longer patterns: "I would highly recommend this"

**Advantages**:
1. **Captures Local Patterns**: N-grams indicative of sentiment/topic
2. **Translation Invariant**: Same filter applied everywhere
3. **Parallel Computation**: All filters computed simultaneously (fast on GPUs)
4. **Pre-trained Embeddings**: Initialize with Word2Vec/GloVe (transfer learning)

**Limitations**:
1. **Fixed Receptive Field**: Cannot capture dependencies beyond n-gram size
2. **Loses Long-Range Order**: Max-pooling discards position information
3. **No Sequential Dependencies**: Unlike RNNs, doesn't model word order globally

**Recurrent Neural Networks: Long Short-Term Memory (LSTM)**

**Motivation**: Standard RNNs suffer from **vanishing gradient problem**—gradients decay exponentially with sequence length, preventing learning of long-term dependencies.

**Vanishing Gradient in Standard RNN**:

For RNN $\mathbf{h}_t = \tanh(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{U} \mathbf{x}_t)$, gradient at step $t$ w.r.t. step $t-k$:

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}} = \prod_{i=t-k+1}^t \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} = \prod_{i=t-k+1}^t \mathbf{W}^\top \text{diag}[\tanh'(\cdot)]
$$

Since $|\tanh'(z)| \leq 1$ and typically $\|\mathbf{W}\| < 1$ for stability, gradients decay as:

$$
\left\|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-k}}\right\| \approx \gamma^k \quad \text{where } \gamma < 1
$$

For $k=100$ and $\gamma = 0.9$: gradient is $0.9^{100} \approx 10^{-5}$ (vanished!).

**LSTM Solution (Hochreiter & Schmidhuber, 1997)**

Introduce **gating mechanisms** to control information flow, enabling gradients to flow unchanged across many time steps.

**Four Components**:

**1. Forget Gate** $\mathbf{f}_t$ (what to discard from cell state):

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

where $\sigma(z) = 1/(1+e^{-z})$ is sigmoid function outputting values in $(0, 1)$.

**2. Input Gate** $\mathbf{i}_t$ (what new information to store):

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

**Candidate Values** $\tilde{\mathbf{C}}_t$ (potential new information):

$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)
$$

**3. Cell State Update** $\mathbf{C}_t$ (memory highway):

$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t
$$

where $\odot$ is element-wise (Hadamard) product.

**Interpretation**:
- $\mathbf{f}_t \odot \mathbf{C}_{t-1}$: Selectively forget old memory (forget gate acts as filter)
- $\mathbf{i}_t \odot \tilde{\mathbf{C}}_t$: Selectively add new information (input gate controls flow)

**4. Output Gate** $\mathbf{o}_t$ (what to output):

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

**Hidden State**:

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
$$

**Geometric Interpretation**:

- **Cell State $\mathbf{C}_t$**: "Memory highway" carrying information across time with minimal transformation
- **Gates** (sigmoid outputs $\in (0,1)$): Differentiable switches
  - $\sigma(z) \approx 1$: Gate open (information flows)
  - $\sigma(z) \approx 0$: Gate closed (information blocked)

**Gradient Flow**: Crucial property enabling long-term learning:

$$
\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t
$$

Since $\mathbf{f}_t \in (0,1)$ is learned (not fixed like $\mathbf{W}$ in RNN), the network can learn to set $\mathbf{f}_t \approx 1$ for important information, allowing gradients to flow unchanged across 100+ steps.

**Concrete Example**: 

Sentence: "The cat, which I saw yesterday in the park while I was walking, was sleeping."

**Challenge**: Predict verb "was" (singular) requiring subject "cat" from 12 words earlier.

**LSTM Behavior**:
1. At "cat": Input gate $\mathbf{i}_t$ opens, stores "singular subject" in cell state $\mathbf{C}_t$
2. During intervening clause: Forget gate $\mathbf{f}_t \approx 1$ preserves "singular" information
3. At "was": Output gate $\mathbf{o}_t$ opens, retrieves "singular" → selects "was" (not "were")

**Bidirectional LSTM (BiLSTM)**:

Process sequence in both directions:

$$
\begin{aligned}
\overrightarrow{\mathbf{h}}_t &= \text{LSTM}(\overrightarrow{\mathbf{h}}_{t-1}, \mathbf{x}_t) \quad \text{(forward)} \\
\overleftarrow{\mathbf{h}}_t &= \text{LSTM}(\overleftarrow{\mathbf{h}}_{t+1}, \mathbf{x}_t) \quad \text{(backward)} \\
\mathbf{h}_t &= [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \quad \text{(concatenate)}
\end{aligned}
$$

**Advantage**: Access to both past and future context (crucial for classification where entire document is available).

**Advantages of LSTMs**:
1. **Long-Range Dependencies**: Captures patterns across 100+ tokens
2. **Variable Length**: Naturally handles sequences of any length
3. **Sequential Structure**: Models inherent word order
4. **Bidirectionality**: BiLSTM sees full context

**Limitations**:
1. **Sequential Processing**: Cannot parallelize across time steps (slow training)
2. **Still Limited**: Struggles with 500+ token dependencies
3. **Computational Cost**: Requires two passes for BiLSTM

**Critical Limitation of Phase 2: Context-Independent Embeddings**

**Fundamental Problem**: Word2Vec and GloVe produce **static embeddings**—each word receives a single fixed vector regardless of context.

**Example Failure**: Word "bank"

Word2Vec assigns fixed vector $\mathbf{v}_{\text{bank}}$ used in both:
- "I deposited money at the **bank**" (financial institution)
- "We sat by the river **bank**" (land alongside water)

These usages have completely different meanings, but receive identical representations!

**Attempted Solution**: Use LSTM to compute contextualized representations:

$$
\mathbf{h}_t = \text{BiLSTM}(\mathbf{v}_{w_1}, \ldots, \mathbf{v}_{w_t}, \ldots, \mathbf{v}_{w_n})
$$

**Remaining Issue**: LSTM still initializes from static embeddings and suffers from:
- Sequential processing bottleneck
- Limited context window (100-200 tokens effective range)
- Difficulty modeling very long-range dependencies

**This limitation motivated Phase 3: Attention mechanisms enabling truly context-dependent representations with global receptive field.**

---

#### Phase 3: Attention Mechanisms and Transformers (2017-2019)

**The Attention Revolution**

**Core Insight**: Instead of forcing the network to compress entire sequence into fixed-size vector, allow it to selectively **attend** to relevant parts for each prediction.

**Self-Attention Mechanism**

**Motivation**: For each token, compute representation as weighted combination of all tokens in sequence, with weights determined by relevance.

**Mathematical Formulation**:

Given input sequence of $n$ tokens represented as matrix:

$$
\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}
$$

where each $\mathbf{x}_i \in \mathbb{R}^d$ is token embedding.

**Step 1: Linear Projections**

Transform input to three representations via learned matrices:

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{n \times d_k} \quad \text{(Queries)} \\
\mathbf{K} &= \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{n \times d_k} \quad \text{(Keys)} \\
\mathbf{V} &= \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{n \times d_v} \quad \text{(Values)}
\end{aligned}
$$

where $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$ are learned projection matrices.

**Interpretation**:
- **Query** $\mathbf{q}_i$: "What am I looking for?" (what information does token $i$ need)
- **Key** $\mathbf{k}_j$: "What information do I contain?" (what token $j$ offers)
- **Value** $\mathbf{v}_j$: "What information do I provide?" (actual content from token $j$)

**Step 2: Compute Attention Scores**

Measure relevance between all token pairs via dot product:

$$
\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{n \times n}
$$

where $S_{ij} = \mathbf{q}_i^\top \mathbf{k}_j$ measures compatibility between query $i$ and key $j$.

**Scaled Dot-Product** (prevent gradients from vanishing):

$$
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}
$$

**Why scaling?** For random vectors $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with unit variance:

$$
\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = 0, \quad \text{Var}[\mathbf{q}^\top \mathbf{k}] = d_k
$$

Dot products grow with dimension → softmax saturates → gradients vanish. Dividing by $\sqrt{d_k}$ maintains unit variance.

**Step 3: Attention Weights**

Convert scores to probability distribution via softmax (row-wise):

$$
\mathbf{A} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{n \times n}
$$

$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^n \exp(S_{ik})}
$$

**Properties**:
- Each row sums to 1: $\sum_{j=1}^n A_{ij} = 1$
- All values positive: $A_{ij} \in (0, 1)$
- $A_{ij}$ represents "how much token $i$ attends to token $j$"

**Step 4: Weighted Aggregation**

Compute output as attention-weighted sum of values:

$$
\mathbf{Z} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{n \times d_v}
$$

$$
\mathbf{z}_i = \sum_{j=1}^n A_{ij} \mathbf{v}_j
$$

**Complete Self-Attention Formula**:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

**Concrete Example**:

Sentence: "The cat sat on the mat"

For token "sat" (query):
- High attention to "cat" (subject performing action): $A_{\text{sat,cat}} = 0.45$
- High attention to "mat" (object of preposition): $A_{\text{sat,mat}} = 0.30$
- Moderate attention to "on" (preposition): $A_{\text{sat,on}} = 0.15$
- Low attention to "the": $A_{\text{sat,the}} = 0.05$ each

Output representation $\mathbf{z}_{\text{sat}}$ is weighted combination:

$$
\mathbf{z}_{\text{sat}} = 0.45 \mathbf{v}_{\text{cat}} + 0.30 \mathbf{v}_{\text{mat}} + 0.15 \mathbf{v}_{\text{on}} + \ldots
$$

This representation captures that "sat" relates primarily to "cat" and "mat"—syntactic and semantic structure discovered automatically!

**Multi-Head Attention**

**Motivation**: Single attention mechanism may focus on one relationship type (e.g., syntactic). Multiple heads can capture different relationship types in parallel.

**Formulation**: Run $h$ attention heads with different projection matrices:

$$
\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_Q^i, \mathbf{K} \mathbf{W}_K^i, \mathbf{V} \mathbf{W}_V^i)
$$

where $\mathbf{W}_Q^i, \mathbf{W}_K^i \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}_V^i \in \mathbb{R}^{d \times d_v}$ are learned parameters for head $i$.

**Concatenate and Project**:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O
$$

where $\mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d}$ projects back to original dimension.

**Typical Configuration**: 
- BERT-Base: $h=12$ heads, $d_k = d_v = d/h = 768/12 = 64$
- Each head has 64-dimensional queries/keys/values

**Empirical Finding**: Different heads specialize in different patterns:

- **Head 1**: Subject-verb agreement ("cat **was**" vs. "cats **were**")
- **Head 2**: Object-verb relationships
- **Head 3**: Prepositional attachments
- **Head 4**: Coreference resolution (pronouns → antecedents: "John ... **he**")
- **Head 5**: Positional proximity (adjacent words)
- **Head 6**: Semantic similarity (synonyms, related concepts)

**Visualization**: Attention patterns reveal linguistic structure without explicit supervision!

**The Transformer Architecture (Vaswani et al., 2017)**

**Revolutionary Design**: Entirely based on attention, completely removing recurrence and convolution.

**Encoder Architecture** (for classification):

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────┐
│  Encoder Block (×N layers)      │
│  ┌───────────────────────────┐  │
│  │ Multi-Head Self-Attention │  │
│  └───────────────────────────┘  │
│            ↓                     │
│      Add & Normalize             │
│            ↓                     │
│  ┌───────────────────────────┐  │
│  │  Feed-Forward Network     │  │
│  └───────────────────────────┘  │
│            ↓                     │
│      Add & Normalize             │
└─────────────────────────────────┘
    ↓
Classification Head (pooling + linear)
    ↓
Output Probabilities
```

**Key Components**:

**1. Positional Encoding**

**Problem**: Attention is **permutation invariant**—reordering tokens doesn't change attention output. But word order matters in language!

**Solution**: Add position-dependent patterns to input embeddings.

**Sinusoidal Encoding** (original Transformer):

$$
\begin{aligned}
PE_{(\text{pos}, 2i)} &= \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right) \\
PE_{(\text{pos}, 2i+1)} &= \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)
\end{aligned}
$$

where:
- $\text{pos} \in \{0, 1, \ldots, n-1\}$: Position in sequence
- $i \in \{0, 1, \ldots, d/2-1\}$: Dimension index
- Even dimensions use sine, odd use cosine

**Properties**:
- Each position has unique encoding
- Relative positions have consistent patterns: $PE_{\text{pos}+k}$ is linear function of $PE_{\text{pos}}$ (enables learning of relative position relationships)
- Extrapolates to longer sequences than seen during training

**Alternative: Learned Positional Embeddings** (BERT):

$$
PE_{\text{pos}} = \mathbf{W}_{\text{pos}}[\text{pos}] \in \mathbb{R}^d
$$

where $\mathbf{W}_{\text{pos}} \in \mathbb{R}^{n_{\max} \times d}$ is learned embedding matrix for positions up to $n_{\max}$.

**Input to First Layer**:

$$
\mathbf{x}_i^{(0)} = \text{TokenEmbed}(w_i) + PE_i
$$

**2. Feed-Forward Network**

Applied independently to each position (no interaction between positions):

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$: Expand dimension (typically $d_{\text{ff}} = 4d$)
- $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$: Project back
- ReLU activation: $\max(0, \cdot)$

**Intuition**: 
- Self-attention mixes information across positions
- FFN processes each position independently to extract features

**3. Layer Normalization**

Normalize activations across feature dimension (not batch like BatchNorm):

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:
- $\mu = \frac{1}{d} \sum_{i=1}^d x_i$: Mean across features
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$: Variance
- $\gamma, \beta \in \mathbb{R}^d$: Learned scale and shift parameters
- $\epsilon$: Small constant for numerical stability (typically $10^{-12}$)

**Why Layer Norm?** Stabilizes training of deep networks by preventing internal covariate shift.

**4. Residual Connections**

Add input to output of each sublayer:

$$
\mathbf{x}^{(\ell+1)} = \text{LayerNorm}(\mathbf{x}^{(\ell)} + \text{Sublayer}(\mathbf{x}^{(\ell)}))
$$

**Benefit**: Enable gradient flow through deep networks (up to 24 layers in BERT-Large).

**Gradient Backpropagation**: Residual connections create direct paths:

$$
\frac{\partial \mathbf{x}^{(L)}}{\partial \mathbf{x}^{(0)}} = \mathbf{I} + \frac{\partial}{\partial \mathbf{x}^{(0)}} \sum_{\ell=1}^L \text{Sublayer}^{(\ell)}
$$

Identity $\mathbf{I}$ ensures gradient has magnitude at least 1 (prevents vanishing).

**Complete Encoder Layer**:

$$
\begin{aligned}
\mathbf{z}^{(\ell)} &= \text{LayerNorm}(\mathbf{x}^{(\ell-1)} + \text{MultiHead}(\mathbf{x}^{(\ell-1)})) \\
\mathbf{x}^{(\ell)} &= \text{LayerNorm}(\mathbf{z}^{(\ell)} + \text{FFN}(\mathbf{z}^{(\ell)}))
\end{aligned}
$$

**Advantages Over RNNs and CNNs**:

| Aspect | RNN/LSTM | CNN | Transformer |
|--------|----------|-----|-------------|
| **Parallelization** | Sequential (one token at a time) | Parallel within layer | Fully parallel |
| **Training Speed** | Slow ($O(n)$ sequential steps) | Fast | Very fast |
| **Long-Range Dependencies** | Limited (gradient decay) | Limited (receptive field) | Unlimited (direct connections) |
| **Path Length** | $O(n)$ between distant tokens | $O(\log n)$ (stacked layers) | $O(1)$ (direct attention) |
| **Memory** | $O(n)$ | $O(n)$ | $O(n^2)$ (attention matrix) |
| **Receptive Field** | Full sequence | Local then global (stacking) | Full sequence from layer 1 |

**Computational Complexity Analysis**:

For sequence length $n$ and dimension $d$:

**Self-Attention**:
- $\mathbf{Q} \mathbf{K}^\top$: $O(n^2 \cdot d)$ (bottleneck for long sequences)
- Softmax: $O(n^2)$
- Attention $\times$ Values: $O(n^2 \cdot d)$
- **Total**: $O(n^2 \cdot d)$

**Feed-Forward**:
- Two matrix multiplications: $O(n \cdot d \cdot d_{\text{ff}}) = O(n \cdot d^2)$ (since $d_{\text{ff}} = 4d$)

**Trade-off**:
- Short sequences ($n < d$): Self-attention faster
- Long sequences ($n > d$): FFN dominates

For typical transformers: $n=512$, $d=768$ → $n < d$ → attention is bottleneck

**Maximum Sequence Length**: Quadratic memory $O(n^2)$ limits practical length:
- BERT: 512 tokens
- RoBERTa: 512 tokens
- Longformer: 4096 tokens (sparse attention)
- BigBird: 4096 tokens (random/window/global attention)

This concludes Phase 3. Shall I continue with Phase 4 (Pre-trained Language Models) and Phase 5 (LLMs and Parameter Efficiency)?

## Project Structure

The repository is organized as follows:

```plaintext
ag-news-text-classification/
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── ARCHITECTURE.md
├── PERFORMANCE.md
├── SECURITY.md
├── TROUBLESHOOTING.md
├── SOTA_MODELS_GUIDE.md
├── OVERFITTING_PREVENTION.md
├── ROADMAP.md
├── FREE_DEPLOYMENT_GUIDE.md
├── PLATFORM_OPTIMIZATION_GUIDE.md
├── IDE_SETUP_GUIDE.md
├── LOCAL_MONITORING_GUIDE.md
├── QUICK_START.md
├── HEALTH_CHECK.md
├── setup.py
├── setup.cfg
├── MANIFEST.in
├── pyproject.toml
├── poetry.lock
├── Makefile
├── install.sh
├── .env.example
├── .env.test
├── .env.local
├── .gitignore
├── .gitattributes
├── .dockerignore
├── .editorconfig
├── .pre-commit-config.yaml
├── .flake8
├── commitlint.config.js
│
├── requirements/
│   ├── base.txt
│   ├── ml.txt
│   ├── llm.txt
│   ├── efficient.txt
│   ├── local_prod.txt
│   ├── dev.txt
│   ├── data.txt
│   ├── ui.txt
│   ├── docs.txt
│   ├── minimal.txt
│   ├── research.txt
│   ├── robustness.txt
│   ├── all_local.txt
│   ├── colab.txt
│   ├── kaggle.txt
│   ├── free_tier.txt
│   ├── platform_minimal.txt
│   ├── local_monitoring.txt
│   └── lock/
│       ├── base.lock
│       ├── ml.lock
│       ├── llm.lock
│       ├── all.lock
│       └── README.md
│
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
│
├── .husky/
│   ├── pre-commit
│   └── commit-msg
│
├── .ide/
│   ├── SOURCE_OF_TRUTH.yaml
│   │
│   ├── vscode/
│   │   ├── settings.json
│   │   ├── launch.json
│   │   ├── tasks.json
│   │   ├── extensions.json
│   │   └── snippets/
│   │       ├── python.json
│   │       └── yaml.json
│   │
│   ├── pycharm/
│   │   ├── .idea/
│   │   │   ├── workspace.xml
│   │   │   ├── misc.xml
│   │   │   ├── modules.xml
│   │   │   ├── inspectionProfiles/
│   │   │   ├── runConfigurations/
│   │   │   │   ├── train_model.xml
│   │   │   │   ├── run_tests.xml
│   │   │   │   └── start_api.xml
│   │   │   └── codeStyles/
│   │   │       └── Project.xml
│   │   ├── README_PYCHARM.md
│   │   └── settings.zip
│   │
│   ├── jupyter/
│   │   ├── jupyter_notebook_config.py
│   │   ├── jupyter_lab_config.py
│   │   ├── custom/
│   │   │   ├── custom.css
│   │   │   └── custom.js
│   │   ├── nbextensions_config.json
│   │   ├── lab/
│   │   │   ├── user-settings/
│   │   │   └── workspaces/
│   │   └── kernels/
│   │       └── ag-news/
│   │           └── kernel.json
│   │
│   ├── vim/
│   │   ├── .vimrc
│   │   ├── coc-settings.json
│   │   ├── ultisnips/
│   │   │   └── python.snippets
│   │   └── README_VIM.md
│   │
│   ├── neovim/
│   │   ├── init.lua
│   │   ├── lua/
│   │   │   ├── plugins.lua
│   │   │   ├── lsp.lua
│   │   │   ├── keymaps.lua
│   │   │   └── ag-news/
│   │   │       ├── config.lua
│   │   │       └── commands.lua
│   │   ├── coc-settings.json
│   │   └── README_NEOVIM.md
│   │
│   ├── sublime/
│   │   ├── ag-news.sublime-project
│   │   ├── ag-news.sublime-workspace
│   │   ├── Preferences.sublime-settings
│   │   ├── Python.sublime-settings
│   │   ├── snippets/
│   │   │   ├── pytorch-model.sublime-snippet
│   │   │   └── lora-config.sublime-snippet
│   │   ├── build_systems/
│   │   │   ├── Train Model.sublime-build
│   │   │   └── Run Tests.sublime-build
│   │   └── README_SUBLIME.md
│   │
│   └── cloud_ides/
│       ├── gitpod/
│       │   ├── .gitpod.yml
│       │   └── .gitpod.Dockerfile
│       ├── codespaces/
│       │   └── .devcontainer.json
│       ├── colab/
│       │   ├── colab_setup.py
│       │   └── drive_mount.py
│       └── kaggle/
│           └── kaggle_setup.py
│
├── images/
│   ├── pipeline.png
│   ├── api_architecture.png
│   ├── local_deployment_flow.png
│   ├── overfitting_prevention_flow.png
│   ├── sota_model_architecture.png
│   ├── decision_tree.png
│   ├── platform_detection_flow.png
│   ├── auto_training_workflow.png
│   ├── quota_management_diagram.png
│   └── progressive_disclosure.png
│
├── configs/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── config_validator.py
│   ├── config_schema.py
│   ├── constants.py
│   ├── compatibility_matrix.yaml
│   ├── smart_defaults.py
│   │
│   ├── api/
│   │   ├── rest_config.yaml
│   │   ├── auth_config.yaml
│   │   └── rate_limit_config.yaml
│   │
│   ├── services/
│   │   ├── prediction_service.yaml
│   │   ├── training_service.yaml
│   │   ├── data_service.yaml
│   │   ├── model_service.yaml
│   │   └── local_monitoring.yaml
│   │
│   ├── environments/
│   │   ├── dev.yaml
│   │   ├── local_prod.yaml
│   │   ├── colab.yaml
│   │   └── kaggle.yaml
│   │
│   ├── features/
│   │   └── feature_flags.yaml
│   │
│   ├── secrets/
│   │   ├── secrets.template.yaml
│   │   └── local_secrets.yaml
│   │
│   ├── templates/
│   │   ├── README.md
│   │   ├── deberta_template.yaml.j2
│   │   ├── roberta_template.yaml.j2
│   │   ├── llm_template.yaml.j2
│   │   ├── ensemble_template.yaml.j2
│   │   └── training_template.yaml.j2
│   │
│   ├── generation/
│   │   ├── model_specs.yaml
│   │   ├── training_specs.yaml
│   │   └── ensemble_specs.yaml
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── SELECTION_GUIDE.md
│   │   │
│   │   ├── recommended/
│   │   │   ├── README.md
│   │   │   ├── ag_news_best_practices.yaml
│   │   │   ├── quick_start.yaml
│   │   │   ├── balanced.yaml
│   │   │   ├── sota_accuracy.yaml
│   │   │   │
│   │   │   ├── tier_1_sota/
│   │   │   │   ├── deberta_v3_xlarge_lora.yaml
│   │   │   │   ├── deberta_v2_xxlarge_qlora.yaml
│   │   │   │   ├── roberta_large_lora.yaml
│   │   │   │   ├── electra_large_lora.yaml
│   │   │   │   └── xlnet_large_lora.yaml
│   │   │   │
│   │   │   ├── tier_2_llm/
│   │   │   │   ├── llama2_7b_qlora.yaml
│   │   │   │   ├── llama2_13b_qlora.yaml
│   │   │   │   ├── llama3_8b_qlora.yaml
│   │   │   │   ├── mistral_7b_qlora.yaml
│   │   │   │   ├── mixtral_8x7b_qlora.yaml
│   │   │   │   ├── falcon_7b_qlora.yaml
│   │   │   │   ├── phi_3_qlora.yaml
│   │   │   │   └── mpt_7b_qlora.yaml
│   │   │   │
│   │   │   ├── tier_3_ensemble/
│   │   │   │   ├── xlarge_ensemble.yaml
│   │   │   │   ├── llm_ensemble.yaml
│   │   │   │   ├── hybrid_ensemble.yaml
│   │   │   │   └── open_source_llm_ensemble.yaml
│   │   │   │
│   │   │   ├── tier_4_distilled/
│   │   │   │   ├── llama_distilled_deberta.yaml
│   │   │   │   ├── mistral_distilled_roberta.yaml
│   │   │   │   └── ensemble_distilled.yaml
│   │   │   │
│   │   │   └── tier_5_free_optimized/
│   │   │       ├── auto_selected/
│   │   │       │   ├── README.md
│   │   │       │   ├── colab_free_auto.yaml
│   │   │       │   ├── colab_pro_auto.yaml
│   │   │       │   ├── kaggle_auto.yaml
│   │   │       │   ├── local_auto.yaml
│   │   │       │   └── platform_matrix.yaml
│   │   │       │
│   │   │       ├── platform_specific/
│   │   │       │   ├── colab_optimized.yaml
│   │   │       │   ├── kaggle_tpu_optimized.yaml
│   │   │       │   ├── local_cpu_optimized.yaml
│   │   │       │   └── local_gpu_optimized.yaml
│   │   │       │
│   │   │       ├── colab_friendly/
│   │   │       │   ├── deberta_large_lora_colab.yaml
│   │   │       │   ├── distilroberta_efficient.yaml
│   │   │       │   └── ensemble_lightweight.yaml
│   │   │       │
│   │   │       └── cpu_friendly/
│   │   │           ├── distilled_cpu_optimized.yaml
│   │   │           └── quantized_int8.yaml
│   │   │
│   │   ├── single/
│   │   │   ├── transformers/
│   │   │   │   ├── deberta/
│   │   │   │   │   ├── deberta_v3_base.yaml
│   │   │   │   │   ├── deberta_v3_large.yaml
│   │   │   │   │   ├── deberta_v3_xlarge.yaml
│   │   │   │   │   ├── deberta_v2_xlarge.yaml
│   │   │   │   │   ├── deberta_v2_xxlarge.yaml
│   │   │   │   │   └── deberta_sliding_window.yaml
│   │   │   │   │
│   │   │   │   ├── roberta/
│   │   │   │   │   ├── roberta_base.yaml
│   │   │   │   │   ├── roberta_large.yaml
│   │   │   │   │   ├── roberta_large_mnli.yaml
│   │   │   │   │   └── xlm_roberta_large.yaml
│   │   │   │   │
│   │   │   │   ├── electra/
│   │   │   │   │   ├── electra_base.yaml
│   │   │   │   │   ├── electra_large.yaml
│   │   │   │   │   └── electra_discriminator.yaml
│   │   │   │   │
│   │   │   │   ├── xlnet/
│   │   │   │   │   ├── xlnet_base.yaml
│   │   │   │   │   └── xlnet_large.yaml
│   │   │   │   │
│   │   │   │   ├── longformer/
│   │   │   │   │   ├── longformer_base.yaml
│   │   │   │   │   └── longformer_large.yaml
│   │   │   │   │
│   │   │   │   └── t5/
│   │   │   │       ├── t5_base.yaml
│   │   │   │       ├── t5_large.yaml
│   │   │   │       ├── t5_3b.yaml
│   │   │   │       └── flan_t5_xl.yaml
│   │   │   │
│   │   │   └── llm/
│   │   │       ├── llama/
│   │   │       │   ├── llama2_7b.yaml
│   │   │       │   ├── llama2_13b.yaml
│   │   │       │   ├── llama2_70b.yaml
│   │   │       │   ├── llama3_8b.yaml
│   │   │       │   └── llama3_70b.yaml
│   │   │       │
│   │   │       ├── mistral/
│   │   │       │   ├── mistral_7b.yaml
│   │   │       │   ├── mistral_7b_instruct.yaml
│   │   │       │   └── mixtral_8x7b.yaml
│   │   │       │
│   │   │       ├── falcon/
│   │   │       │   ├── falcon_7b.yaml
│   │   │       │   └── falcon_40b.yaml
│   │   │       │
│   │   │       ├── mpt/
│   │   │       │   ├── mpt_7b.yaml
│   │   │       │   └── mpt_30b.yaml
│   │   │       │
│   │   │       └── phi/
│   │   │           ├── phi_2.yaml
│   │   │           └── phi_3.yaml
│   │   │
│   │   └── ensemble/
│   │       ├── ENSEMBLE_SELECTION_GUIDE.yaml
│   │       ├── presets/
│   │       │   ├── quick_start.yaml
│   │       │   ├── sota_accuracy.yaml
│   │       │   └── balanced.yaml
│   │       │
│   │       ├── voting/
│   │       │   ├── soft_voting_xlarge.yaml
│   │       │   ├── weighted_voting_llm.yaml
│   │       │   └── rank_voting_hybrid.yaml
│   │       │
│   │       ├── stacking/
│   │       │   ├── stacking_xlarge_xgboost.yaml
│   │       │   ├── stacking_llm_lightgbm.yaml
│   │       │   └── stacking_hybrid_catboost.yaml
│   │       │
│   │       ├── blending/
│   │       │   ├── blending_xlarge.yaml
│   │       │   └── dynamic_blending_llm.yaml
│   │       │
│   │       └── advanced/
│   │           ├── bayesian_ensemble_xlarge.yaml
│   │           ├── snapshot_ensemble_llm.yaml
│   │           └── multi_level_ensemble.yaml
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   │
│   │   ├── standard/
│   │   │   ├── base_training.yaml
│   │   │   ├── mixed_precision.yaml
│   │   │   └── distributed.yaml
│   │   │
│   │   ├── platform_adaptive/
│   │   │   ├── README.md
│   │   │   ├── colab_free_training.yaml
│   │   │   ├── colab_pro_training.yaml
│   │   │   ├── kaggle_gpu_training.yaml
│   │   │   ├── kaggle_tpu_training.yaml
│   │   │   ├── local_gpu_training.yaml
│   │   │   └── local_cpu_training.yaml
│   │   │
│   │   ├── efficient/
│   │   │   ├── lora/
│   │   │   │   ├── lora_config.yaml
│   │   │   │   ├── lora_xlarge.yaml
│   │   │   │   ├── lora_llm.yaml
│   │   │   │   ├── lora_rank_experiments.yaml
│   │   │   │   └── lora_target_modules_experiments.yaml
│   │   │   │
│   │   │   ├── qlora/
│   │   │   │   ├── qlora_4bit.yaml
│   │   │   │   ├── qlora_8bit.yaml
│   │   │   │   ├── qlora_nf4.yaml
│   │   │   │   └── qlora_llm.yaml
│   │   │   │
│   │   │   ├── adapters/
│   │   │   │   ├── adapter_houlsby.yaml
│   │   │   │   ├── adapter_pfeiffer.yaml
│   │   │   │   ├── adapter_parallel.yaml
│   │   │   │   ├── adapter_fusion.yaml
│   │   │   │   └── adapter_stacking.yaml
│   │   │   │
│   │   │   ├── prefix_tuning/
│   │   │   │   ├── prefix_tuning.yaml
│   │   │   │   ├── prefix_tuning_llm.yaml
│   │   │   │   └── prefix_length_experiments.yaml
│   │   │   │
│   │   │   ├── prompt_tuning/
│   │   │   │   ├── soft_prompt_tuning.yaml
│   │   │   │   ├── p_tuning_v2.yaml
│   │   │   │   └── prompt_length_experiments.yaml
│   │   │   │
│   │   │   ├── ia3/
│   │   │   │   └── ia3_config.yaml
│   │   │   │
│   │   │   └── combined/
│   │   │       ├── lora_plus_adapters.yaml
│   │   │       ├── qlora_plus_prompt.yaml
│   │   │       └── multi_method_fusion.yaml
│   │   │
│   │   ├── tpu/
│   │   │   ├── kaggle_tpu_v3.yaml
│   │   │   └── tpu_optimization.yaml
│   │   │
│   │   ├── advanced/
│   │   │   ├── curriculum_learning.yaml
│   │   │   ├── adversarial_training.yaml
│   │   │   ├── multitask_learning.yaml
│   │   │   ├── contrastive_learning.yaml
│   │   │   ├── knowledge_distillation/
│   │   │   │   ├── standard_distillation.yaml
│   │   │   │   ├── llama_distillation.yaml
│   │   │   │   ├── mistral_distillation.yaml
│   │   │   │   ├── llm_to_xlarge_distillation.yaml
│   │   │   │   ├── xlarge_to_large_distillation.yaml
│   │   │   │   ├── ensemble_distillation.yaml
│   │   │   │   └── self_distillation.yaml
│   │   │   │
│   │   │   ├── meta_learning.yaml
│   │   │   ├── instruction_tuning/
│   │   │   │   ├── alpaca_style.yaml
│   │   │   │   ├── dolly_style.yaml
│   │   │   │   ├── vicuna_style.yaml
│   │   │   │   └── custom_instructions.yaml
│   │   │   │
│   │   │   └── multi_stage/
│   │   │       ├── stage_manager.yaml
│   │   │       ├── progressive_training.yaml
│   │   │       ├── iterative_refinement.yaml
│   │   │       └── base_to_xlarge_progressive.yaml
│   │   │
│   │   ├── regularization/
│   │   │   ├── dropout_strategies/
│   │   │   │   ├── standard_dropout.yaml
│   │   │   │   ├── variational_dropout.yaml
│   │   │   │   ├── dropconnect.yaml
│   │   │   │   ├── adaptive_dropout.yaml
│   │   │   │   ├── monte_carlo_dropout.yaml
│   │   │   │   └── scheduled_dropout.yaml
│   │   │   │
│   │   │   ├── advanced_regularization/
│   │   │   │   ├── r_drop.yaml
│   │   │   │   ├── mixout.yaml
│   │   │   │   ├── spectral_normalization.yaml
│   │   │   │   ├── gradient_penalty.yaml
│   │   │   │   ├── weight_decay_schedule.yaml
│   │   │   │   └── elastic_weight_consolidation.yaml
│   │   │   │
│   │   │   ├── data_regularization/
│   │   │   │   ├── mixup.yaml
│   │   │   │   ├── cutmix.yaml
│   │   │   │   ├── cutout.yaml
│   │   │   │   ├── manifold_mixup.yaml
│   │   │   │   └── augmax.yaml
│   │   │   │
│   │   │   └── combined/
│   │   │       ├── heavy_regularization.yaml
│   │   │       ├── xlarge_safe_config.yaml
│   │   │       └── llm_safe_config.yaml
│   │   │
│   │   └── safe/
│   │       ├── xlarge_safe_training.yaml
│   │       ├── llm_safe_training.yaml
│   │       ├── ensemble_safe_training.yaml
│   │       └── ultra_safe_training.yaml
│   │
│   ├── overfitting_prevention/
│   │   ├── __init__.py
│   │   │
│   │   ├── constraints/
│   │   │   ├── model_size_constraints.yaml
│   │   │   ├── xlarge_constraints.yaml
│   │   │   ├── llm_constraints.yaml
│   │   │   ├── ensemble_constraints.yaml
│   │   │   ├── training_constraints.yaml
│   │   │   └── parameter_efficiency_requirements.yaml
│   │   │
│   │   ├── monitoring/
│   │   │   ├── realtime_monitoring.yaml
│   │   │   ├── thresholds.yaml
│   │   │   ├── metrics_to_track.yaml
│   │   │   └── reporting_schedule.yaml
│   │   │
│   │   ├── validation/
│   │   │   ├── cross_validation_strategy.yaml
│   │   │   ├── holdout_validation.yaml
│   │   │   ├── test_set_protection.yaml
│   │   │   ├── data_split_rules.yaml
│   │   │   └── hyperparameter_tuning_rules.yaml
│   │   │
│   │   ├── recommendations/
│   │   │   ├── dataset_specific/
│   │   │   │   ├── ag_news_recommendations.yaml
│   │   │   │   ├── small_dataset.yaml
│   │   │   │   ├── medium_dataset.yaml
│   │   │   │   └── large_dataset.yaml
│   │   │   │
│   │   │   ├── model_recommendations/
│   │   │   │   ├── xlarge_models.yaml
│   │   │   │   ├── llm_models.yaml
│   │   │   │   └── model_selection_guide.yaml
│   │   │   │
│   │   │   └── technique_recommendations/
│   │   │       ├── lora_recommendations.yaml
│   │   │       ├── qlora_recommendations.yaml
│   │   │       ├── distillation_recommendations.yaml
│   │   │       └── ensemble_recommendations.yaml
│   │   │
│   │   └── safe_defaults/
│   │       ├── xlarge_safe_defaults.yaml
│   │       ├── llm_safe_defaults.yaml
│   │       └── beginner_safe_defaults.yaml
│   │
│   ├── data/
│   │   ├── preprocessing/
│   │   │   ├── standard.yaml
│   │   │   ├── advanced.yaml
│   │   │   ├── llm_preprocessing.yaml
│   │   │   ├── instruction_formatting.yaml
│   │   │   └── domain_specific.yaml
│   │   │
│   │   ├── augmentation/
│   │   │   ├── safe_augmentation.yaml
│   │   │   ├── basic_augmentation.yaml
│   │   │   ├── back_translation.yaml
│   │   │   ├── paraphrase_generation.yaml
│   │   │   ├── llm_augmentation/
│   │   │   │   ├── llama_augmentation.yaml
│   │   │   │   ├── mistral_augmentation.yaml
│   │   │   │   └── controlled_generation.yaml
│   │   │   │
│   │   │   ├── mixup_strategies.yaml
│   │   │   ├── adversarial_augmentation.yaml
│   │   │   └── contrast_sets.yaml
│   │   │
│   │   ├── selection/
│   │   │   ├── coreset_selection.yaml
│   │   │   ├── influence_functions.yaml
│   │   │   └── active_selection.yaml
│   │   │
│   │   ├── validation/
│   │   │   ├── stratified_split.yaml
│   │   │   ├── k_fold_cv.yaml
│   │   │   ├── nested_cv.yaml
│   │   │   ├── time_based_split.yaml
│   │   │   └── holdout_validation.yaml
│   │   │
│   │   └── external/
│   │       ├── news_corpus.yaml
│   │       ├── wikipedia.yaml
│   │       ├── domain_adaptive_pretraining.yaml
│   │       └── synthetic_data/
│   │           ├── llm_generated.yaml
│   │           └── quality_filtering.yaml
│   │
│   ├── deployment/
│   │   ├── local/
│   │   │   ├── docker_local.yaml
│   │   │   ├── api_local.yaml
│   │   │   └── inference_local.yaml
│   │   │
│   │   ├── free_tier/
│   │   │   ├── colab_deployment.yaml
│   │   │   ├── kaggle_deployment.yaml
│   │   │   └── huggingface_spaces.yaml
│   │   │
│   │   └── platform_profiles/
│   │       ├── colab_profile.yaml
│   │       ├── kaggle_profile.yaml
│   │       ├── gitpod_profile.yaml
│   │       ├── codespaces_profile.yaml
│   │       └── hf_spaces_profile.yaml
│   │
│   ├── quotas/
│   │   ├── quota_limits.yaml
│   │   ├── quota_tracking.yaml
│   │   └── platform_quotas.yaml
│   │
│   └── experiments/
│       ├── baselines/
│       │   ├── classical_ml.yaml
│       │   └── transformer_baseline.yaml
│       │
│       ├── ablations/
│       │   ├── model_size_ablation.yaml
│       │   ├── data_amount.yaml
│       │   ├── lora_rank_ablation.yaml
│       │   ├── qlora_bits_ablation.yaml
│       │   ├── regularization_ablation.yaml
│       │   ├── augmentation_impact.yaml
│       │   ├── ensemble_size_ablation.yaml
│       │   ├── ensemble_components.yaml
│       │   ├── prompt_ablation.yaml
│       │   └── distillation_temperature_ablation.yaml
│       │
│       ├── hyperparameter_search/
│       │   ├── lora_search.yaml
│       │   ├── qlora_search.yaml
│       │   ├── regularization_search.yaml
│       │   └── ensemble_weights_search.yaml
│       │
│       ├── sota_experiments/
│       │   ├── phase1_xlarge_models.yaml
│       │   ├── phase2_llm_models.yaml
│       │   ├── phase3_llm_distillation.yaml
│       │   ├── phase4_ensemble_sota.yaml
│       │   ├── phase5_ultimate_sota.yaml
│       │   └── phase6_production_sota.yaml
│       │
│       └── reproducibility/
│           ├── seeds.yaml
│           └── hardware_specs.yaml
│
├── data/
│   ├── raw/
│   │   ├── ag_news/
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   └── README.md
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   ├── stratified_folds/
│   │   ├── instruction_formatted/
│   │   └── .test_set_hash
│   │
│   ├── augmented/
│   │   ├── back_translated/
│   │   ├── paraphrased/
│   │   ├── synthetic/
│   │   ├── llm_generated/
│   │   │   ├── llama2/
│   │   │   ├── mistral/
│   │   │   └── mixtral/
│   │   ├── mixup/
│   │   └── contrast_sets/
│   │
│   ├── external/
│   │   ├── news_corpus/
│   │   ├── pretrain_data/
│   │   └── distillation_data/
│   │       ├── llama_outputs/
│   │       ├── mistral_outputs/
│   │       └── teacher_ensemble_outputs/
│   │
│   ├── pseudo_labeled/
│   ├── selected_subsets/
│   │
│   ├── test_samples/
│   │   ├── api_test_cases.json
│   │   └── mock_responses.json
│   │
│   ├── metadata/
│   │   ├── split_info.json
│   │   ├── statistics.json
│   │   ├── leakage_check.json
│   │   └── model_predictions/
│   │       ├── xlarge_predictions.json
│   │       ├── llm_predictions.json
│   │       └── ensemble_predictions.json
│   │
│   ├── test_access_log.json
│   │
│   ├── platform_cache/
│   │   ├── colab_cache/
│   │   ├── kaggle_cache/
│   │   └── local_cache/
│   │
│   ├── quota_tracking/
│   │   ├── quota_history.json
│   │   ├── session_logs.json
│   │   └── platform_usage.db
│   │
│   └── cache/
│       ├── local_cache/
│       ├── model_cache/
│       └── huggingface_cache/
│
├── src/
│   ├── __init__.py
│   ├── __version__.py
│   ├── cli.py
│   │
│   ├── cli_commands/
│   │   ├── __init__.py
│   │   ├── auto_train.py
│   │   ├── choose_platform.py
│   │   ├── check_quota.py
│   │   └── platform_info.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── factory.py
│   │   ├── types.py
│   │   ├── exceptions.py
│   │   ├── interfaces.py
│   │   │
│   │   ├── health/
│   │   │   ├── __init__.py
│   │   │   ├── health_checker.py
│   │   │   ├── dependency_checker.py
│   │   │   ├── gpu_checker.py
│   │   │   ├── config_checker.py
│   │   │   └── data_checker.py
│   │   │
│   │   ├── auto_fix/
│   │   │   ├── __init__.py
│   │   │   ├── config_fixer.py
│   │   │   ├── dependency_fixer.py
│   │   │   ├── cache_cleaner.py
│   │   │   └── ide_sync_fixer.py
│   │   │
│   │   └── overfitting_prevention/
│   │       ├── __init__.py
│   │       │
│   │       ├── validators/
│   │       │   ├── __init__.py
│   │       │   ├── test_set_validator.py
│   │       │   ├── config_validator.py
│   │       │   ├── data_leakage_detector.py
│   │       │   ├── hyperparameter_validator.py
│   │       │   ├── split_validator.py
│   │       │   ├── model_size_validator.py
│   │       │   ├── lora_config_validator.py
│   │       │   └── ensemble_validator.py
│   │       │
│   │       ├── monitors/
│   │       │   ├── __init__.py
│   │       │   ├── training_monitor.py
│   │       │   ├── overfitting_detector.py
│   │       │   ├── complexity_monitor.py
│   │       │   ├── benchmark_comparator.py
│   │       │   ├── metrics_tracker.py
│   │       │   ├── gradient_monitor.py
│   │       │   └── lora_rank_monitor.py
│   │       │
│   │       ├── constraints/
│   │       │   ├── __init__.py
│   │       │   ├── model_constraints.py
│   │       │   ├── ensemble_constraints.py
│   │       │   ├── augmentation_constraints.py
│   │       │   ├── training_constraints.py
│   │       │   ├── constraint_enforcer.py
│   │       │   └── parameter_efficiency_enforcer.py
│   │       │
│   │       ├── guards/
│   │       │   ├── __init__.py
│   │       │   ├── test_set_guard.py
│   │       │   ├── validation_guard.py
│   │       │   ├── experiment_guard.py
│   │       │   ├── access_control.py
│   │       │   └── parameter_freeze_guard.py
│   │       │
│   │       ├── recommendations/
│   │       │   ├── __init__.py
│   │       │   ├── model_recommender.py
│   │       │   ├── config_recommender.py
│   │       │   ├── prevention_recommender.py
│   │       │   ├── ensemble_recommender.py
│   │       │   ├── lora_recommender.py
│   │       │   ├── distillation_recommender.py
│   │       │   └── parameter_efficiency_recommender.py
│   │       │
│   │       ├── reporting/
│   │       │   ├── __init__.py
│   │       │   ├── overfitting_reporter.py
│   │       │   ├── risk_scorer.py
│   │       │   ├── comparison_reporter.py
│   │       │   ├── html_report_generator.py
│   │       │   └── parameter_efficiency_reporter.py
│   │       │
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── hash_utils.py
│   │           ├── statistical_tests.py
│   │           └── visualization_utils.py
│   │
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── platform_detector.py
│   │   ├── smart_selector.py
│   │   ├── cache_manager.py
│   │   ├── checkpoint_manager.py
│   │   ├── quota_tracker.py
│   │   ├── storage_sync.py
│   │   ├── session_manager.py
│   │   └── resource_monitor.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   │
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── base_handler.py
│   │   │   ├── auth.py
│   │   │   ├── rate_limiter.py
│   │   │   ├── error_handler.py
│   │   │   ├── cors_handler.py
│   │   │   └── request_validator.py
│   │   │
│   │   ├── rest/
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── classification.py
│   │   │   │   ├── training.py
│   │   │   │   ├── models.py
│   │   │   │   ├── data.py
│   │   │   │   ├── health.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── overfitting.py
│   │   │   │   ├── llm.py
│   │   │   │   ├── platform.py
│   │   │   │   └── admin.py
│   │   │   │
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── request_schemas.py
│   │   │   │   ├── response_schemas.py
│   │   │   │   ├── error_schemas.py
│   │   │   │   └── common_schemas.py
│   │   │   │
│   │   │   ├── middleware/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── logging_middleware.py
│   │   │   │   ├── metrics_middleware.py
│   │   │   │   └── security_middleware.py
│   │   │   │
│   │   │   ├── dependencies.py
│   │   │   ├── validators.py
│   │   │   └── websocket_handler.py
│   │   │
│   │   └── local/
│   │       ├── __init__.py
│   │       ├── simple_api.py
│   │       ├── batch_api.py
│   │       └── streaming_api.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── base_service.py
│   │   ├── service_registry.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── prediction_service.py
│   │   │   ├── training_service.py
│   │   │   ├── data_service.py
│   │   │   ├── model_management_service.py
│   │   │   └── llm_service.py
│   │   │
│   │   ├── local/
│   │   │   ├── __init__.py
│   │   │   ├── local_cache_service.py
│   │   │   ├── local_queue_service.py
│   │   │   └── file_storage_service.py
│   │   │
│   │   └── monitoring/
│   │       ├── __init__.py
│   │       ├── monitoring_router.py
│   │       ├── tensorboard_service.py
│   │       ├── mlflow_service.py
│   │       ├── wandb_service.py
│   │       ├── local_metrics_service.py
│   │       └── logging_service.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   │
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── ag_news.py
│   │   │   ├── external_news.py
│   │   │   ├── combined_dataset.py
│   │   │   ├── prompted_dataset.py
│   │   │   ├── instruction_dataset.py
│   │   │   └── distillation_dataset.py
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── text_cleaner.py
│   │   │   ├── tokenization.py
│   │   │   ├── feature_extraction.py
│   │   │   ├── sliding_window.py
│   │   │   ├── prompt_formatter.py
│   │   │   └── instruction_formatter.py
│   │   │
│   │   ├── augmentation/
│   │   │   ├── __init__.py
│   │   │   ├── base_augmenter.py
│   │   │   ├── back_translation.py
│   │   │   ├── paraphrase.py
│   │   │   ├── token_replacement.py
│   │   │   ├── mixup.py
│   │   │   ├── cutmix.py
│   │   │   ├── adversarial.py
│   │   │   ├── contrast_set_generator.py
│   │   │   └── llm_augmenter/
│   │   │       ├── __init__.py
│   │   │       ├── llama_augmenter.py
│   │   │       ├── mistral_augmenter.py
│   │   │       └── controlled_generation.py
│   │   │
│   │   ├── sampling/
│   │   │   ├── __init__.py
│   │   │   ├── balanced_sampler.py
│   │   │   ├── curriculum_sampler.py
│   │   │   ├── active_learning.py
│   │   │   ├── uncertainty_sampling.py
│   │   │   └── coreset_sampler.py
│   │   │
│   │   ├── selection/
│   │   │   ├── __init__.py
│   │   │   ├── influence_function.py
│   │   │   ├── gradient_matching.py
│   │   │   ├── diversity_selection.py
│   │   │   └── quality_filtering.py
│   │   │
│   │   ├── validation/
│   │   │   ├── __init__.py
│   │   │   ├── split_strategies.py
│   │   │   ├── cross_validator.py
│   │   │   ├── nested_cross_validator.py
│   │   │   └── holdout_manager.py
│   │   │
│   │   └── loaders/
│   │       ├── __init__.py
│   │       ├── dataloader.py
│   │       ├── dynamic_batching.py
│   │       └── prefetch_loader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   │
│   │   ├── base/
│   │   │   ├── base_model.py
│   │   │   ├── model_wrapper.py
│   │   │   ├── complexity_tracker.py
│   │   │   └── pooling_strategies.py
│   │   │
│   │   ├── transformers/
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── deberta/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── deberta_v3_base.py
│   │   │   │   ├── deberta_v3_large.py
│   │   │   │   ├── deberta_v3_xlarge.py
│   │   │   │   ├── deberta_v2_xlarge.py
│   │   │   │   ├── deberta_v2_xxlarge.py
│   │   │   │   ├── deberta_sliding_window.py
│   │   │   │   └── deberta_hierarchical.py
│   │   │   │
│   │   │   ├── roberta/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── roberta_base.py
│   │   │   │   ├── roberta_large.py
│   │   │   │   ├── roberta_large_mnli.py
│   │   │   │   ├── roberta_enhanced.py
│   │   │   │   ├── roberta_domain.py
│   │   │   │   └── xlm_roberta_large.py
│   │   │   │
│   │   │   ├── electra/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── electra_base.py
│   │   │   │   ├── electra_large.py
│   │   │   │   └── electra_discriminator.py
│   │   │   │
│   │   │   ├── xlnet/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── xlnet_base.py
│   │   │   │   ├── xlnet_large.py
│   │   │   │   └── xlnet_classifier.py
│   │   │   │
│   │   │   ├── longformer/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── longformer_large.py
│   │   │   │   └── longformer_global.py
│   │   │   │
│   │   │   └── t5/
│   │   │       ├── __init__.py
│   │   │       ├── t5_base.py
│   │   │       ├── t5_large.py
│   │   │       ├── t5_3b.py
│   │   │       ├── flan_t5_xl.py
│   │   │       └── t5_classifier.py
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── llama/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── llama2_7b.py
│   │   │   │   ├── llama2_13b.py
│   │   │   │   ├── llama2_70b.py
│   │   │   │   ├── llama3_8b.py
│   │   │   │   ├── llama3_70b.py
│   │   │   │   └── llama_for_classification.py
│   │   │   │
│   │   │   ├── mistral/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mistral_7b.py
│   │   │   │   ├── mistral_7b_instruct.py
│   │   │   │   ├── mixtral_8x7b.py
│   │   │   │   └── mistral_for_classification.py
│   │   │   │
│   │   │   ├── falcon/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── falcon_7b.py
│   │   │   │   ├── falcon_40b.py
│   │   │   │   └── falcon_for_classification.py
│   │   │   │
│   │   │   ├── mpt/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mpt_7b.py
│   │   │   │   ├── mpt_30b.py
│   │   │   │   └── mpt_for_classification.py
│   │   │   │
│   │   │   └── phi/
│   │   │       ├── __init__.py
│   │   │       ├── phi_2.py
│   │   │       ├── phi_3.py
│   │   │       └── phi_for_classification.py
│   │   │
│   │   ├── prompt_based/
│   │   │   ├── __init__.py
│   │   │   ├── prompt_model.py
│   │   │   ├── soft_prompt.py
│   │   │   ├── instruction_model.py
│   │   │   └── template_manager.py
│   │   │
│   │   ├── efficient/
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── lora/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── lora_model.py
│   │   │   │   ├── lora_config.py
│   │   │   │   ├── lora_layers.py
│   │   │   │   ├── lora_utils.py
│   │   │   │   ├── rank_selection.py
│   │   │   │   └── target_modules_selector.py
│   │   │   │
│   │   │   ├── qlora/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── qlora_model.py
│   │   │   │   ├── qlora_config.py
│   │   │   │   ├── quantization.py
│   │   │   │   └── dequantization.py
│   │   │   │
│   │   │   ├── adapters/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adapter_model.py
│   │   │   │   ├── adapter_config.py
│   │   │   │   ├── houlsby_adapter.py
│   │   │   │   ├── pfeiffer_adapter.py
│   │   │   │   ├── parallel_adapter.py
│   │   │   │   ├── adapter_fusion.py
│   │   │   │   └── adapter_stacking.py
│   │   │   │
│   │   │   ├── prefix_tuning/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── prefix_tuning_model.py
│   │   │   │   ├── prefix_encoder.py
│   │   │   │   └── prefix_length_selector.py
│   │   │   │
│   │   │   ├── prompt_tuning/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── soft_prompt_model.py
│   │   │   │   ├── prompt_encoder.py
│   │   │   │   ├── p_tuning_v2.py
│   │   │   │   └── prompt_initialization.py
│   │   │   │
│   │   │   ├── ia3/
│   │   │   │   ├── __init__.py
│   │   │   │   └── ia3_model.py
│   │   │   │
│   │   │   ├── quantization/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── int8_quantization.py
│   │   │   │   └── dynamic_quantization.py
│   │   │   │
│   │   │   ├── pruning/
│   │   │   │   ├── __init__.py
│   │   │   │   └── magnitude_pruning.py
│   │   │   │
│   │   │   └── combined/
│   │   │       ├── __init__.py
│   │   │       ├── lora_plus_adapter.py
│   │   │       └── multi_method_model.py
│   │   │
│   │   ├── ensemble/
│   │   │   ├── __init__.py
│   │   │   ├── base_ensemble.py
│   │   │   ├── ensemble_selector.py
│   │   │   │
│   │   │   ├── voting/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── soft_voting.py
│   │   │   │   ├── hard_voting.py
│   │   │   │   ├── weighted_voting.py
│   │   │   │   ├── rank_averaging.py
│   │   │   │   └── confidence_weighted_voting.py
│   │   │   │
│   │   │   ├── stacking/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── stacking_classifier.py
│   │   │   │   ├── meta_learners.py
│   │   │   │   ├── cross_validation_stacking.py
│   │   │   │   └── neural_stacking.py
│   │   │   │
│   │   │   ├── blending/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── blending_ensemble.py
│   │   │   │   └── dynamic_blending.py
│   │   │   │
│   │   │   ├── advanced/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bayesian_ensemble.py
│   │   │   │   ├── snapshot_ensemble.py
│   │   │   │   ├── multi_level_ensemble.py
│   │   │   │   └── mixture_of_experts.py
│   │   │   │
│   │   │   └── diversity/
│   │   │       ├── __init__.py
│   │   │       ├── diversity_calculator.py
│   │   │       ├── diversity_optimizer.py
│   │   │       └── ensemble_pruning.py
│   │   │
│   │   └── heads/
│   │       ├── __init__.py
│   │       ├── classification_head.py
│   │       ├── multitask_head.py
│   │       ├── hierarchical_head.py
│   │       ├── attention_head.py
│   │       ├── prompt_head.py
│   │       └── adaptive_head.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   │
│   │   ├── trainers/
│   │   │   ├── __init__.py
│   │   │   ├── base_trainer.py
│   │   │   ├── standard_trainer.py
│   │   │   ├── distributed_trainer.py
│   │   │   ├── apex_trainer.py
│   │   │   ├── safe_trainer.py
│   │   │   ├── auto_trainer.py
│   │   │   ├── lora_trainer.py
│   │   │   ├── qlora_trainer.py
│   │   │   ├── adapter_trainer.py
│   │   │   ├── prompt_trainer.py
│   │   │   ├── instruction_trainer.py
│   │   │   └── multi_stage_trainer.py
│   │   │
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── curriculum/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── curriculum_learning.py
│   │   │   │   ├── self_paced.py
│   │   │   │   └── competence_based.py
│   │   │   │
│   │   │   ├── adversarial/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── fgm.py
│   │   │   │   ├── pgd.py
│   │   │   │   ├── freelb.py
│   │   │   │   └── smart.py
│   │   │   │
│   │   │   ├── regularization/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── r_drop.py
│   │   │   │   ├── mixout.py
│   │   │   │   ├── spectral_norm.py
│   │   │   │   ├── adaptive_dropout.py
│   │   │   │   ├── gradient_penalty.py
│   │   │   │   ├── elastic_weight_consolidation.py
│   │   │   │   └── sharpness_aware_minimization.py
│   │   │   │
│   │   │   ├── distillation/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── knowledge_distillation.py
│   │   │   │   ├── feature_distillation.py
│   │   │   │   ├── self_distillation.py
│   │   │   │   ├── llama_distillation.py
│   │   │   │   ├── mistral_distillation.py
│   │   │   │   ├── ensemble_distillation.py
│   │   │   │   └── progressive_distillation.py
│   │   │   │
│   │   │   ├── meta/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── maml.py
│   │   │   │   └── reptile.py
│   │   │   │
│   │   │   ├── prompt_based/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── prompt_tuning.py
│   │   │   │   ├── prefix_tuning.py
│   │   │   │   ├── p_tuning.py
│   │   │   │   └── soft_prompt_tuning.py
│   │   │   │
│   │   │   ├── tpu_training.py
│   │   │   ├── adaptive_training.py
│   │   │   │
│   │   │   └── multi_stage/
│   │   │       ├── __init__.py
│   │   │       ├── stage_manager.py
│   │   │       ├── progressive_training.py
│   │   │       ├── iterative_refinement.py
│   │   │       └── base_to_xlarge_progression.py
│   │   │
│   │   ├── objectives/
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── losses/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── focal_loss.py
│   │   │   │   ├── label_smoothing.py
│   │   │   │   ├── contrastive_loss.py
│   │   │   │   ├── triplet_loss.py
│   │   │   │   ├── custom_ce_loss.py
│   │   │   │   ├── instruction_loss.py
│   │   │   │   └── distillation_loss.py
│   │   │   │
│   │   │   └── regularizers/
│   │   │       ├── __init__.py
│   │   │       ├── l2_regularizer.py
│   │   │       ├── gradient_penalty.py
│   │   │       ├── complexity_regularizer.py
│   │   │       └── parameter_norm_regularizer.py
│   │   │
│   │   ├── optimization/
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── optimizers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── adamw_custom.py
│   │   │   │   ├── lamb.py
│   │   │   │   ├── lookahead.py
│   │   │   │   ├── sam.py
│   │   │   │   └── adafactor.py
│   │   │   │
│   │   │   ├── schedulers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cosine_warmup.py
│   │   │   │   ├── polynomial_decay.py
│   │   │   │   ├── cyclic_scheduler.py
│   │   │   │   └── inverse_sqrt_scheduler.py
│   │   │   │
│   │   │   └── gradient/
│   │   │       ├── __init__.py
│   │   │       ├── gradient_accumulation.py
│   │   │       ├── gradient_clipping.py
│   │   │       └── gradient_checkpointing.py
│   │   │
│   │   └── callbacks/
│   │       ├── __init__.py
│   │       ├── early_stopping.py
│   │       ├── model_checkpoint.py
│   │       ├── tensorboard_logger.py
│   │       ├── wandb_logger.py
│   │       ├── mlflow_logger.py
│   │       ├── learning_rate_monitor.py
│   │       ├── overfitting_monitor.py
│   │       ├── complexity_regularizer_callback.py
│   │       ├── test_protection_callback.py
│   │       ├── lora_rank_callback.py
│   │       ├── memory_monitor_callback.py
│   │       ├── colab_callback.py
│   │       ├── kaggle_callback.py
│   │       ├── platform_callback.py
│   │       ├── quota_callback.py
│   │       └── session_callback.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   │
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── classification_metrics.py
│   │   │   ├── overfitting_metrics.py
│   │   │   ├── diversity_metrics.py
│   │   │   └── efficiency_metrics.py
│   │   │
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── error_analysis.py
│   │   │   ├── overfitting_analysis.py
│   │   │   ├── train_val_test_comparison.py
│   │   │   ├── lora_rank_analysis.py
│   │   │   └── ensemble_analysis.py
│   │   │
│   │   └── visualizations/
│   │       ├── __init__.py
│   │       ├── training_curves.py
│   │       ├── confusion_matrix.py
│   │       ├── attention_visualization.py
│   │       └── lora_weight_visualization.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   │
│   │   ├── predictors/
│   │   │   ├── __init__.py
│   │   │   ├── single_predictor.py
│   │   │   ├── ensemble_predictor.py
│   │   │   ├── lora_predictor.py
│   │   │   └── qlora_predictor.py
│   │   │
│   │   ├── optimization/
│   │   │   ├── __init__.py
│   │   │   ├── model_quantization.py
│   │   │   ├── model_pruning.py
│   │   │   ├── onnx_export.py
│   │   │   └── openvino_optimization.py
│   │   │
│   │   └── serving/
│   │       ├── __init__.py
│   │       ├── local_server.py
│   │       ├── batch_predictor.py
│   │       └── streaming_predictor.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io_utils.py
│       ├── logging_config.py
│       ├── reproducibility.py
│       ├── distributed_utils.py
│       ├── memory_utils.py
│       ├── profiling_utils.py
│       ├── experiment_tracking.py
│       ├── prompt_utils.py
│       ├── api_utils.py
│       ├── local_utils.py
│       ├── platform_utils.py
│       ├── resource_utils.py
│       └── quota_utils.py
│
├── experiments/
│   ├── __init__.py
│   ├── experiment_runner.py
│   ├── experiment_tagger.py
│   │
│   ├── hyperparameter_search/
│   │   ├── __init__.py
│   │   ├── optuna_search.py
│   │   ├── ray_tune_search.py
│   │   ├── hyperband.py
│   │   ├── bayesian_optimization.py
│   │   ├── lora_rank_search.py
│   │   └── ensemble_weight_search.py
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── speed_benchmark.py
│   │   ├── memory_benchmark.py
│   │   ├── accuracy_benchmark.py
│   │   ├── robustness_benchmark.py
│   │   ├── sota_comparison.py
│   │   ├── overfitting_benchmark.py
│   │   └── parameter_efficiency_benchmark.py
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── svm_baseline.py
│   │   │   ├── random_forest.py
│   │   │   └── logistic_regression.py
│   │   └── neural/
│   │       ├── __init__.py
│   │       ├── lstm_baseline.py
│   │       ├── cnn_baseline.py
│   │       └── bert_vanilla.py
│   │
│   ├── ablation_studies/
│   │   ├── __init__.py
│   │   ├── component_ablation.py
│   │   ├── data_ablation.py
│   │   ├── model_size_ablation.py
│   │   ├── feature_ablation.py
│   │   ├── lora_rank_ablation.py
│   │   ├── qlora_bits_ablation.py
│   │   ├── regularization_ablation.py
│   │   ├── prompt_ablation.py
│   │   └── distillation_temperature_ablation.py
│   │
│   ├── sota_experiments/
│   │   ├── __init__.py
│   │   ├── phase1_xlarge_lora.py
│   │   ├── phase2_llm_qlora.py
│   │   ├── phase3_llm_distillation.py
│   │   ├── phase4_ensemble_xlarge.py
│   │   ├── phase5_ultimate_sota.py
│   │   ├── single_model_sota.py
│   │   ├── ensemble_sota.py
│   │   ├── full_pipeline_sota.py
│   │   ├── production_sota.py
│   │   ├── prompt_based_sota.py
│   │   └── compare_all_approaches.py
│   │
│   └── results/
│       ├── __init__.py
│       ├── experiment_tracker.py
│       ├── result_aggregator.py
│       └── leaderboard_generator.py
│
├── monitoring/
│   ├── README.md
│   ├── local/
│   │   ├── docker-compose.local.yml
│   │   ├── tensorboard_config.yaml
│   │   ├── mlflow_config.yaml
│   │   └── setup_local_monitoring.sh
│   │
│   ├── dashboards/
│   │   ├── tensorboard/
│   │   │   ├── scalar_config.json
│   │   │   ├── image_config.json
│   │   │   └── custom_scalars.json
│   │   │
│   │   ├── mlflow/
│   │   │   ├── experiment_dashboard.py
│   │   │   └── model_registry.py
│   │   │
│   │   ├── wandb/
│   │   │   ├── training_dashboard.json
│   │   │   ├── overfitting_dashboard.json
│   │   │   └── parameter_efficiency_dashboard.json
│   │   │
│   │   ├── platform_dashboard.json
│   │   └── quota_dashboard.json
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── custom_metrics.py
│   │   ├── metric_collectors.py
│   │   ├── local_metrics.py
│   │   ├── model_metrics.py
│   │   ├── training_metrics.py
│   │   ├── overfitting_metrics.py
│   │   ├── platform_metrics.py
│   │   └── quota_metrics.py
│   │
│   ├── logs_analysis/
│   │   ├── __init__.py
│   │   ├── log_parser.py
│   │   ├── anomaly_detector.py
│   │   └── log_aggregator.py
│   │
│   └── scripts/
│       ├── start_tensorboard.sh
│       ├── start_mlflow.sh
│       ├── start_wandb.sh
│       ├── monitor_platform.sh
│       ├── export_metrics.py
│       ├── export_quota_metrics.py
│       └── generate_report.py
│
├── security/
│   ├── local_auth/
│   │   ├── simple_token.py
│   │   └── local_rbac.py
│   ├── data_privacy/
│   │   ├── pii_detector.py
│   │   └── data_masking.py
│   └── model_security/
│       ├── adversarial_defense.py
│       └── model_checksum.py
│
├── plugins/
│   ├── custom_models/
│   │   ├── __init__.py
│   │   └── plugin_interface.py
│   ├── data_sources/
│   │   ├── __init__.py
│   │   └── custom_loaders/
│   ├── evaluators/
│   │   ├── __init__.py
│   │   └── custom_metrics/
│   └── processors/
│       ├── __init__.py
│       └── custom_preprocessors/
│
├── migrations/
│   ├── data/
│   │   ├── 001_initial_schema.py
│   │   └── migration_runner.py
│   ├── models/
│   │   ├── version_converter.py
│   │   └── compatibility_layer.py
│   └── configs/
│       └── config_migrator.py
│
├── cache/
│   ├── local/
│   │   ├── disk_cache.py
│   │   ├── memory_cache.py
│   │   └── lru_cache.py
│   │
│   └── sqlite/
│       └── cache_db_schema.sql
│
├── backup/
│   ├── strategies/
│   │   ├── incremental_backup.yaml
│   │   └── local_backup.yaml
│   ├── scripts/
│   │   ├── backup_local.sh
│   │   └── restore_local.sh
│   └── recovery/
│       └── local_recovery_plan.md
│
├── quickstart/
│   ├── README.md
│   ├── SIMPLE_START.md
│   ├── setup_wizard.py
│   ├── interactive_cli.py
│   ├── decision_tree.py
│   ├── minimal_example.py
│   ├── train_simple.py
│   ├── evaluate_simple.py
│   ├── demo_app.py
│   ├── local_api_quickstart.py
│   ├── auto_start.py
│   ├── auto_train_demo.py
│   ├── colab_notebook.ipynb
│   ├── kaggle_notebook.ipynb
│   │
│   ├── use_cases/
│   │   ├── quick_demo_5min.py
│   │   ├── auto_demo_2min.py
│   │   ├── research_experiment_30min.py
│   │   ├── production_deployment_1hr.py
│   │   ├── learning_exploration.py
│   │   └── platform_comparison_demo.py
│   │
│   └── docker_quickstart/
│       ├── Dockerfile.local
│       └── docker-compose.local.yml
│
├── templates/
│   ├── experiment/
│   │   ├── experiment_template.py
│   │   └── config_template.yaml
│   ├── model/
│   │   ├── model_template.py
│   │   └── README_template.md
│   ├── dataset/
│   │   └── dataset_template.py
│   ├── evaluation/
│   │   └── metric_template.py
│   └── ide/
│       ├── pycharm_run_config.xml
│       ├── vscode_task.json
│       └── jupyter_template.ipynb
│
├── scripts/
│   ├── setup/
│   │   ├── download_all_data.py
│   │   ├── setup_local_environment.sh
│   │   ├── setup_platform.py
│   │   ├── setup_colab.sh
│   │   ├── setup_kaggle.sh
│   │   ├── verify_installation.py
│   │   ├── verify_dependencies.py
│   │   ├── verify_platform.py
│   │   ├── optimize_for_platform.sh
│   │   └── download_pretrained_models.py
│   │
│   ├── data_preparation/
│   │   ├── prepare_ag_news.py
│   │   ├── prepare_external_data.py
│   │   ├── create_augmented_data.py
│   │   ├── create_instruction_data.py
│   │   ├── generate_with_llama.py
│   │   ├── generate_with_mistral.py
│   │   ├── generate_pseudo_labels.py
│   │   ├── create_data_splits.py
│   │   ├── generate_contrast_sets.py
│   │   ├── select_quality_data.py
│   │   ├── verify_data_splits.py
│   │   └── register_test_set.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   │
│   │   ├── single_model/
│   │   │   ├── train_xlarge_lora.py
│   │   │   ├── train_xxlarge_qlora.py
│   │   │   ├── train_llm_qlora.py
│   │   │   └── train_with_adapters.py
│   │   │
│   │   ├── ensemble/
│   │   │   ├── train_xlarge_ensemble.py
│   │   │   ├── train_llm_ensemble.py
│   │   │   └── train_hybrid_ensemble.py
│   │   │
│   │   ├── distillation/
│   │   │   ├── distill_from_llama.py
│   │   │   ├── distill_from_mistral.py
│   │   │   ├── distill_from_ensemble.py
│   │   │   └── progressive_distillation.py
│   │   │
│   │   ├── instruction_tuning/
│   │   │   ├── instruction_tuning_llama.py
│   │   │   └── instruction_tuning_mistral.py
│   │   │
│   │   ├── multi_stage/
│   │   │   ├── base_to_xlarge.py
│   │   │   └── pretrain_finetune_distill.py
│   │   │
│   │   ├── auto_train.sh
│   │   ├── train_all_models.sh
│   │   ├── train_single_model.py
│   │   ├── train_ensemble.py
│   │   ├── train_local.py
│   │   ├── resume_training.py
│   │   └── train_with_prompts.py
│   │
│   ├── domain_adaptation/
│   │   ├── pretrain_on_news.py
│   │   ├── download_news_corpus.py
│   │   └── run_dapt.sh
│   │
│   ├── evaluation/
│   │   ├── evaluate_all_models.py
│   │   ├── evaluate_with_guard.py
│   │   ├── final_evaluation.py
│   │   ├── generate_reports.py
│   │   ├── create_leaderboard.py
│   │   ├── check_overfitting.py
│   │   ├── evaluate_parameter_efficiency.py
│   │   ├── statistical_analysis.py
│   │   └── evaluate_contrast_sets.py
│   │
│   ├── optimization/
│   │   ├── hyperparameter_search.py
│   │   ├── lora_rank_search.py
│   │   ├── ensemble_optimization.py
│   │   ├── quantization_optimization.py
│   │   ├── architecture_search.py
│   │   └── prompt_optimization.py
│   │
│   ├── deployment/
│   │   ├── export_models.py
│   │   ├── optimize_for_inference.py
│   │   ├── create_docker_local.sh
│   │   ├── deploy_to_local.py
│   │   ├── deploy_auto.py
│   │   └── deploy_to_hf_spaces.py
│   │
│   ├── overfitting_prevention/
│   │   ├── get_model_recommendations.py
│   │   ├── validate_experiment_config.py
│   │   ├── check_data_leakage.py
│   │   ├── monitor_training_live.py
│   │   └── generate_overfitting_report.py
│   │
│   ├── platform/
│   │   ├── colab/
│   │   │   ├── mount_drive.py
│   │   │   ├── setup_colab.py
│   │   │   └── keep_alive.py
│   │   │
│   │   ├── kaggle/
│   │   │   ├── setup_kaggle.py
│   │   │   ├── setup_tpu.py
│   │   │   └── create_dataset.py
│   │   │
│   │   └── local/
│   │       ├── detect_gpu.py
│   │       └── optimize_local.py
│   │
│   ├── monitoring/
│   │   ├── monitor_quota.py
│   │   └── monitor_session.py
│   │
│   ├── ide/
│   │   ├── setup_pycharm.py
│   │   ├── setup_vscode.py
│   │   ├── setup_jupyter.py
│   │   ├── setup_vim.py
│   │   └── setup_all_ides.sh
│   │
│   ├── local/
│   │   ├── start_local_api.sh
│   │   ├── start_monitoring.sh
│   │   ├── cleanup_cache.sh
│   │   └── backup_experiments.sh
│   │
│   └── ci/
│       ├── run_tests.sh
│       ├── run_benchmarks.sh
│       ├── build_docker_local.sh
│       ├── test_local_deployment.sh
│       ├── check_docs_sync.py
│       └── verify_all.sh
│
├── prompts/
│   ├── classification/
│   │   ├── zero_shot.txt
│   │   ├── few_shot.txt
│   │   └── chain_of_thought.txt
│   ├── instruction/
│   │   ├── base_instruction.txt
│   │   ├── detailed_instruction.txt
│   │   └── task_specific.txt
│   └── distillation/
│       ├── llm_prompts.txt
│       └── explanation_prompts.txt
│
├── notebooks/
│   ├── README.md
│   │
│   ├── 00_setup/
│   │   ├── 00_auto_setup.ipynb
│   │   ├── 00_local_setup.ipynb
│   │   ├── 01_colab_setup.ipynb
│   │   ├── 02_kaggle_setup.ipynb
│   │   ├── 03_vscode_setup.ipynb
│   │   ├── 04_pycharm_setup.ipynb
│   │   └── 05_jupyterlab_setup.ipynb
│   │
│   ├── 01_tutorials/
│   │   ├── 00_auto_training_tutorial.ipynb
│   │   ├── 00_environment_setup.ipynb
│   │   ├── 01_data_loading_basics.ipynb
│   │   ├── 02_preprocessing_tutorial.ipynb
│   │   ├── 03_model_training_basics.ipynb
│   │   ├── 04_lora_tutorial.ipynb
│   │   ├── 05_qlora_tutorial.ipynb
│   │   ├── 06_distillation_tutorial.ipynb
│   │   ├── 07_ensemble_tutorial.ipynb
│   │   ├── 08_overfitting_prevention.ipynb
│   │   ├── 09_safe_training_workflow.ipynb
│   │   ├── 10_evaluation_tutorial.ipynb
│   │   ├── 11_prompt_engineering.ipynb
│   │   ├── 12_instruction_tuning.ipynb
│   │   ├── 13_local_api_usage.ipynb
│   │   ├── 14_monitoring_setup.ipynb
│   │   ├── 15_platform_optimization.ipynb
│   │   └── 16_quota_management.ipynb
│   │
│   ├── 02_exploratory/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_model_size_analysis.ipynb
│   │   ├── 03_parameter_efficiency_analysis.ipynb
│   │   ├── 04_data_statistics.ipynb
│   │   ├── 05_label_distribution.ipynb
│   │   ├── 06_text_length_analysis.ipynb
│   │   ├── 07_vocabulary_analysis.ipynb
│   │   └── 08_contrast_set_exploration.ipynb
│   │
│   ├── 03_experiments/
│   │   ├── 01_baseline_experiments.ipynb
│   │   ├── 02_xlarge_lora_experiments.ipynb
│   │   ├── 03_llm_qlora_experiments.ipynb
│   │   ├── 04_ensemble_experiments.ipynb
│   │   ├── 05_distillation_experiments.ipynb
│   │   ├── 06_sota_experiments.ipynb
│   │   ├── 07_ablation_studies.ipynb
│   │   ├── 08_sota_reproduction.ipynb
│   │   ├── 09_prompt_experiments.ipynb
│   │   └── 10_single_model_experiments.ipynb
│   │
│   ├── 04_analysis/
│   │   ├── 01_error_analysis.ipynb
│   │   ├── 02_overfitting_analysis.ipynb
│   │   ├── 03_lora_rank_analysis.ipynb
│   │   ├── 04_ensemble_diversity_analysis.ipynb
│   │   ├── 05_parameter_efficiency_comparison.ipynb
│   │   ├── 06_model_interpretability.ipynb
│   │   ├── 07_attention_visualization.ipynb
│   │   ├── 08_embedding_analysis.ipynb
│   │   └── 09_failure_cases.ipynb
│   │
│   ├── 05_deployment/
│   │   ├── 01_model_export.ipynb
│   │   ├── 02_quantization.ipynb
│   │   ├── 03_local_serving.ipynb
│   │   ├── 04_model_optimization.ipynb
│   │   ├── 05_inference_pipeline.ipynb
│   │   ├── 06_api_demo.ipynb
│   │   └── 07_hf_spaces_deploy.ipynb
│   │
│   └── 06_platform_specific/
│       ├── local/
│       │   ├── auto_training_local.ipynb
│       │   ├── cpu_training.ipynb
│       │   ├── gpu_training.ipynb
│       │   ├── multi_gpu_local.ipynb
│       │   └── inference_demo.ipynb
│       │
│       ├── colab/
│       │   ├── auto_training_colab.ipynb
│       │   ├── quick_start_colab.ipynb
│       │   ├── full_training_colab.ipynb
│       │   ├── drive_optimization.ipynb
│       │   ├── keep_alive_demo.ipynb
│       │   └── inference_demo_colab.ipynb
│       │
│       ├── kaggle/
│       │   ├── auto_training_kaggle.ipynb
│       │   ├── kaggle_submission.ipynb
│       │   ├── kaggle_training.ipynb
│       │   ├── tpu_training.ipynb
│       │   └── dataset_caching.ipynb
│       │
│       └── huggingface/
│           └── spaces_demo.ipynb
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── gradio_app.py
│   │
│   ├── pages/
│   │   ├── 01_Home.py
│   │   ├── 02_Single_Prediction.py
│   │   ├── 03_Batch_Analysis.py
│   │   ├── 04_Model_Comparison.py
│   │   ├── 05_Overfitting_Dashboard.py
│   │   ├── 06_Model_Recommender.py
│   │   ├── 07_Parameter_Efficiency_Dashboard.py
│   │   ├── 08_Interpretability.py
│   │   ├── 09_Performance_Dashboard.py
│   │   ├── 10_Real_Time_Demo.py
│   │   ├── 11_Model_Selection.py
│   │   ├── 12_Documentation.py
│   │   ├── 13_Prompt_Testing.py
│   │   ├── 14_Local_Monitoring.py
│   │   ├── 15_IDE_Setup_Guide.py
│   │   ├── 16_Experiment_Tracker.py
│   │   ├── 17_Platform_Info.py
│   │   ├── 18_Quota_Dashboard.py
│   │   ├── 19_Platform_Selector.py
│   │   └── 20_Auto_Train_UI.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── prediction_component.py
│   │   ├── overfitting_monitor.py
│   │   ├── lora_config_selector.py
│   │   ├── ensemble_builder.py
│   │   ├── visualization_component.py
│   │   ├── model_selector.py
│   │   ├── file_uploader.py
│   │   ├── result_display.py
│   │   ├── performance_monitor.py
│   │   ├── prompt_builder.py
│   │   ├── ide_configurator.py
│   │   ├── platform_info_component.py
│   │   ├── quota_monitor_component.py
│   │   └── resource_gauge.py
│   │
│   ├── utils/
│   │   ├── session_manager.py
│   │   ├── caching.py
│   │   ├── theming.py
│   │   └── helpers.py
│   │
│   └── assets/
│       ├── css/
│       │   └── custom.css
│       ├── js/
│       │   └── custom.js
│       └── images/
│           ├── logo.png
│           └── banner.png
│
├── outputs/
│   ├── models/
│   │   ├── checkpoints/
│   │   ├── pretrained/
│   │   ├── fine_tuned/
│   │   ├── lora_adapters/
│   │   ├── qlora_adapters/
│   │   ├── ensembles/
│   │   ├── distilled/
│   │   ├── optimized/
│   │   ├── exported/
│   │   └── prompted/
│   │
│   ├── results/
│   │   ├── experiments/
│   │   ├── benchmarks/
│   │   ├── overfitting_reports/
│   │   ├── parameter_efficiency_reports/
│   │   ├── ablations/
│   │   └── reports/
│   │
│   ├── analysis/
│   │   ├── error_analysis/
│   │   ├── interpretability/
│   │   └── statistical/
│   │
│   ├── logs/
│   │   ├── training/
│   │   ├── tensorboard/
│   │   ├── mlflow/
│   │   ├── wandb/
│   │   └── local/
│   │
│   ├── profiling/
│   │   ├── memory/
│   │   ├── speed/
│   │   └── traces/
│   │
│   └── artifacts/
│       ├── figures/
│       ├── tables/
│       ├── lora_visualizations/
│       └── presentations/
│
├── docs/
│   ├── index.md
│   ├── 00_START_HERE.md
│   ├── limitations.md
│   ├── ethical_considerations.md
│   │
│   ├── getting_started/
│   │   ├── installation.md
│   │   ├── local_setup.md
│   │   ├── ide_setup.md
│   │   ├── quickstart.md
│   │   ├── auto_mode.md
│   │   ├── platform_detection.md
│   │   ├── overfitting_prevention_quickstart.md
│   │   ├── choosing_model.md
│   │   ├── choosing_platform.md
│   │   ├── free_deployment.md
│   │   └── troubleshooting.md
│   │
│   ├── level_1_beginner/
│   │   ├── README.md
│   │   ├── 01_installation.md
│   │   ├── 02_first_model.md
│   │   ├── 03_evaluation.md
│   │   ├── 04_deployment.md
│   │   └── quick_demo.md
│   │
│   ├── level_2_intermediate/
│   │   ├── README.md
│   │   ├── 01_lora_qlora.md
│   │   ├── 02_ensemble.md
│   │   ├── 03_distillation.md
│   │   └── 04_optimization.md
│   │
│   ├── level_3_advanced/
│   │   ├── README.md
│   │   ├── 01_sota_pipeline.md
│   │   ├── 02_custom_models.md
│   │   └── 03_research_workflow.md
│   │
│   ├── platform_guides/
│   │   ├── README.md
│   │   ├── colab_guide.md
│   │   ├── colab_advanced.md
│   │   ├── kaggle_guide.md
│   │   ├── kaggle_tpu.md
│   │   ├── local_guide.md
│   │   ├── gitpod_guide.md
│   │   └── platform_comparison.md
│   │
│   ├── user_guide/
│   │   ├── data_preparation.md
│   │   ├── model_training.md
│   │   ├── auto_training.md
│   │   ├── lora_guide.md
│   │   ├── qlora_guide.md
│   │   ├── distillation_guide.md
│   │   ├── ensemble_guide.md
│   │   ├── overfitting_prevention.md
│   │   ├── safe_training_practices.md
│   │   ├── evaluation.md
│   │   ├── local_deployment.md
│   │   ├── quota_management.md
│   │   ├── platform_optimization.md
│   │   ├── prompt_engineering.md
│   │   └── advanced_techniques.md
│   │
│   ├── developer_guide/
│   │   ├── architecture.md
│   │   ├── adding_models.md
│   │   ├── custom_datasets.md
│   │   ├── local_api_development.md
│   │   └── contributing.md
│   │
│   ├── api_reference/
│   │   ├── rest_api.md
│   │   ├── data_api.md
│   │   ├── models_api.md
│   │   ├── training_api.md
│   │   ├── lora_api.md
│   │   ├── ensemble_api.md
│   │   ├── overfitting_prevention_api.md
│   │   ├── platform_api.md
│   │   ├── quota_api.md
│   │   └── evaluation_api.md
│   │
│   ├── ide_guides/
│   │   ├── vscode_guide.md
│   │   ├── pycharm_guide.md
│   │   ├── jupyter_guide.md
│   │   ├── vim_guide.md
│   │   ├── sublime_guide.md
│   │   └── comparison.md
│   │
│   ├── tutorials/
│   │   ├── basic_usage.md
│   │   ├── xlarge_model_tutorial.md
│   │   ├── llm_tutorial.md
│   │   ├── distillation_tutorial.md
│   │   ├── sota_pipeline_tutorial.md
│   │   ├── local_training_tutorial.md
│   │   ├── free_deployment_tutorial.md
│   │   └── best_practices.md
│   │
│   ├── best_practices/
│   │   ├── model_selection.md
│   │   ├── parameter_efficient_finetuning.md
│   │   ├── avoiding_overfitting.md
│   │   ├── local_optimization.md
│   │   └── ensemble_building.md
│   │
│   ├── examples/
│   │   ├── 00_hello_world.md
│   │   ├── 01_train_baseline.md
│   │   ├── 02_sota_pipeline.md
│   │   └── 03_custom_model.md
│   │
│   ├── cheatsheets/
│   │   ├── model_selection_cheatsheet.pdf
│   │   ├── overfitting_prevention_checklist.pdf
│   │   ├── free_deployment_comparison.pdf
│   │   ├── platform_comparison_chart.pdf
│   │   ├── auto_train_cheatsheet.pdf
│   │   ├── quota_limits_reference.pdf
│   │   └── cli_commands.pdf
│   │
│   ├── troubleshooting/
│   │   ├── platform_issues.md
│   │   └── quota_issues.md
│   │
│   ├── architecture/
│   │   ├── decisions/
│   │   │   ├── 001-model-selection.md
│   │   │   ├── 002-ensemble-strategy.md
│   │   │   ├── 003-local-first-design.md
│   │   │   ├── 004-overfitting-prevention.md
│   │   │   └── 005-parameter-efficiency.md
│   │   ├── diagrams/
│   │   │   ├── system-overview.puml
│   │   │   ├── data-flow.puml
│   │   │   ├── local-deployment.puml
│   │   │   └── overfitting-prevention-flow.puml
│   │   └── patterns/
│   │       ├── factory-pattern.md
│   │       └── strategy-pattern.md
│   │
│   ├── operations/
│   │   ├── runbooks/
│   │   │   ├── local_deployment.md
│   │   │   └── troubleshooting.md
│   │   └── sops/
│   │       ├── model-update.md
│   │       └── data-refresh.md
│   │
│   └── _static/
│       └── custom.css
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.local
│   │   ├── Dockerfile.gpu.local
│   │   ├── docker-compose.local.yml
│   │   └── .dockerignore
│   │
│   ├── auto_deploy/
│   │   ├── auto_deploy.py
│   │   ├── platform_deploy.sh
│   │   └── README.md
│   │
│   ├── platform_specific/
│   │   ├── colab_deploy.md
│   │   ├── kaggle_deploy.md
│   │   └── local_deploy.md
│   │
│   ├── huggingface/
│   │   ├── spaces_config.yaml
│   │   ├── requirements.txt
│   │   ├── app.py
│   │   └── README.md
│   │
│   ├── streamlit_cloud/
│   │   ├── .streamlit/
│   │   │   └── config.toml
│   │   └── requirements.txt
│   │
│   └── local/
│       ├── systemd/
│       │   ├── ag-news-api.service
│       │   └── ag-news-monitor.service
│       ├── nginx/
│       │   └── ag-news.conf
│       └── scripts/
│           ├── start_all.sh
│           └── stop_all.sh
│
├── benchmarks/
│   ├── accuracy/
│   │   ├── model_comparison.json
│   │   ├── xlarge_models.json
│   │   ├── llm_models.json
│   │   ├── ensemble_results.json
│   │   └── sota_benchmarks.json
│   │
│   ├── efficiency/
│   │   ├── parameter_efficiency.json
│   │   ├── memory_usage.json
│   │   ├── training_time.json
│   │   ├── inference_speed.json
│   │   └── platform_comparison.json
│   │
│   ├── robustness/
│   │   ├── adversarial_results.json
│   │   ├── ood_detection.json
│   │   └── contrast_set_results.json
│   │
│   └── overfitting/
│       ├── train_val_gaps.json
│       ├── lora_ranks.json
│       └── prevention_effectiveness.json
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   │
│   ├── unit/
│   │   ├── data/
│   │   │   ├── test_preprocessing.py
│   │   │   ├── test_augmentation.py
│   │   │   ├── test_dataloader.py
│   │   │   └── test_contrast_sets.py
│   │   │
│   │   ├── models/
│   │   │   ├── test_transformers.py
│   │   │   ├── test_ensemble.py
│   │   │   ├── test_efficient.py
│   │   │   └── test_prompt_models.py
│   │   │
│   │   ├── training/
│   │   │   ├── test_trainers.py
│   │   │   ├── test_auto_trainer.py
│   │   │   ├── test_strategies.py
│   │   │   ├── test_callbacks.py
│   │   │   └── test_multi_stage.py
│   │   │
│   │   ├── deployment/
│   │   │   ├── test_platform_detector.py
│   │   │   ├── test_smart_selector.py
│   │   │   ├── test_cache_manager.py
│   │   │   ├── test_checkpoint_manager.py
│   │   │   └── test_quota_tracker.py
│   │   │
│   │   ├── api/
│   │   │   ├── test_rest_api.py
│   │   │   ├── test_local_api.py
│   │   │   └── test_auth.py
│   │   │
│   │   ├── overfitting_prevention/
│   │   │   ├── test_validators.py
│   │   │   ├── test_monitors.py
│   │   │   ├── test_constraints.py
│   │   │   ├── test_guards.py
│   │   │   └── test_recommenders.py
│   │   │
│   │   └── utils/
│   │       ├── test_memory_utils.py
│   │       └── test_utilities.py
│   │
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_auto_train_flow.py
│   │   ├── test_ensemble_pipeline.py
│   │   ├── test_inference_pipeline.py
│   │   ├── test_local_api_flow.py
│   │   ├── test_prompt_pipeline.py
│   │   ├── test_llm_integration.py
│   │   ├── test_platform_workflows.py
│   │   ├── test_quota_tracking_flow.py
│   │   └── test_overfitting_prevention_flow.py
│   │
│   ├── platform_specific/
│   │   ├── test_colab_integration.py
│   │   ├── test_kaggle_integration.py
│   │   └── test_local_integration.py
│   │
│   ├── performance/
│   │   ├── test_model_speed.py
│   │   ├── test_memory_usage.py
│   │   ├── test_accuracy_benchmarks.py
│   │   ├── test_local_performance.py
│   │   ├── test_sla_compliance.py
│   │   └── test_throughput.py
│   │
│   ├── e2e/
│   │   ├── test_complete_workflow.py
│   │   ├── test_user_scenarios.py
│   │   ├── test_local_deployment.py
│   │   ├── test_free_deployment.py
│   │   ├── test_quickstart_pipeline.py
│   │   ├── test_sota_pipeline.py
│   │   ├── test_auto_train_colab.py
│   │   ├── test_auto_train_kaggle.py
│   │   └── test_quota_enforcement.py
│   │
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── test_model_accuracy.py
│   │   ├── test_ensemble_diversity.py
│   │   ├── test_inference_speed.py
│   │   └── baseline_results.json
│   │
│   ├── chaos/
│   │   ├── __init__.py
│   │   ├── test_fault_tolerance.py
│   │   ├── test_corrupted_config.py
│   │   ├── test_oom_handling.py
│   │   └── test_network_failures.py
│   │
│   ├── compatibility/
│   │   ├── __init__.py
│   │   ├── test_torch_versions.py
│   │   ├── test_transformers_versions.py
│   │   └── test_cross_platform.py
│   │
│   └── fixtures/
│       ├── sample_data.py
│       ├── mock_models.py
│       ├── test_configs.py
│       └── local_fixtures.py
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── tests.yml
│   │   ├── documentation.yml
│   │   ├── benchmarks.yml
│   │   ├── overfitting_checks.yml
│   │   ├── docs_sync_check.yml
│   │   ├── local_deployment_test.yml
│   │   ├── dependency_updates.yml
│   │   ├── compatibility_matrix.yml
│   │   ├── regression_tests.yml
│   │   ├── test_platform_detection.yml
│   │   ├── test_auto_train.yml
│   │   └── platform_compatibility.yml
│   │
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   ├── ide_support_request.md
│   │   └── overfitting_report.md
│   │
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
│
└── tools/
    │
    ├── profiling/
    │   ├── memory_profiler.py
    │   ├── speed_profiler.py
    │   ├── parameter_counter.py
    │   └── local_profiler.py
    │
    ├── debugging/
    │   ├── model_debugger.py
    │   ├── overfitting_debugger.py
    │   ├── lora_debugger.py
    │   ├── data_validator.py
    │   ├── platform_debugger.py
    │   ├── quota_debugger.py
    │   └── local_debugger.py
    │
    ├── visualization/
    │   ├── training_monitor.py
    │   ├── lora_weight_plotter.py
    │   ├── ensemble_diversity_plotter.py
    │   └── result_plotter.py
    │
    ├── config_tools/
    │   ├── __init__.py
    │   ├── config_generator.py
    │   ├── config_explainer.py
    │   ├── config_comparator.py
    │   ├── config_optimizer.py
    │   ├── sync_manager.py
    │   ├── auto_sync.sh
    │   └── validate_all_configs.py
    │
    ├── platform_tools/
    │   ├── __init__.py
    │   ├── detector_tester.py
    │   ├── quota_simulator.py
    │   └── platform_benchmark.py
    │
    ├── cost_tools/
    │   ├── cost_estimator.py
    │   └── cost_comparator.py
    │
    ├── ide_tools/
    │   ├── pycharm_config_generator.py
    │   ├── vscode_tasks_generator.py
    │   ├── jupyter_kernel_setup.py
    │   ├── vim_plugin_installer.sh
    │   ├── universal_ide_generator.py
    │   └── sync_ide_configs.py
    │
    ├── compatibility/
    │   ├── __init__.py
    │   ├── compatibility_checker.py
    │   ├── version_matrix_tester.py
    │   └── upgrade_path_finder.py
    │
    ├── automation/
    │   ├── __init__.py
    │   ├── health_check_runner.py
    │   ├── auto_fix_runner.py
    │   ├── batch_config_generator.py
    │   ├── platform_health.py
    │   └── nightly_tasks.sh
    │
    └── cli_helpers/
        ├── __init__.py
        ├── rich_console.py
        ├── progress_bars.py
        ├── interactive_prompts.py
        └── ascii_art.py
```

## Usage
