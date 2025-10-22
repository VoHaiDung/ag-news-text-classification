# AG News Text Classification

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

# Introduction

## 1.1 Research Motivation and Context

Text classification constitutes one of the fundamental tasks in natural language processing, serving as a cornerstone for applications ranging from sentiment analysis and spam detection to news categorization and intent recognition. The task is formally defined as learning a mapping function from a document space to a discrete set of predefined categories, where the objective is to minimize prediction error on unseen examples drawn from the same underlying distribution as the training data.

Over the past decade, the field has witnessed a paradigm shift from feature-engineering approaches based on bag-of-words representations and classical machine learning algorithms, to end-to-end neural architectures that learn hierarchical representations directly from raw text. The introduction of attention mechanisms and pre-trained transformer models has further revolutionized the landscape, enabling competitive accuracy on benchmark datasets through transfer learning from massive unsupervised corpora.

Despite these advances, a fundamental tension persists between the capacity of modern neural architectures and the size of available labeled datasets. Contemporary state-of-the-art models such as BERT, RoBERTa, DeBERTa, and large language models like LLaMA and Mistral contain parameters numbering in the hundreds of millions to tens of billions, while supervised classification datasets typically provide only thousands to hundreds of thousands of labeled examples. This disparity creates a severe risk of overfitting, where models achieve near-perfect accuracy on training data yet fail to generalize to held-out test sets.

This project addresses the challenge of developing text classification systems that achieve competitive accuracy while maintaining rigorous generalization guarantees. We focus specifically on the AG News dataset as an experimental testbed—not because it represents the frontier of difficulty in modern NLP, but precisely because its moderate size and balanced structure provide an ideal controlled environment for studying the interplay between model capacity, training methodology, and generalization performance. The complete dataset characteristics and experimental protocols are detailed in [Dataset](#dataset).

## 1.2 The Generalization Challenge in Modern NLP

The core theoretical challenge in supervised machine learning is the minimization of expected risk, defined over an unknown data distribution. Let us denote the input space of text documents as $\mathcal{X}$ and the output space of class labels as $\mathcal{Y}$. Given a training dataset $\mathcal{D}$ consisting of $N$ independently and identically distributed samples:

$$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\}$$

where each pair $(x_i, y_i)$ is drawn from an unknown joint distribution $P$ over $\mathcal{X} \times \mathcal{Y}$, the learning objective is to find a function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that minimizes the expected risk:

$$R(f) = \mathbb{E}_{(x,y) \sim P}[\ell(f(x), y)]$$

Here, $\ell$ denotes a loss function quantifying the penalty for incorrect predictions. For classification tasks, this is commonly the zero-one loss, which equals one when the prediction differs from the true label and zero otherwise.

Since the true distribution $P$ is unknown and inaccessible, practical learning algorithms instead minimize the empirical risk computed on the observed training data:

$$R_{\text{emp}}(f) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i), y_i)$$

The fundamental question is whether a function $f$ that achieves low empirical risk will also achieve low expected risk on new, unseen examples. The difference between these two quantities is termed the **generalization gap**:

$$\Delta(f) = R(f) - R_{\text{emp}}(f)$$

This generalization gap represents the core challenge addressed throughout this framework. When $\Delta(f)$ is large, the model has overfit to the training data and will perform poorly on new examples despite excellent training accuracy.

Statistical learning theory provides bounds on this generalization gap. According to the Vapnik-Chervonenkis theory, with probability at least $1 - \delta$, the true risk is bounded by:

$$R(f) \leq R_{\text{emp}}(f) + \sqrt{\frac{d \log(2N/d) + \log(4/\delta)}{N}}$$

where $d$ represents the VC dimension, a measure of the model's capacity or expressiveness. 

**Interpreting this bound**: The inequality reveals a fundamental trade-off. The first term $R_{\text{emp}}(f)$ decreases as we use more expressive models with higher capacity (larger $d$), since such models can fit the training data better. However, the second term—the generalization gap—increases with model capacity $d$ and decreases with dataset size $N$. Models with high capacity can achieve low empirical risk but may suffer from a large generalization gap. Conversely, models with limited capacity have tighter generalization bounds but may be unable to capture the true underlying patterns, leading to high bias. The optimal model complexity minimizes the sum of both terms.

In the context of modern transformer-based language models, the effective capacity is enormous. For instance, DeBERTa-v3-XLarge contains approximately 710 million parameters, while the AG News training set provides 120,000 labeled examples. This yields a parameter-to-sample ratio of approximately 5,917:1, meaning each parameter is informed by fewer than 0.0002 training samples on average. Classical statistical learning theory would suggest that such models should catastrophically overfit, memorizing training examples without learning generalizable patterns.

However, empirical practice demonstrates that careful application of several techniques can mitigate this overfitting risk:

- **Transfer Learning**: Pre-training on large unsupervised corpora allows models to learn general linguistic representations before fine-tuning on task-specific data. This effectively reduces the number of parameters that must be learned from the supervised dataset alone, as discussed in [Section 1.6.2](#162-transfer-learning-and-pre-training).

- **Parameter-Efficient Fine-Tuning**: Methods such as Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) constrain the fine-tuning process to modify only a small subset of model parameters, dramatically reducing effective capacity. Theoretical foundations are provided in [Section 1.6.3](#163-parameter-efficient-fine-tuning-theory).

- **Regularization**: Techniques including dropout, weight decay, and early stopping impose explicit or implicit penalties on model complexity, encouraging simpler solutions that generalize better. Our regularization strategies are detailed in [configs/training/regularization/](./configs/training/regularization/).

- **Ensemble Methods**: Combining predictions from multiple diverse models reduces variance and improves robustness, even when individual models may overfit. The theoretical basis is explained in [Section 1.6.4](#164-ensemble-methods-and-diversity).

- **Knowledge Distillation**: Training smaller student models to mimic the behavior of larger teacher models or ensembles preserves much of the performance gain while reducing inference cost and overfitting risk, as detailed in [Section 1.6.5](#165-knowledge-distillation).

The challenge addressed by this project is to systematically combine these techniques within a unified framework that treats overfitting prevention not as an afterthought, but as a primary architectural consideration from the outset. The complete overfitting prevention architecture is documented in [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md).

## 1.3 Bridging Theory and Practice

A significant gap exists in the current landscape between theoretical understanding of generalization and practical implementation of text classification systems. Research papers often present novel architectures or training techniques with impressive benchmark results, yet the code repositories accompanying these papers frequently lack the infrastructure necessary to ensure that these results are reliable, reproducible, and generalizable.

Common issues include:

- **Inadequate Validation Protocols**: Using a single train-test split without proper validation set management, leading to potential overfitting to the validation set through repeated hyperparameter tuning.

- **Test Set Contamination**: Inadvertent exposure of test set information during development, such as through exploratory data analysis, error analysis on test samples, or repeated evaluation during model selection.

- **Unreported Hyperparameter Sensitivity**: Publishing results from a single random seed or configuration without acknowledging variance across runs, making results difficult to reproduce.

- **Platform-Specific Assumptions**: Code that assumes access to specific computational infrastructure (multi-GPU clusters, high-memory machines, fast storage) that is not available to many researchers and practitioners.

- **Lack of Systematic Overfitting Detection**: Relying on manual inspection of learning curves rather than automated monitoring systems that can detect subtle signs of overfitting early in training.

This project seeks to bridge this gap by providing not merely a collection of high-performing models, but a complete experimental framework that embeds best practices for generalization into every component of the system. The overfitting prevention mechanisms described in detail in [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md) are not separate modules to be optionally enabled, but rather integral parts of the training pipeline that operate by default.

Furthermore, we recognize that reproducibility requires more than fixing random seeds. It requires comprehensive logging of the complete experimental environment, including hardware specifications, software versions, data preprocessing steps, and the full hyperparameter configuration. It requires statistical rigor in comparing models, using multiple random seeds and appropriate significance testing. And it requires transparent reporting of all results, including negative results and failed experiments, to provide an honest assessment of what approaches work and under what conditions.

Our implementation of these principles includes:

- **Automated Environment Logging**: Capture of hardware specifications, CUDA versions, library versions, and platform characteristics. See [src/utils/experiment_tracking.py](./src/utils/experiment_tracking.py).

- **Multi-Seed Evaluation**: All reported results average over at least 3-5 random seeds with standard deviations. Configuration templates in [configs/experiments/reproducibility/](./configs/experiments/reproducibility/).

- **Statistical Significance Testing**: Paired t-tests with Bonferroni correction for comparing multiple models. Implementation in [src/evaluation/metrics/](./src/evaluation/metrics/).

- **Comprehensive Experiment Tracking**: Integration with TensorBoard, MLflow, and Weights & Biases for complete audit trails. Setup guides in [LOCAL_MONITORING_GUIDE.md](./LOCAL_MONITORING_GUIDE.md).

## 1.4 Research Gaps and Motivations

Through extensive review of existing text classification implementations and research codebases, we have identified several specific gaps that motivated the design decisions in this project.

### 1.4.1 Post-Hoc Versus Preventive Overfitting Management

**Current Practice**: The typical workflow in developing text classification systems treats overfitting as a problem to be diagnosed after it occurs. Researchers train models, observe divergence between training and validation metrics, and then apply corrective measures such as increasing regularization strength, reducing model capacity, or implementing early stopping.

**Limitation**: This reactive approach has several drawbacks. First, it wastes computational resources by allowing training to proceed even when the configuration is almost certain to overfit. Second, it creates opportunities for inadvertent test set leakage, as researchers may be tempted to check test performance to gauge whether their overfitting mitigation strategies are working. Third, it lacks systematic principles for determining appropriate hyperparameter values, leading to ad-hoc choices that may not generalize across different datasets or model architectures.

**Our Approach**: We implement a preventive overfitting management system that operates at multiple stages:

1. **Pre-Training Validation**: Before any GPU computation begins, the system analyzes the proposed configuration against a set of constraint rules based on dataset size, model capacity, and historical performance patterns. Configurations likely to result in severe overfitting are rejected with explanatory messages and recommended alternatives. Implementation in [src/core/overfitting_prevention/validators/](./src/core/overfitting_prevention/validators/).

2. **Real-Time Monitoring**: During training, multiple metrics are tracked to detect early warning signs of overfitting, including the train-validation gap in both loss and accuracy, the rate of change in these metrics, gradient magnitudes, and parameter norms. When predefined thresholds are exceeded, training can be automatically halted with diagnostic information. See [src/core/overfitting_prevention/monitors/](./src/core/overfitting_prevention/monitors/).

3. **Post-Training Analysis**: After training completes, comprehensive overfitting risk scores are computed based on multiple factors, and detailed reports are generated comparing the observed behavior to expected patterns for the given configuration. Report generation in [src/core/overfitting_prevention/reporting/](./src/core/overfitting_prevention/reporting/).

The theoretical foundations and implementation details of this system are described in [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md), while this document focuses on the high-level rationale and integration with the overall experimental workflow.

### 1.4.2 Fragmented State-of-the-Art Techniques

**Current Practice**: Achieving competitive results on text classification benchmarks increasingly requires combining multiple advanced techniques: large pre-trained transformers, parameter-efficient fine-tuning methods, ensemble approaches, and knowledge distillation. However, these techniques are often developed and published in separate research efforts, with implementations residing in different codebases using incompatible interfaces.

**Limitation**: A researcher wishing to reproduce state-of-the-art results must manually integrate code from multiple sources, resolve dependency conflicts, adapt to different configuration formats, and navigate undocumented assumptions about data formats and training procedures. This integration burden is particularly challenging for researchers who are domain experts but not deep learning engineers.

**Our Approach**: We provide a unified model zoo spanning classical baselines to large language models, with consistent interfaces and composable configurations. The system is organized into tiers that represent different points in the accuracy-efficiency trade-off space:

- **Tier 1 - Classical Baselines**: Traditional machine learning approaches including Naive Bayes, Support Vector Machines, and Logistic Regression, serving as sanity checks and computational efficiency baselines. Implementations in [experiments/baselines/classical/](./experiments/baselines/classical/).

- **Tier 2 - Standard Transformers**: Full fine-tuning of models like BERT-Base and RoBERTa-Base, establishing the performance ceiling for conventional approaches. Configurations in [configs/models/single/transformers/](./configs/models/single/transformers/).

- **Tier 3 - Large Transformers with PEFT**: Models such as DeBERTa-v3-XLarge and DeBERTa-v2-XXLarge fine-tuned using Low-Rank Adaptation, demonstrating parameter-efficient scaling to larger architectures. Configurations in [configs/models/recommended/tier_1_sota/](./configs/models/recommended/tier_1_sota/).

- **Tier 4 - Large Language Models**: Instruction-tuned models like LLaMA 2 and Mistral fine-tuned with Quantized LoRA, representing the current frontier in transfer learning. Configurations in [configs/models/recommended/tier_2_llm/](./configs/models/recommended/tier_2_llm/).

- **Tier 5 - Ensembles and Distillation**: Multi-model ensembles achieving maximum accuracy, and distilled student models that compress ensemble knowledge into efficient single-model architectures. Configurations in [configs/models/recommended/tier_3_ensemble/](./configs/models/recommended/tier_3_ensemble/) and [configs/models/recommended/tier_4_distilled/](./configs/models/recommended/tier_4_distilled/).

Detailed guidance on selecting appropriate models for different use cases is provided in [SOTA_MODELS_GUIDE.md](./SOTA_MODELS_GUIDE.md), while the system architecture enabling this modularity is documented in [ARCHITECTURE.md](./ARCHITECTURE.md).

### 1.4.3 Platform Dependence and Accessibility Barriers

**Current Practice**: Much academic research in NLP assumes access to substantial computational resources, such as multi-GPU clusters with high-bandwidth interconnects, large amounts of RAM, and fast persistent storage. Code is often optimized for specific environments like institutional SLURM clusters or cloud platforms with particular instance types.

**Limitation**: This creates significant barriers for independent researchers, students, and practitioners in resource-constrained settings. Even when pre-trained models are publicly available, fine-tuning them on custom datasets may be infeasible without access to expensive infrastructure. Furthermore, differences in hardware and software configurations across platforms frequently lead to reproducibility failures, where code that runs successfully in one environment produces errors or different results in another.

**Our Approach**: We adopt a platform-agnostic design that automatically adapts to available computational resources:

1. **Automatic Platform Detection**: The system identifies the execution environment at runtime (Google Colab, Kaggle Notebooks, local CPU, local GPU, etc.) and loads appropriate configuration defaults. Implementation in [src/deployment/platform_detector.py](./src/deployment/platform_detector.py).

2. **Resource-Aware Model Selection**: Based on detected GPU memory, CPU count, and time quotas, the system recommends models and training configurations that will complete within available resources. Decision logic in [src/deployment/smart_selector.py](./src/deployment/smart_selector.py).

3. **Checkpoint Resilience**: Training state is periodically synchronized to persistent storage (Google Drive for Colab, Kaggle Datasets for Kaggle) so that sessions interrupted due to platform time limits can be seamlessly resumed. Implementation in [src/deployment/checkpoint_manager.py](./src/deployment/checkpoint_manager.py).

4. **Quota Management**: For platforms with usage limits (Colab's 12-hour sessions, Kaggle's weekly GPU quotas), the system tracks consumption and optimizes checkpoint intervals to maximize effective training time. Quota tracking in [src/deployment/quota_tracker.py](./src/deployment/quota_tracker.py).

The platform adaptation mechanisms are detailed in [PLATFORM_OPTIMIZATION_GUIDE.md](./PLATFORM_OPTIMIZATION_GUIDE.md), while platform-specific configurations are available in [configs/environments/](./configs/environments/) and [configs/training/platform_adaptive/](./configs/training/platform_adaptive/).

### 1.4.4 Test Set Integrity and Experimental Validity

**Current Practice**: In many machine learning projects, the test set is treated as merely another data split, stored in the same directory structure as training and validation data and accessible to all code components. Researchers manually ensure that test data is not used during development, relying on discipline rather than technical safeguards.

**Limitation**: Human error and the iterative nature of machine learning experimentation make test set leakage a persistent risk. Subtle forms of leakage can occur through:

- **Direct leakage**: Accidentally including test samples in training batches due to indexing errors or incorrect data split logic.

- **Indirect leakage**: Making modeling decisions (feature selection, architecture choices, hyperparameter ranges) based on test set characteristics observed during exploratory analysis.

- **Adaptive leakage**: Trying multiple models or configurations and selecting based on test performance, effectively overfitting to the test set through the model selection process itself.

- **Preprocessing leakage**: Computing normalization statistics, vocabulary mappings, or other preprocessing parameters on the full dataset including test samples, then applying these to create test features.

**Our Approach**: We implement cryptographic test set protection mechanisms:

1. **Hash-Based Integrity**: Upon first loading the test set, a SHA-256 hash is computed over the sorted sample identifiers and stored in a protected file at [data/processed/.test_set_hash](./data/processed/.test_set_hash). Every subsequent test evaluation verifies that the hash matches, detecting any modification or corruption of the test set. Implementation in [src/core/overfitting_prevention/utils/hash_utils.py](./src/core/overfitting_prevention/utils/hash_utils.py).

2. **Access Logging**: All code paths that access test data are instrumented to log the timestamp, calling function, purpose statement, and stack trace to [data/test_access_log.json](./data/test_access_log.json). This creates an auditable record of test set usage. Implementation in [src/core/overfitting_prevention/guards/test_set_guard.py](./src/core/overfitting_prevention/guards/test_set_guard.py).

3. **Access Control**: The test set loader requires explicit authorization through configuration flags. By default, attempting to load test data during training or hyperparameter tuning raises an exception. Guard implementation in [src/core/overfitting_prevention/guards/access_control.py](./src/core/overfitting_prevention/guards/access_control.py).

4. **Budget Enforcement**: A configurable limit on the number of test evaluations prevents adaptive overfitting through repeated model selection based on test performance. Configuration in [configs/overfitting_prevention/validation/test_set_protection.yaml](./configs/overfitting_prevention/validation/test_set_protection.yaml).

These mechanisms, described in detail in [OVERFITTING_PREVENTION.md § Test Set Protection](./OVERFITTING_PREVENTION.md), provide technical enforcement of best practices that would otherwise rely solely on researcher discipline.

## 1.5 Core Contributions

This project makes several specific contributions to the practice of text classification:

### 1. Comprehensive Overfitting Prevention Framework

We provide a multi-layered system for preventing, detecting, and diagnosing overfitting that operates throughout the experimental lifecycle. This includes pre-training configuration validation, real-time monitoring during training, post-training risk assessment, and test set protection mechanisms. The system is designed to be both technically rigorous—implementing ideas from statistical learning theory and adaptive data analysis—and practically usable, with clear error messages and actionable recommendations.

Key components:

- Pre-training validators in [src/core/overfitting_prevention/validators/](./src/core/overfitting_prevention/validators/)
- Real-time monitors in [src/core/overfitting_prevention/monitors/](./src/core/overfitting_prevention/monitors/)
- Constraint enforcers in [src/core/overfitting_prevention/constraints/](./src/core/overfitting_prevention/constraints/)
- Test set guards in [src/core/overfitting_prevention/guards/](./src/core/overfitting_prevention/guards/)
- Automated recommenders in [src/core/overfitting_prevention/recommendations/](./src/core/overfitting_prevention/recommendations/)

Complete documentation in [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md).

### 2. Unified Parameter-Efficient Fine-Tuning Infrastructure

We integrate multiple PEFT methods (LoRA, QLoRA, Adapters, Prefix Tuning, Prompt Tuning) within a consistent interface, enabling fair comparisons and hybrid approaches. Extensive hyperparameter search results for LoRA rank selection, target module choices, and regularization strategies are provided to guide users toward configurations that balance accuracy and efficiency for the AG News dataset.

Implementations:

- LoRA: [src/models/efficient/lora/](./src/models/efficient/lora/)
- QLoRA: [src/models/efficient/qlora/](./src/models/efficient/qlora/)
- Adapters: [src/models/efficient/adapters/](./src/models/efficient/adapters/)
- Prefix Tuning: [src/models/efficient/prefix_tuning/](./src/models/efficient/prefix_tuning/)
- Prompt Tuning: [src/models/efficient/prompt_tuning/](./src/models/efficient/prompt_tuning/)

Configuration templates in [configs/training/efficient/](./configs/training/efficient/) and ablation studies in [experiments/ablation_studies/](./experiments/ablation_studies/).

### 3. Multi-Stage SOTA Pipeline

We demonstrate a systematic progression from individual high-capacity models through ensemble aggregation to knowledge distillation, achieving competitive accuracy while maintaining interpretability and deployability. Each stage is fully reproducible with provided configurations and scripts, and ablation studies isolate the contribution of each component.

Pipeline stages:

- Phase 1: XLarge model training with LoRA - [experiments/sota_experiments/phase1_xlarge_lora.py](./experiments/sota_experiments/phase1_xlarge_lora.py)
- Phase 2: LLM fine-tuning with QLoRA - [experiments/sota_experiments/phase2_llm_qlora.py](./experiments/sota_experiments/phase2_llm_qlora.py)
- Phase 3: Knowledge distillation - [experiments/sota_experiments/phase3_llm_distillation.py](./experiments/sota_experiments/phase3_llm_distillation.py)
- Phase 4: Ensemble aggregation - [experiments/sota_experiments/phase4_ensemble_xlarge.py](./experiments/sota_experiments/phase4_ensemble_xlarge.py)
- Phase 5: Ultimate SOTA - [experiments/sota_experiments/phase5_ultimate_sota.py](./experiments/sota_experiments/phase5_ultimate_sota.py)

Complete pipeline guide in [SOTA_MODELS_GUIDE.md](./SOTA_MODELS_GUIDE.md).

### 4. Platform-Agnostic Reproducibility

Through automatic platform detection, resource-aware configuration selection, and comprehensive environment logging, we ensure that experiments can be reproduced across diverse computational environments. This includes consumer laptops, cloud platforms (Google Colab, Kaggle Notebooks), and institutional compute clusters.

Platform support:

- Detection system: [src/deployment/platform_detector.py](./src/deployment/platform_detector.py)
- Smart configuration selection: [src/deployment/smart_selector.py](./src/deployment/smart_selector.py)
- Platform-specific configs: [configs/environments/](./configs/environments/)
- Colab optimization: [configs/training/platform_adaptive/colab_free_training.yaml](./configs/training/platform_adaptive/colab_free_training.yaml)
- Kaggle optimization: [configs/training/platform_adaptive/kaggle_gpu_training.yaml](./configs/training/platform_adaptive/kaggle_gpu_training.yaml)

Platform guide in [PLATFORM_OPTIMIZATION_GUIDE.md](./PLATFORM_OPTIMIZATION_GUIDE.md).

### 5. Extensive Documentation and Educational Resources

Beyond this technical README, we provide detailed guides targeted at different user expertise levels:

- **Beginners**: Step-by-step tutorials covering basic concepts and common workflows in [docs/level_1_beginner/](./docs/level_1_beginner/)
- **Practitioners**: Best practice guides for model selection, hyperparameter tuning, and deployment in [docs/best_practices/](./docs/best_practices/)
- **Researchers**: In-depth theoretical treatments of overfitting prevention, ensemble methods, and distillation in [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md) and [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Developers**: API documentation and architecture descriptions for extending the codebase in [docs/developer_guide/](./docs/developer_guide/) and [docs/api_reference/](./docs/api_reference/)

This progressive disclosure approach allows users to engage with the system at a level appropriate to their background and goals. Complete navigation guide in [Section 1.8](#18-organization-of-documentation).

## 1.6 Theoretical Foundations: Overview

This section provides a high-level overview of the theoretical principles underlying our implementation. Detailed mathematical treatments, proofs, and empirical validations are provided in specialized documentation referenced below.

### 1.6.1 Statistical Learning Theory and Capacity Control

The fundamental question in supervised learning is how well a model trained on finite data will perform on new examples. Statistical learning theory, particularly the Vapnik-Chervonenkis framework, provides rigorous bounds on generalization error as a function of model capacity and sample size.

For a hypothesis space with VC dimension $d$, the generalization bound states that with probability at least $1 - \delta$, the true risk $R(f)$ of any hypothesis $f$ is bounded by:

$$R(f) \leq R_{\text{emp}}(f) + \sqrt{\frac{d \log(2N/d) + \log(4/\delta)}{N}}$$

where:

- $R_{\text{emp}}(f)$ is the empirical risk (average loss on training data)
- $N$ is the number of training samples
- $d$ is the VC dimension (a measure of model complexity)
- $\delta$ is the confidence parameter

**Interpretation**: This bound consists of two terms. The first term, empirical risk, can be made arbitrarily small by using sufficiently complex models that fit the training data well. However, the second term—the generalization gap—grows with model capacity ($d$) and shrinks with dataset size ($N$). The optimal model complexity minimizes the sum of these two terms.

More intuitively, the bound tells us that as we increase model complexity $d$, we can fit the training data better (reducing $R_{\text{emp}}(f)$), but the gap between training and true performance grows (the second term increases). Conversely, with more training data $N$, we can safely use more complex models since the generalization gap shrinks proportionally to $1/\sqrt{N}$.

**Practical Implications for This Project**:

1. **Parameter-Efficient Fine-Tuning**: Methods like LoRA reduce the effective VC dimension by constraining the fine-tuning process to low-rank subspaces. For a transformer with $d_{\text{model}}$ hidden dimensions and rank $r \ll d_{\text{model}}$, LoRA updates have far fewer degrees of freedom than full fine-tuning, leading to tighter generalization bounds. Implementation in [src/models/efficient/lora/](./src/models/efficient/lora/).

2. **Early Stopping**: Rather than training until convergence on the training set, we halt training when validation performance plateaus. This implicitly limits effective capacity by restricting the number of gradient updates. Callback implementation in [src/training/callbacks/early_stopping.py](./src/training/callbacks/early_stopping.py).

3. **Ensemble Methods**: While individual models may have high capacity, ensemble averaging acts as a form of regularization. The variance reduction achieved through ensembling can be formalized through bias-variance decomposition as shown in [Section 1.6.4](#164-ensemble-methods-and-diversity).

4. **Validation-Based Model Selection**: By selecting models based on validation rather than test performance, we avoid the adaptive overfitting problem identified by Dwork et al. (2015) in their work on preserving validity in adaptive data analysis. Our validation protocols are detailed in [configs/overfitting_prevention/validation/](./configs/overfitting_prevention/validation/).

Complete theoretical treatments including Rademacher complexity analysis, PAC learning bounds, and empirical process theory perspectives are provided in [OVERFITTING_PREVENTION.md § Theoretical Framework](./OVERFITTING_PREVENTION.md).

### 1.6.2 Transfer Learning and Pre-Training

The remarkable success of transformer-based models on text classification despite limited labeled data is largely attributable to transfer learning through unsupervised pre-training. The pre-training phase learns general linguistic representations from massive unlabeled corpora, while fine-tuning specializes these representations to task-specific patterns.

**Mathematical Formulation**: Let $\theta$ denote the model parameters. Pre-training solves:

$$\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{pretrain}}(\theta; \mathcal{D}_{\text{unlabeled}})$$

where $\mathcal{L}_{\text{pretrain}}$ is an unsupervised objective such as masked language modeling. For masked language modeling, given a sequence of tokens $\mathbf{x} = (x_1, x_2, \ldots, x_T)$, we randomly mask a subset of positions $\mathcal{M}$ and train the model to predict the masked tokens:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\backslash \mathcal{M}}; \theta)$$

where $\mathbf{x}_{\backslash \mathcal{M}}$ denotes the sequence with masked positions replaced by a special [MASK] token.

Fine-tuning then solves:

$$\theta_{\text{task}} = \arg\min_{\theta} \mathcal{L}_{\text{task}}(\theta; \mathcal{D}_{\text{labeled}})$$

initialized at $\theta^*$ rather than random initialization. For classification, the task loss is typically cross-entropy:

$$\mathcal{L}_{\text{task}} = -\sum_{(x,y) \in \mathcal{D}} \log P(y | x; \theta)$$

**Why This Works**: Pre-training on diverse text learns broadly useful features—syntactic patterns, semantic relationships, world knowledge—that transfer across tasks. Fine-tuning requires learning only task-specific decision boundaries, which can be accomplished with far less labeled data than learning both representations and decision boundaries from scratch.

The effectiveness can be understood through a two-stage perspective:
1. In the pre-training stage, the model learns to encode general linguistic knowledge into its parameters $\theta^*$
2. In the fine-tuning stage, only the final classification layer (and optionally, small adjustments to the encoder) needs to be learned from the limited labeled data

**Evidence**: Empirical studies (Devlin et al., 2019; Liu et al., 2019; He et al., 2021) demonstrate that pre-trained transformers achieve competitive performance with hundreds to thousands of labeled examples, whereas comparable architectures trained from scratch require orders of magnitude more data.

**Application in This Project**: All Tier 2 and higher models leverage publicly available pre-trained checkpoints from Hugging Face Hub. We additionally explore domain-adaptive pre-training on news corpora to further specialize representations to the AG News domain. 

Pre-trained model configurations:
- Standard transformers: [configs/models/single/transformers/](./configs/models/single/transformers/)
- Large language models: [configs/models/single/llm/](./configs/models/single/llm/)
- Domain adaptation scripts: [scripts/domain_adaptation/](./scripts/domain_adaptation/)

The transfer learning pipeline is detailed in [ARCHITECTURE.md § Transfer Learning](./ARCHITECTURE.md).

### 1.6.3 Parameter-Efficient Fine-Tuning Theory

Full fine-tuning of large pre-trained models is parameter-inefficient: it requires storing and updating hundreds of millions to billions of parameters, most of which change only slightly from their pre-trained values. Parameter-efficient fine-tuning (PEFT) methods address this by constraining updates to low-dimensional subspaces or small adapter modules.

**Low-Rank Adaptation (LoRA)**: The key insight is that the weight updates during fine-tuning often have low intrinsic dimensionality. For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA represents updates as a low-rank decomposition:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$ is the rank.

**Detailed Explanation**: 
- $W_0$ is the original pre-trained weight matrix, which remains frozen during training
- $\Delta W = BA$ represents the update to the weights, factorized as a product of two smaller matrices
- Matrix $B$ has dimensions $d \times r$ and matrix $A$ has dimensions $r \times k$
- The rank $r$ controls the capacity of the adaptation; smaller $r$ means fewer trainable parameters and stronger regularization

During forward pass, instead of using $W_0 x$, we compute:
$$y = W_0 x + BAx = W_0 x + \Delta W x$$

**Parameter Count**: The original matrix has $d \times k$ parameters. LoRA introduces only $r(d + k)$ trainable parameters while freezing $W_0$. For typical values ($d = k = 768$, $r = 8$), this represents a reduction from 589,824 to 12,288 trainable parameters—a 48× reduction.

For example, in a standard transformer attention layer:
- Original: Query, Key, Value, Output matrices each have $768 \times 768 = 589,824$ parameters
- With LoRA (rank 8): Each adaptation has only $8 \times (768 + 768) = 12,288$ trainable parameters
- Total reduction: From ~2.4M parameters per attention layer to ~50K trainable parameters

**Theoretical Justification**: Li et al. (2018) demonstrated that the optimization landscape of neural networks lies approximately on low-dimensional manifolds. More recently, Aghajanyan et al. (2021) showed empirically that fine-tuning is intrinsically low-dimensional, with effective rank often much smaller than the explicit parameter count. This means that even though the full parameter space is high-dimensional, the optimization trajectory primarily moves in a low-dimensional subspace.

**Quantized LoRA (QLoRA)**: Building on LoRA, QLoRA applies 4-bit quantization to the frozen base model weights, reducing memory requirements by approximately 4× while maintaining fine-tuning quality through additional techniques:

1. **4-bit NormalFloat Quantization**: Uses a special data type optimized for normally distributed weights
2. **Double Quantization**: Quantizes the quantization constants themselves to save additional memory
3. **Paged Optimizers**: Uses NVIDIA unified memory to handle memory spikes during training

The quantization function maps full-precision weights $W_0$ to 4-bit representation:
$$W_0^{\text{quant}} = \text{Quantize}(W_0, \text{dtype}=\text{NF4})$$

During fine-tuning, only the LoRA adapters $B$ and $A$ are kept in full precision.

**Application in This Project**: We provide extensive configurations for LoRA (ranks 4, 8, 16, 32, 64) and QLoRA (4-bit, 8-bit) across multiple target modules (query, key, value, output projections). 

Configuration files:
- LoRA configs: [configs/training/efficient/lora/](./configs/training/efficient/lora/)
- QLoRA configs: [configs/training/efficient/qlora/](./configs/training/efficient/qlora/)
- Rank ablation: [experiments/ablation_studies/lora_rank_ablation.py](./experiments/ablation_studies/lora_rank_ablation.py)
- Target module ablation: [configs/training/efficient/lora/lora_target_modules_experiments.yaml](./configs/training/efficient/lora/lora_target_modules_experiments.yaml)

Theoretical analysis is provided in [OVERFITTING_PREVENTION.md § Parameter Efficiency](./OVERFITTING_PREVENTION.md).

### 1.6.4 Ensemble Methods and Diversity

Ensemble methods combine predictions from multiple models to achieve better performance than any individual model. The effectiveness of ensembling is well-understood through bias-variance decomposition.

**Bias-Variance-Covariance Decomposition**: For squared loss in regression, the expected error of an ensemble can be decomposed as:

$$E_{\text{ensemble}} = \overline{\text{Bias}^2} + \frac{1}{M}\overline{\text{Var}} + \frac{M-1}{M}\overline{\text{Cov}}$$

where:
- $M$ is the number of models in the ensemble
- $\overline{\text{Bias}^2}$ is the average squared bias of individual models
- $\overline{\text{Var}}$ is the average variance of individual models
- $\overline{\text{Cov}}$ is the average covariance between model predictions

**Detailed Interpretation**: 

This decomposition reveals three key insights:

1. **Bias Term ($\overline{\text{Bias}^2}$)**: Represents systematic errors that all models in the ensemble share. Ensembling does not reduce bias—if all models make the same systematic mistake, averaging them preserves that mistake. The bias term remains constant regardless of ensemble size.

2. **Variance Term ($\frac{1}{M}\overline{\text{Var}}$)**: Represents random errors due to finite training data. This term is reduced by a factor of $M$ through averaging. With $M=5$ models, variance is reduced to 20% of the single-model variance. This is why ensembles are particularly effective when individual models have high variance (overfitting).

3. **Covariance Term ($\frac{M-1}{M}\overline{\text{Cov}}$)**: Represents correlation between model errors. If models make identical errors, $\overline{\text{Cov}} = \overline{\text{Var}}$, and the variance reduction benefit is completely negated. Maximum variance reduction occurs when models are diverse—making errors on different examples, so $\overline{\text{Cov}} \approx 0$.

For classification, while the exact decomposition differs, the same principles apply: ensemble performance improves when individual models are both accurate (low bias) and diverse (low error correlation).

**Diversity Promotion Strategies**: We employ several approaches to create diverse ensemble members:

1. **Different Architectures**: DeBERTa, RoBERTa, ELECTRA, XLNet have different inductive biases:
   - DeBERTa uses disentangled attention mechanisms
   - RoBERTa uses dynamic masking during pre-training
   - ELECTRA uses discriminative pre-training objectives
   - XLNet uses permutation language modeling
   
   Configurations in [configs/models/ensemble/](./configs/models/ensemble/)

2. **Different Initializations**: Multiple random seeds create different optimization trajectories, even with the same architecture. Configuration templates in [configs/experiments/reproducibility/seeds.yaml](./configs/experiments/reproducibility/seeds.yaml).

3. **Different Data Views**: Varied data augmentation strategies (back-translation, paraphrasing, LLM-based generation) expose models to different perturbations of the training data. Augmentation configs in [configs/data/augmentation/](./configs/data/augmentation/).

4. **Different Training Procedures**: Varied learning rates, dropout rates, and regularization strengths create models with different bias-variance trade-offs. Configurations in [configs/training/regularization/](./configs/training/regularization/).

**Aggregation Methods**: We compare multiple ensemble aggregation strategies:

1. **Soft Voting**: Average predicted probabilities across models before taking argmax:
   $$\hat{y} = \arg\max_c \frac{1}{M} \sum_{m=1}^{M} P_m(y=c|x)$$
   
   Implementation in [src/models/ensemble/voting/soft_voting.py](./src/models/ensemble/voting/soft_voting.py)

2. **Weighted Voting**: Learn optimal weights $w_m$ on validation set:
   $$\hat{y} = \arg\max_c \sum_{m=1}^{M} w_m P_m(y=c|x)$$
   
   where $\sum_m w_m = 1$ and $w_m \geq 0$. Weights can be learned using linear regression or more complex meta-learners. Implementation in [src/models/ensemble/voting/weighted_voting.py](./src/models/ensemble/voting/weighted_voting.py).

3. **Stacking**: Train a meta-classifier (logistic regression, gradient boosting) on concatenated predictions from base models:
   $$\hat{y} = f_{\text{meta}}([P_1(y|x), P_2(y|x), \ldots, P_M(y|x)])$$
   
   Implementation in [src/models/ensemble/stacking/](./src/models/ensemble/stacking/)

Implementation details and empirical comparisons are provided in [src/models/ensemble/](./src/models/ensemble/) with selection guidance in [SOTA_MODELS_GUIDE.md § Ensemble Selection](./SOTA_MODELS_GUIDE.md).

### 1.6.5 Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) addresses a key limitation of ensemble methods: while ensembles achieve high accuracy, they are computationally expensive at inference time, requiring multiple forward passes. Distillation trains a single student model to mimic an ensemble teacher, preserving much of the performance gain while drastically reducing inference cost.

**Distillation Loss**: The student is trained on a combination of two objectives:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, p_{\text{student}}) + (1-\alpha) \cdot \mathcal{L}_{\text{KL}}(p_{\text{teacher}}, p_{\text{student}})$$

where:
- $\mathcal{L}_{\text{CE}}$ is cross-entropy loss with ground truth labels $y$
- $\mathcal{L}_{\text{KL}}$ is KL divergence between teacher and student probability distributions
- $\alpha \in [0,1]$ balances the two objectives

**Detailed Explanation**:

The cross-entropy term ensures the student learns from the ground truth labels:
$$\mathcal{L}_{\text{CE}}(y, p_{\text{student}}) = -\sum_{c=1}^{C} \mathbb{1}[y=c] \log p_{\text{student}}(c)$$

where $C$ is the number of classes and $\mathbb{1}[y=c]$ is 1 if the true label is $c$, else 0.

The KL divergence term transfers knowledge from the teacher:
$$\mathcal{L}_{\text{KL}}(p_{\text{teacher}}, p_{\text{student}}) = \sum_{c=1}^{C} p_{\text{teacher}}(c) \log \frac{p_{\text{teacher}}(c)}{p_{\text{student}}(c)}$$

The hyperparameter $\alpha$ controls the trade-off:
- $\alpha = 1$: Pure supervised learning (ignores teacher)
- $\alpha = 0$: Pure distillation (ignores ground truth)
- Typical values: $\alpha \in [0.1, 0.5]$ work well in practice

**Temperature Scaling**: To transfer richer information, both teacher and student logits $z$ are divided by temperature $T$ before applying softmax:

$$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

**Interpretation**: Higher temperatures ($T > 1$) produce softer probability distributions. Consider an example with 4 classes:

- **Hard prediction** ($T=1$): $[0.98, 0.01, 0.01, 0.00]$
  - The model is very confident about class 1
  - Provides almost no information about relationships between other classes
  
- **Soft prediction** ($T=3$): $[0.70, 0.15, 0.12, 0.03]$
  - Still predicts class 1, but with less confidence
  - Reveals that class 2 and 3 are more similar to class 1 than class 4
  - The relative ordering of classes 2 and 3 provides information about semantic similarity

This softer distribution reveals the teacher's uncertainty and relative similarities between classes—information lost when using only hard labels (one-hot vectors). The temperature effectively "smooths" the distribution, making it easier for the student to learn the fine-grained knowledge encoded in the teacher's predictions.

During distillation training:
$$p_{\text{teacher}}(T) = \text{softmax}(z_{\text{teacher}}/T)$$
$$p_{\text{student}}(T) = \text{softmax}(z_{\text{student}}/T)$$

And the KL divergence is computed at temperature $T$, then scaled by $T^2$ to ensure gradient magnitudes match the cross-entropy term:
$$\mathcal{L}_{\text{KL}} = T^2 \cdot \text{KL}(p_{\text{teacher}}(T) \| p_{\text{student}}(T))$$

**Why It Works**: The soft labels from the teacher ensemble contain information about class similarities and ambiguous cases that is not present in the one-hot ground truth labels. By training on this richer supervision signal, the student learns to mimic not just the teacher's final predictions but its confidence calibration and inter-class relationships.

For example, in AG News classification:
- Ground truth might simply say: "This is a Sports article"
- Teacher's soft labels might say: "90% Sports, 7% Business (because it discusses sports contracts), 2% World, 1% Sci/Tech"
- The student learns both the correct class AND the semantic relationships between categories

**Multi-Stage Distillation in This Project**:

1. **Stage 1 (LLM → Transformer)**: Distill knowledge from large language models (LLaMA 2-13B, Mistral-7B) into large transformers (DeBERTa-v3-Large), achieving 40× parameter reduction with minimal accuracy loss. Scripts in [scripts/training/distillation/distill_from_llama.py](./scripts/training/distillation/distill_from_llama.py) and [scripts/training/distillation/distill_from_mistral.py](./scripts/training/distillation/distill_from_mistral.py).

2. **Stage 2 (Ensemble → Single)**: Distill an ensemble of 5-7 diverse DeBERTa-Large models into a single DeBERTa-Large student, capturing ensemble benefits without ensemble inference cost. Configuration in [configs/training/advanced/knowledge_distillation/ensemble_distillation.yaml](./configs/training/advanced/knowledge_distillation/ensemble_distillation.yaml).

3. **Stage 3 (Compression)**: Apply INT8 quantization to the distilled model for 4× size reduction and faster CPU inference. Scripts in [scripts/optimization/quantization_optimization.py](./scripts/optimization/quantization_optimization.py).

Experimental results demonstrating accuracy retention through this pipeline are provided in [experiments/sota_experiments/phase3_llm_distillation.py](./experiments/sota_experiments/phase3_llm_distillation.py). Theoretical analysis and best practices are detailed in [SOTA_MODELS_GUIDE.md § Knowledge Distillation](./SOTA_MODELS_GUIDE.md).

## 1.7 Scope and Limitations

To establish appropriate expectations and clarify the boundaries of this project, we explicitly state what is and is not within scope.

### 1.7.1 Within Scope

This project provides:

- **Complete Text Classification Pipeline**: End-to-end workflow from raw text preprocessing through model training, evaluation, and deployment for supervised classification tasks. Pipeline documentation in [ARCHITECTURE.md](./ARCHITECTURE.md).

- **Rigorous Experimental Protocols**: Infrastructure for conducting statistically valid experiments including proper train-validation-test splits, multiple random seed evaluation, significance testing, and prevention of data leakage. Protocols detailed in [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md).

- **Comprehensive Model Coverage**: Implementations spanning classical machine learning baselines (Naive Bayes, SVM) through modern transformers (BERT, RoBERTa, DeBERTa) to large language models (LLaMA, Mistral) with parameter-efficient fine-tuning. Complete model zoo in [configs/models/](./configs/models/).

- **Advanced Training Techniques**: Support for ensemble methods, knowledge distillation, adversarial training, data augmentation, and multi-stage training pipelines. Training strategies in [configs/training/](./configs/training/).

- **Platform Portability**: Configurations and tooling for running experiments on consumer laptops, cloud platforms (Google Colab, Kaggle), and institutional compute clusters with automatic resource adaptation. Platform guide in [PLATFORM_OPTIMIZATION_GUIDE.md](./PLATFORM_OPTIMIZATION_GUIDE.md).

- **Extensive Documentation**: Technical documentation targeted at multiple expertise levels from beginners to advanced researchers, including theoretical treatments, API references, and step-by-step tutorials. Documentation index in [docs/](./docs/).

### 1.7.2 Explicitly Out of Scope

This project does not provide:

- **Novel Architectural Contributions**: We implement and combine existing published techniques rather than proposing new model architectures or training algorithms. Original research contributions are in the domain of systematic evaluation and rigorous experimental methodology.

- **Multilingual Classification**: Focus is on English-language text classification as determined by the AG News dataset. Extensions to multilingual scenarios would require different pre-trained models and evaluation protocols.

- **Streaming or Real-Time Inference**: The system is designed for batch processing and offline evaluation. While inference latency is measured and reported in [benchmarks/efficiency/](./benchmarks/efficiency/), we do not optimize for real-time serving constraints or provide production-grade serving infrastructure.

- **Adversarial Robustness Certification**: While we include adversarial training as an optional technique in [configs/training/advanced/adversarial_training.yaml](./configs/training/advanced/adversarial_training.yaml), we do not provide formally verified robustness guarantees or certified defense mechanisms against adversarial examples.

- **Automatic Data Collection**: We assume users have access to their data. The project provides tools for processing and analyzing data in [src/data/](./src/data/), but does not include web scraping, data purchasing, or crowdsourcing annotation pipelines.

- **Production Deployment Infrastructure**: While we provide basic Docker configurations in [deployment/docker/](./deployment/docker/) and deployment guidelines in [docs/user_guide/local_deployment.md](./docs/user_guide/local_deployment.md), comprehensive production infrastructure (Kubernetes orchestration, auto-scaling, monitoring, A/B testing) is beyond scope.

- **Graphical User Interfaces**: All interfaces are command-line based or notebook-based. We do not provide web dashboards or GUI tools for non-technical users, though Streamlit and Gradio apps are available in [app/](./app/) for demonstration purposes.

### 1.7.3 Assumptions and Prerequisites

**Data Assumptions**:

- **Balanced or Near-Balanced Classes**: Many techniques (ensemble diversity metrics, stratified sampling) assume approximately balanced class distributions. Highly imbalanced datasets may require additional techniques like class weighting or resampling not extensively covered in this framework.

- **Text Length**: Default configurations assume document length of 512 tokens or fewer, matching common transformer limits. Longer documents require alternative architectures (Longformer, BigBird) available in [configs/models/single/transformers/longformer/](./configs/models/single/transformers/longformer/) or truncation strategies.

- **Clean Labels**: We assume labels are accurate and consistent. Label noise handling is not explicitly addressed beyond standard regularization techniques.

- **Single-Label Classification**: The focus is on assigning each document to exactly one category. Multi-label classification (assigning multiple categories per document) requires modifications to loss functions and evaluation metrics.

**Computational Assumptions**:

- **GPU Access**: Training transformer-based models requires GPU acceleration. While CPU-only training is technically possible, it is prohibitively slow for models beyond simple baselines. Access to cloud platforms with GPU quotas or local GPU hardware is assumed. CPU-optimized configs available in [configs/models/recommended/tier_5_free_optimized/cpu_friendly/](./configs/models/recommended/tier_5_free_optimized/cpu_friendly/).

- **Internet Connectivity**: Downloading pre-trained model checkpoints requires stable internet access and sufficient bandwidth. Models range from hundreds of megabytes (BERT-Base) to tens of gigabytes (LLaMA-70B).

- **Storage Capacity**: Training runs generate checkpoints, logs, and cached preprocessed data. We recommend at least 50GB of available disk space for a typical experiment series.

**User Knowledge Assumptions**:

- **Python Programming**: Users should be comfortable with Python syntax, including functions, classes, imports, and common libraries (NumPy, pandas).

- **Command-Line Interfaces**: Basic familiarity with terminal commands, file paths, and environment variables.

- **Machine Learning Fundamentals**: Understanding of core concepts including train-validation-test splits, overfitting, hyperparameters, and evaluation metrics (accuracy, precision, recall, F1).

- **Deep Learning Basics** (for advanced features): Familiarity with neural network architectures, backpropagation, optimization algorithms, and regularization techniques.

Users without these prerequisites are encouraged to consult the beginner-level tutorials in [docs/level_1_beginner/](./docs/level_1_beginner/) which provide gentler introductions with more extensive explanations.

### 1.7.4 Known Limitations

**Limitation 1: Quadratic Complexity of Self-Attention**

Transformer architectures compute pairwise attention between all tokens, resulting in $O(n^2)$ memory and computational complexity where $n$ is sequence length. This limits practical sequence lengths to 512-1024 tokens for standard models.

**Mitigation**: We provide configurations for efficient transformers (Longformer, BigBird) that use sparse attention patterns to achieve $O(n)$ complexity in [configs/models/single/transformers/longformer/](./configs/models/single/transformers/longformer/). However, these are not the primary focus and have received less extensive tuning.

**Future Direction**: Integration of recent linear attention mechanisms (Performers, FNet, Linformer) could enable processing of longer documents without quadratic scaling. Tracked in [ROADMAP.md](./ROADMAP.md).

**Limitation 2: Memory Requirements for Large Models**

Even with parameter-efficient fine-tuning and quantization, the largest models (LLaMA-70B, Mixtral-8x7B) require 40-80GB of GPU memory. This exceeds the capacity of consumer GPUs and requires either model parallelism across multiple devices or cloud instances with high-memory GPUs.

**Mitigation**: QLoRA with 4-bit quantization reduces memory requirements by approximately 4×, making models up to 13B parameters accessible on consumer GPUs with 16-24GB VRAM. Configurations in [configs/training/efficient/qlora/](./configs/training/efficient/qlora/). For larger models, we provide configurations for gradient checkpointing and CPU offloading that trade computation time for memory in [configs/training/efficient/](./configs/training/efficient/).

**Future Direction**: Exploration of more aggressive quantization (INT4, INT2) and structured pruning could further reduce memory footprints. Research directions in [ROADMAP.md](./ROADMAP.md).

**Limitation 3: Ensemble Inference Overhead**

While ensemble methods achieve the highest accuracy in our experiments (results in [benchmarks/accuracy/ensemble_results.json](./benchmarks/accuracy/ensemble_results.json)), they require multiple forward passes at inference time, increasing latency proportionally to ensemble size. A 7-model ensemble is 7× slower than a single model.

**Mitigation**: Knowledge distillation compresses ensemble knowledge into a single student model, retaining typically 90-95% of ensemble improvement while restoring single-model inference speed. Distillation configurations in [configs/training/advanced/knowledge_distillation/](./configs/training/advanced/knowledge_distillation/).

**Future Direction**: Investigation of fast ensemble approximation methods such as dropout ensembles or BatchEnsemble could provide accuracy benefits closer to full ensembles with minimal computational overhead. Tracked in [ROADMAP.md](./ROADMAP.md).

**Limitation 4: Platform-Specific Quota Management**

Our platform detection and quota management systems use heuristics based on typical platform limits (Colab 12-hour sessions, Kaggle 30-hour GPU weekly quotas). These limits may change over time and vary based on account type (free tier versus paid subscriptions).

**Mitigation**: Conservative checkpoint intervals and explicit quota tracking provide safety margins in [src/deployment/quota_tracker.py](./src/deployment/quota_tracker.py). Users can override automatic settings if they have specific information about their quota limits through [configs/quotas/](./configs/quotas/).

**Future Direction**: Integration with platform APIs (when available) could provide real-time quota information rather than relying on hardcoded assumptions. Tracked in [ROADMAP.md](./ROADMAP.md).

**Limitation 5: English-Only Evaluation**

All experiments are conducted on the English-language AG News dataset. While the underlying transformer models (mBERT, XLM-R) support multilingual text, we have not evaluated cross-lingual transfer or performance on non-English datasets.

**Mitigation**: The framework architecture is language-agnostic, and extending to other languages primarily requires different datasets and evaluation protocols rather than code modifications. The modular design in [src/](./src/) supports such extensions.

**Future Direction**: Systematic evaluation on multilingual benchmarks (XNLI, PAWS-X) and investigation of cross-lingual transfer learning techniques. Multilingual support tracked in [ROADMAP.md](./ROADMAP.md).

## 1.8 Organization of Documentation

This project provides extensive documentation organized according to progressive disclosure principles, allowing users to engage at levels appropriate to their expertise and goals.

### 1.8.1 Entry Points by User Type

**New Users Seeking Quick Start**:

- [QUICK_START.md](./QUICK_START.md): Minimal setup to first working model in under 10 minutes
- [quickstart/auto_start.py](./quickstart/auto_start.py): Single-command demo requiring no configuration
- [docs/getting_started/installation.md](./docs/getting_started/installation.md): Comprehensive installation guide for different platforms
- [quickstart/decision_tree.py](./quickstart/decision_tree.py): Interactive CLI for guided model selection

**Practitioners Seeking Best Practices**:

- [SOTA_MODELS_GUIDE.md](./SOTA_MODELS_GUIDE.md): Model selection flowcharts and performance comparisons
- [docs/user_guide/](./docs/user_guide/): Detailed guides on data preparation, training, evaluation, and deployment
- [docs/best_practices/](./docs/best_practices/): Domain-specific recommendations for model selection, hyperparameter tuning, and avoiding common pitfalls
- [notebooks/01_tutorials/](./notebooks/01_tutorials/): Interactive Jupyter tutorials with executable examples

**Researchers Seeking Theoretical Depth**:

- [OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md): Complete theoretical framework including proofs and statistical guarantees
- [ARCHITECTURE.md](./ARCHITECTURE.md): Detailed system architecture with transformer internals and optimization techniques
- [docs/level_3_advanced/](./docs/level_3_advanced/): Advanced topics including custom model development and research workflows
- [experiments/](./experiments/): Ablation studies, hyperparameter searches, and baseline comparisons with full experimental protocols

**Developers Seeking to Extend the System**:

- [docs/developer_guide/](./docs/developer_guide/): Architecture overview, coding standards, and extension points
- [docs/api_reference/](./docs/api_reference/): Complete API documentation for all modules
- [CONTRIBUTING.md](./CONTRIBUTING.md): Guidelines for contributing code, documentation, and bug reports
- [ARCHITECTURE.md](./ARCHITECTURE.md): Design patterns and architectural decisions

### 1.8.2 Document Hierarchy and Specialization

To avoid duplication and maintain a single source of truth, different documents have clearly defined scopes:

**[README.md](./README.md)** (This Document):
- High-level introduction to the project and its motivation
- Overview of theoretical foundations with links to detailed treatments
- Dataset description and experimental setup (Section 2)
- Quick start instructions and navigation guide

**[OVERFITTING_PREVENTION.md](./OVERFITTING_PREVENTION.md)**:
- Complete mathematical treatment of generalization theory
- Detailed description of overfitting prevention architecture
- Empirical validation of prevention mechanisms
- Decision trees for selecting prevention strategies

**[ARCHITECTURE.md](./ARCHITECTURE.md)**:
- System architecture and component interactions
- Detailed explanations of transformer architectures
- Optimization techniques and efficiency considerations
- Design patterns and extension points

**[SOTA_MODELS_GUIDE.md](./SOTA_MODELS_GUIDE.md)**:
- Model zoo overview and tier descriptions
- Performance benchmarks and accuracy-efficiency trade-offs
- Hyperparameter recommendations for each model family
- Selection flowcharts for different use cases

**[PLATFORM_OPTIMIZATION_GUIDE.md](./PLATFORM_OPTIMIZATION_GUIDE.md)**:
- Platform detection mechanisms
- Resource profiling and quota management
- Checkpoint synchronization strategies
- Platform-specific optimization techniques

**User Guides in [docs/user_guide/](./docs/user_guide/)**:
- Step-by-step instructions for common workflows
- Code examples and configuration templates
- Troubleshooting common issues
- Best practices for specific tasks

**API Reference in [docs/api_reference/](./docs/api_reference/)**:
- Function and class signatures
- Parameter descriptions and types
- Return value specifications
- Usage examples for each API component

This hierarchical organization ensures that:

1. Users can find information at the appropriate level of detail
2. Each piece of information has a single authoritative source
3. Updates to concepts need to be made in only one location
4. Cross-references guide users to related information in other documents

For a visual overview of the documentation structure, see [docs/00_START_HERE.md](./docs/00_START_HERE.md), which provides a guided navigation map based on your learning objectives and experience level.



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
