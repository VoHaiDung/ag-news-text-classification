# ARCHITECTURE.md

### Executive Summary

The AG News Text Classification system represents a state-of-the-art, production-ready natural language processing platform specifically engineered for high-precision news article categorization. This comprehensive architecture document delineates the intricate design decisions, structural compositions, and technological foundations that underpin the system's capability to achieve superior classification performance across four distinct news categories: World, Sports, Business, and Science/Technology.

The architecture embraces modern microservices principles, employing containerized deployments, service mesh orchestration, and multi-protocol API support to ensure scalability, maintainability, and extensibility. Through the integration of cutting-edge transformer-based models, sophisticated ensemble techniques, and advanced training strategies, the system achieves state-of-the-art performance while maintaining production-grade reliability and efficiency.

### Project Structure

```
ag-news-text-classification/
├── configs/           # Configuration management system
├── data/             # Multi-tier data storage architecture
├── src/              # Core source code implementation
├── experiments/      # Experimental framework and benchmarks
├── monitoring/       # Observability and monitoring stack
├── security/         # Security and compliance infrastructure
├── deployment/       # Multi-cloud deployment configurations
├── tests/           # Comprehensive testing suite
├── notebooks/       # Interactive development and analysis
├── app/            # User interface applications
├── scripts/        # Automation and utility scripts
├── prompts/        # Prompt engineering templates
├── docs/           # Technical documentation
├── benchmarks/     # Performance benchmarking results
└── tools/          # Development and debugging tools
```

## System Architecture Overview

### Architectural Philosophy and Design Principles

The AG News Text Classification system architecture is predicated upon several fundamental design principles that guide its implementation and evolution:

**1. Modularity and Separation of Concerns**: Each component within the system is designed to handle a specific responsibility, facilitating independent development, testing, and deployment. This modular approach enables teams to work on different aspects of the system simultaneously without creating interdependencies that could impede progress.

**2. Scalability Through Horizontal Distribution**: The architecture prioritizes horizontal scalability over vertical scaling, allowing the system to handle increased load by adding more instances rather than upgrading individual machines. This approach provides better fault tolerance and cost efficiency at scale.

**3. API-First Design**: All functionalities are exposed through well-defined APIs, ensuring that the system can be integrated with various client applications and third-party services. The multi-protocol support (REST, gRPC, GraphQL) accommodates different use cases and performance requirements.

**4. Intelligence at the Edge**: By implementing caching strategies and edge computing capabilities, the system minimizes latency and reduces the load on core services, improving overall responsiveness and user experience.

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Client Applications                          │
├─────────────┬──────────────┬──────────────┬─────────────────────────────┤
│  Web Apps   │ Mobile Apps  │  CLI Tools   │   Third-party Systems       │
└─────────────┴──────────────┴──────────────┴─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Gateway Layer                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │   Kong/Nginx: Rate Limiting, Auth, Load Balancing, SSL/TLS       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

The **API Gateway Layer** serves as the single entry point for all external communications, implementing critical cross-cutting concerns such as:

- **Authentication and Authorization**: Validates client credentials and enforces access control policies using JWT tokens and OAuth 2.0 protocols
- **Rate Limiting**: Implements token bucket and sliding window algorithms to prevent API abuse and ensure fair resource allocation
- **Load Balancing**: Distributes incoming requests across multiple service instances using round-robin, least-connections, or weighted algorithms
- **SSL/TLS Termination**: Handles encryption and decryption of HTTPS traffic, offloading this computationally intensive task from backend services

### Service Layer Architecture

The service layer implements a microservices architecture pattern, where each service is responsible for a specific business capability:

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  REST API    │    │   gRPC API   │    │ GraphQL API  │
│  (FastAPI)   │    │ (gRPC-Python)│    │  (Graphene)  │
└──────┬──────┘     └───────┬──────┘    └───────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                ┌───────────────────────┐
                │   Service Registry    │
                │     (Consul/etcd)     │
                └───────────┬───────────┘
                            │
    ┌───────────┬───────────┼───────────┬───────────┬───────────┐
    ▼           ▼           ▼           ▼           ▼           ▼
┌──────────┐┌─────────┐┌──────────┐┌──────────┐┌──────────┐┌────────┐
│Prediction││Training ││   Data   ││  Model   ││Monitoring││ Cache  │
│ Service  ││Service  ││ Service  ││Management││ Service  ││Service │
└──────────┘└─────────┘└──────────┘└──────────┘└──────────┘└────────┘
```

#### Prediction Service

The **Prediction Service** constitutes the core inference engine of the system, responsible for processing text classification requests in real-time. This service implements several sophisticated features:

- **Batch Processing Optimization**: Aggregates multiple prediction requests into batches to leverage GPU parallelization effectively
- **Model Versioning**: Maintains multiple model versions simultaneously, enabling A/B testing and gradual rollouts
- **Fallback Mechanisms**: Implements circuit breaker patterns to handle model failures gracefully
- **Response Caching**: Utilizes Redis-based caching for frequently requested predictions

#### Training Service

The **Training Service** orchestrates the entire model training lifecycle, from data preparation to model evaluation:

- **Distributed Training Coordination**: Manages multi-GPU and multi-node training using Horovod or PyTorch Distributed
- **Hyperparameter Optimization**: Integrates with Optuna or Ray Tune for automated hyperparameter search
- **Experiment Tracking**: Logs all training metrics, parameters, and artifacts to MLflow or Weights & Biases
- **Resource Management**: Implements queue-based job scheduling to optimize GPU utilization

## Core Model Architecture

### Deep Learning Model Foundation

The system's machine learning capabilities are built upon a hierarchical architecture of transformer-based models, each contributing unique strengths to the ensemble prediction system.

#### DeBERTa-v3 Architecture Analysis

DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention) represents the cornerstone of our model architecture, offering several architectural innovations that contribute to its superior performance:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeBERTa-v3-XLarge Model                     │
│                      (900M Parameters)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Processing Layer:                                        │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│  │   Input    │───▶│  WordPiece │───▶│ Position &  │           │
│  │   Text     │    │  Tokenizer │    │  Segment    │           │
│  └────────────┘    └────────────┘    │  Embeddings │           │
│                                       └────────────┘           │
│                                             │                   │
│                                             ▼                   │
│  Disentangled Attention Mechanism:                             │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Content-to-Content  ⊕  Content-to-Position  ⊕      │      │
│  │  Position-to-Content ⊕  Position-to-Position         │      │
│  └──────────────────────────────────────────────────────┘      │
│                                             │                   │
│                                             ▼                   │
│  Transformer Layers (48 layers):                               │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  For each layer i ∈ [1, 48]:                        │      │
│  │  ┌──────────────────────────────────────────────┐   │      │
│  │  │  MultiHeadAttention(Q, K, V) with:           │   │      │
│  │  │  - Hidden Size: 1536                         │   │      │
│  │  │  - Attention Heads: 24                       │   │      │
│  │  │  - Head Dimension: 64                        │   │      │
│  │  └──────────────────────────────────────────────┘   │      │
│  │                         ▼                            │      │
│  │  ┌──────────────────────────────────────────────┐   │      │
│  │  │  Feed-Forward Network:                       │   │      │
│  │  │  Linear(1536, 6144) → GELU → Linear(6144, 1536)│ │      │
│  │  └──────────────────────────────────────────────┘   │      │
│  └──────────────────────────────────────────────────────┘      │
│                                │                                │
│                                ▼                                │
│  Enhanced Mask Decoder (EMD):                                  │
│  ┌────────────────────────────────────────────────┐           │
│  │  Replaces [MASK] tokens with enhanced          │           │
│  │  position-aware representations                 │           │
│  └────────────────────────────────────────────────┘           │
│                                │                                │
│                                ▼                                │
│  Classification Head:                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐              │
│  │   Pooling   │─▶│  Dropout    │─▶│  Linear │─▶[Logits]   │
│  │  (CLS/Mean) │  │   (0.1)     │  │(1536, 4)│              │
│  └─────────────┘  └─────────────┘  └─────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

**Key Architectural Innovations:**

1. **Disentangled Attention Mechanism**: Unlike traditional self-attention where position and content information are combined, DeBERTa-v3 processes them separately, allowing the model to better capture relative position dependencies. The attention score is computed as:

   ```
   A[i,j] = Q_c[i] × K_c[j] + Q_c[i] × K_r[j] + K_c[j] × Q_r[i]
   ```

   Where Q_c/K_c represent content vectors and Q_r/K_r represent relative position vectors.

2. **Enhanced Mask Decoder**: The EMD layer specifically optimizes for masked language modeling tasks by incorporating absolute position information when replacing masked tokens, improving the model's understanding of contextual relationships.

3. **Gradient Disentanglement**: The separated attention computation allows for more stable gradient flow during backpropagation, reducing the vanishing gradient problem in deep networks.

### Ensemble Architecture Strategy

The ensemble architecture employs a sophisticated multi-level combination strategy that leverages the complementary strengths of different model architectures:

```
                    ┌─────────────────────────┐
                    │    Input Text Sample    │
                    │   "Tech giant announces │
                    │    new AI breakthrough"  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Text Preprocessing    │
                    │  • Normalize Unicode    │
                    │  • Remove HTML tags     │
                    │  • Expand contractions  │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   DeBERTa-v3  │      │    RoBERTa    │      │   ELECTRA     │
│   Processor   │      │   Processor   │      │   Processor   │
├───────────────┤      ├───────────────┤      ├───────────────┤
│ • Disentangled│      │ • Byte-level  │      │ • Discrimin.  │
│   Attention   │      │   BPE         │      │   Pretraining │
│ • EMD Layer   │      │ • Dynamic Mask│      │ • RTD Task    │
└───────┬───────┘      └───────┬───────┘      └───────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Logits:[2.3,            Logits:[2.1,           Logits:[2.2,
    -1.2, -0.8, 0.5]        -1.0, -0.9, 0.4]       -1.1, -0.7, 0.3]
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Ensemble Strategy   │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │Weighted Vote │ │   Stacking   │ │   Bayesian   │
        │  α₁=0.45     │ │  XGBoost     │ │  Model Avg.  │
        │  α₂=0.35     │ │  CatBoost    │ │  with MCMC   │
        │  α₃=0.20     │ │  LightGBM    │ │              │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                │               │               │
                └───────────────┼───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Confidence Calibration│
                    │  • Temperature Scaling │
                    │  • Platt Scaling       │
                    │  • Isotonic Regression │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Final Prediction    │
                    │  Class: Technology    │
                    │  Confidence: 0.943    │
                    │  Uncertainty: ±0.021  │
                    └───────────────────────┘
```

#### Ensemble Combination Strategies

**1. Weighted Voting Ensemble**

The weighted voting mechanism assigns learned weights to each model based on their individual performance metrics. The final prediction is computed as:

```
P(y|x) = Σᵢ αᵢ × Pᵢ(y|x)
```

Where αᵢ represents the weight for model i, optimized through cross-validation to minimize the ensemble's classification error.

**2. Stacking Meta-Learning**

The stacking approach uses the base model predictions as features for a second-level meta-learner:

- **Level 0 Models**: DeBERTa-v3, RoBERTa, ELECTRA generate probability distributions
- **Level 1 Meta-Learner**: XGBoost/CatBoost trained on out-of-fold predictions
- **Feature Engineering**: Includes prediction entropy, confidence scores, and pairwise differences

**3. Bayesian Model Averaging**

Implements a probabilistic approach to model combination, accounting for model uncertainty:

```
P(y|x, D) = Σₘ P(y|x, m) × P(m|D)
```

This approach naturally handles model uncertainty and provides calibrated confidence estimates.

## Data Pipeline Architecture

### Comprehensive Data Processing Framework

The data pipeline architecture implements a sophisticated multi-stage processing framework designed to handle diverse data sources, apply advanced augmentation techniques, and ensure data quality throughout the machine learning lifecycle.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   AG News    │  │   External   │  │  Synthetic   │               │
│  │   Dataset    │  │  News Corpus │  │   GPT-4 Gen  │               │
│  │  (120K docs) │  │  (500K docs) │  │  (50K docs)  │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            ▼                                        │
│                 ┌──────────────────┐                                │
│                 │  Data Validator  │                                │
│                 │ • Schema Check   │                                │
│                 │ • Quality Metrics│                                │
│                 │ • Deduplication  │                                │
│                 └─────────┬────────┘                                │
└───────────────────────────┼─────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Processing Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Text Cleaning                                            │
│  ┌────────────────────────────────────────────────────────┐       │
│  │ • HTML/XML tag removal                                 │       │
│  │ • Unicode normalization (NFKC)                         │       │
│  │ • Whitespace standardization                           │       │
│  │ • URL/Email pattern handling                           │       │
│  │ • Special character processing                         │       │
│  └────────────────────────────────────────────────────────┘       │
│                            ▼                                        │
│  Stage 2: Linguistic Processing                                    │
│  ┌────────────────────────────────────────────────────────┐       │
│  │ • Sentence segmentation (spaCy)                        │       │
│  │ • Tokenization (BPE/WordPiece)                        │       │
│  │ • Part-of-speech tagging                              │       │
│  │ • Named entity recognition                            │       │
│  │ • Dependency parsing                                  │       │
│  └────────────────────────────────────────────────────────┘       │
│                            ▼                                        │
│  Stage 3: Feature Engineering                                      │
│  ┌────────────────────────────────────────────────────────┐       │
│  │ • TF-IDF vectors (baseline)                           │       │
│  │ • Word embeddings (GloVe/Word2Vec)                    │       │
│  │ • Contextual embeddings (BERT-based)                  │       │
│  │ • Statistical features (length, complexity)           │       │
│  │ • Domain-specific features                            │       │
│  └────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Augmentation Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │   Back      │  │  Paraphrase │  │  Adversarial│               │
│  │Translation  │  │  Generation │  │   Examples  │               │
│  │ • EN→DE→EN  │  │ • T5-based  │  │ • TextFooler│               │
│  │ • EN→FR→EN  │  │ • GPT-based │  │ • BERT-Attack│              │
│  └─────┬───────┘  └─────┬───────┘  └─────┬───────┘               │
│         │                │                 │                        │
│         └────────────────┼─────────────────┘                        │
│                          ▼                                          │
│  ┌────────────────────────────────────────────────────────┐       │
│  │              Augmentation Controller                    │       │
│  │  • Maintains class balance                             │       │
│  │  • Controls augmentation ratio (1:3)                   │       │
│  │  • Ensures semantic preservation                       │       │
│  │  • Quality filtering (perplexity-based)               │       │
│  └────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Augmentation Strategies

#### Back-Translation Augmentation

Back-translation leverages neural machine translation models to generate semantically equivalent variations of the original text:

**Process Flow:**
1. **Forward Translation**: Translate original English text to intermediate language (German/French/Spanish)
2. **Back Translation**: Translate back to English using a different model
3. **Quality Filtering**: Filter based on semantic similarity (cosine similarity > 0.85)
4. **Diversity Check**: Ensure lexical diversity (Jaccard distance > 0.2)

**Mathematical Formulation:**
```
Given text x in English:
x' = T_back(T_forward(x, L_intermediate), English)
where T represents translation function and L represents language
```

#### MixUp and Manifold Mixup

MixUp creates synthetic training examples by interpolating between pairs of examples in the feature space:

```
x̃ = λ × x_i + (1 - λ) × x_j
ỹ = λ × y_i + (1 - λ) × y_j
where λ ~ Beta(α, α), typically α = 0.2
```

**Implementation Details:**
- Applied at embedding layer (Manifold Mixup) for better regularization
- Sampling strategy ensures inter-class mixing for improved decision boundaries
- Adaptive λ based on training progress (curriculum-based adjustment)

## Training Pipeline Architecture

### Advanced Training Framework

The training pipeline implements a sophisticated multi-stage approach incorporating curriculum learning, adversarial training, and knowledge distillation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Training Orchestration System                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                 Stage 1: Preprocessing                    │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │  • Data loading and validation                           │      │
│  │  • Train/Val/Test splitting (60/20/20)                  │      │
│  │  • Stratified sampling for class balance                 │      │
│  │  • Cross-validation fold generation (5-fold)             │      │
│  └─────────────────────────┬────────────────────────────────┘      │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │              Stage 2: Curriculum Learning                 │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │                                                           │      │
│  │  Epoch 1-5: Easy samples                                 │      │
│  │  ┌─────────────────────────────────────────┐            │      │
│  │  │ • Short texts (< 50 tokens)             │            │      │
│  │  │ • High confidence labels                │            │      │
│  │  │ • Clear category distinctions           │            │      │
│  │  │ • Learning rate: 5e-5                   │            │      │
│  │  └─────────────────────────────────────────┘            │      │
│  │                                                           │      │
│  │  Epoch 6-15: Medium difficulty                           │      │
│  │  ┌─────────────────────────────────────────┐            │      │
│  │  │ • Standard length (50-200 tokens)       │            │      │
│  │  │ • Include augmented samples             │            │      │
│  │  │ • Mixed difficulty examples             │            │      │
│  │  │ • Learning rate: 3e-5                   │            │      │
│  │  └─────────────────────────────────────────┘            │      │
│  │                                                           │      │
│  │  Epoch 16-25: Hard samples                               │      │
│  │  ┌─────────────────────────────────────────┐            │      │
│  │  │ • Long texts (> 200 tokens)             │            │      │
│  │  │ • Ambiguous/borderline cases            │            │      │
│  │  │ • Adversarial examples                  │            │      │
│  │  │ • Learning rate: 1e-5                   │            │      │
│  │  └─────────────────────────────────────────┘            │      │
│  └─────────────────────────┬────────────────────────────────┘      │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │           Stage 3: Adversarial Training                  │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │                                                           │      │
│  │  FGM (Fast Gradient Method):                             │      │
│  │  ┌─────────────────────────────────────────┐            │      │
│  │  │  x_adv = x + ε × sign(∇_x L(θ, x, y))  │            │      │
│  │  │  where ε = 0.5 (embedding perturbation) │            │      │
│  │  └─────────────────────────────────────────┘            │      │
│  │                                                           │      │
│  │  PGD (Projected Gradient Descent):                       │      │
│  │  ┌─────────────────────────────────────────┐            │      │
│  │  │  For k iterations:                      │            │      │
│  │  │  x^(k+1) = Π(x^k + α × sign(∇L))       │            │      │
│  │  │  where Π projects onto ε-ball          │            │      │
│  │  └─────────────────────────────────────────┘            │      │
│  └─────────────────────────┬────────────────────────────────┘      │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │          Stage 4: Knowledge Distillation                 │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │                                                           │      │
│  │  Teacher: GPT-4 / Ensemble Model                         │      │
│  │  Student: Target Model                                   │      │
│  │                                                           │      │
│  │  Loss = α × CE(y_true, y_student) +                      │      │
│  │         β × KL(y_teacher, y_student) +                   │      │
│  │         γ × MSE(h_teacher, h_student)                    │      │
│  │                                                           │      │
│  │  where: α=0.5, β=0.3, γ=0.2                             │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Optimization Strategies

#### Learning Rate Scheduling

The system implements sophisticated learning rate scheduling strategies to optimize convergence:

```
┌─────────────────────────────────────────────────────────┐
│              Learning Rate Schedule                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Cosine Annealing with Warm Restarts:                  │
│                                                          │
│  LR ▲                                                   │
│     │     ╱\      ╱\      ╱\                          │
│ 5e-5├────╱  \    ╱  \    ╱  \                         │
│     │   ╱    \  ╱    \  ╱    \                        │
│ 3e-5├──╱      \╱      \╱      \                       │
│     │                           \                      │
│ 1e-5├─────────────────────────────\___                │
│     │                                                   │
│   0 └────┬────┬────┬────┬────┬────┬────▶             │
│         5    10   15   20   25   30      Epochs       │
│                                                          │
│  Formula: lr = lr_min + 0.5 × (lr_max - lr_min) ×      │
│           (1 + cos(π × T_cur / T_max))                 │
└─────────────────────────────────────────────────────────┘
```

#### Gradient Accumulation and Clipping

To handle large models with limited GPU memory:

```python
Gradient Accumulation Strategy:
- Accumulation steps: 4
- Effective batch size: physical_batch × accumulation_steps
- Gradient clipping: max_norm = 1.0
- Mixed precision training: FP16 with dynamic loss scaling
```

## API and Service Architecture

### Multi-Protocol API Design

The system implements a sophisticated API architecture supporting multiple protocols to accommodate diverse client requirements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        API Architecture                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    REST API (FastAPI)                     │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │                                                           │      │
│  │  Endpoints:                                               │      │
│  │  ┌─────────────────────────────────────────────────┐    │      │
│  │  │ POST   /api/v1/predict                          │    │      │
│  │  │ POST   /api/v1/batch_predict                    │    │      │
│  │  │ GET    /api/v1/models                           │    │      │
│  │  │ POST   /api/v1/train                            │    │      │
│  │  │ GET    /api/v1/metrics                          │    │      │
│  │  │ WS     /api/v1/stream                           │    │      │
│  │  └─────────────────────────────────────────────────┘    │      │
│  │                                                           │      │
│  │  Request Flow:                                            │      │
│  │  Client → Validation → Auth → Rate Limit → Handler       │      │
│  │         → Service → Response → Logging                   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                   gRPC Service                            │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │                                                           │      │
│  │  Service Definitions:                                     │      │
│  │  ┌─────────────────────────────────────────────────┐    │      │
│  │  │ service ClassificationService {                  │    │      │
│  │  │   rpc Predict(PredictRequest)                   │    │      │
│  │  │       returns (PredictResponse);                │    │      │
│  │  │   rpc StreamPredict(stream PredictRequest)      │    │      │
│  │  │       returns (stream PredictResponse);         │    │      │
│  │  │   rpc BatchPredict(BatchPredictRequest)         │    │      │
│  │  │       returns (BatchPredictResponse);           │    │      │
│  │  │ }                                                │    │      │
│  │  └─────────────────────────────────────────────────┘    │      │
│  │                                                           │      │
│  │  Features:                                                │      │
│  │  • Binary protocol (Protocol Buffers)                    │      │
│  │  • Bidirectional streaming                               │      │
│  │  • Multiplexing over single connection                   │      │
│  │  • Built-in load balancing                              │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    GraphQL API                            │      │
│  ├──────────────────────────────────────────────────────────┤      │
│  │                                                           │      │
│  │  Schema:                                                  │      │
│  │  ┌─────────────────────────────────────────────────┐    │      │
│  │  │ type Query {                                     │    │      │
│  │  │   predict(text: String!): Prediction            │    │      │
│  │  │   models: [Model]                               │    │      │
│  │  │   metrics(modelId: ID!): Metrics                │    │      │
│  │  │ }                                                │    │      │
│  │  │                                                  │    │      │
│  │  │ type Mutation {                                  │    │      │
│  │  │   trainModel(config: TrainConfig): Job          │    │      │
│  │  │   updateModel(id: ID!, params: ModelParams)     │    │      │
│  │  │ }                                                │    │      │
│  │  │                                                  │    │      │
│  │  │ type Subscription {                              │    │      │
│  │  │   trainingProgress(jobId: ID!): TrainingStatus  │    │      │
│  │  │ }                                                │    │      │
│  │  └─────────────────────────────────────────────────┘    │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Service Mesh Architecture

The service mesh provides advanced traffic management, security, and observability:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Istio Service Mesh                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐ │
│  │   Service A  │◀──────▶│   Service B  │◀──────▶│   Service C  │ │
│  │   ┌──────┐   │        │   ┌──────┐   │        │   ┌──────┐   │ │
│  │   │Envoy │   │        │   │Envoy │   │        │   │Envoy │   │ │
│  │   │Proxy │   │        │   │Proxy │   │        │   │Proxy │   │ │
│  │   └──────┘   │        │   └──────┘   │        │   └──────┘   │ │
│  └──────────────┘        └──────────────┘        └──────────────┘ │
│         ▲                        ▲                        ▲        │
│         │                        │                        │        │
│         └────────────────────────┼────────────────────────┘        │
│                                  │                                  │
│                         ┌────────▼────────┐                        │
│                         │   Control Plane │                        │
│                         │   • Pilot       │                        │
│                         │   • Citadel     │                        │
│                         │   • Galley      │                        │
│                         └─────────────────┘                        │
│                                                                      │
│  Features:                                                          │
│  • Automatic load balancing (Round-robin, Least request)           │
│  • Fine-grained traffic control (A/B testing, Canary)              │
│  • Circuit breaking and retry logic                                │
│  • Mutual TLS for service-to-service communication                 │
│  • Distributed tracing with Jaeger                                 │
│  • Metrics collection with Prometheus                              │
└─────────────────────────────────────────────────────────────────────┘
```
