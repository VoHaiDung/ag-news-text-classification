# AG News Text Classification

## Introduction

The effective classification of news articles, a fundamental task in Natural Language Processing (NLP), is increasingly challenged by a **methodological gap**: while individual techniques advance rapidly, the lack of a unified experimental environment hinders fair comparison and a holistic understanding of their interactive effects. To address this, our project directly confronts the task of multi-class text classification on the **AG News dataset - a collection of English-language articles categorized into four distinct topics: World, Sports, Business, and Sci/Tech**. Rather than treating AG News as a mere performance benchmark, we utilize it as a **case study environment** to engineer a **holistic framework**, designed to systematically investigate how a wide spectrum of state-of-the-art methodologies interact and contribute to addressing the specific challenges of news classification.

Our framework is designed as a laboratory to **deconstruct** the news classification problem. It facilitates a systematic inquiry into key axes of performance, such as: the interplay between **model architecture** and text format; the comparative efficacy of **data-centric strategies** like domain adaptation versus synthetic augmentation; the impact of **advanced training paradigms** on model robustness against linguistic variance; and the verifiability of the model's **internal reasoning mechanisms** through interpretability methods. **In this project, we present not only the architecture of this platform but also report the initial empirical findings from these investigations**.

Our framework facilitates these investigations through four methodological pillars, each demonstrated directly on the AG News task:

1. **Modeling News Structure with Architectural Flexibility**: To handle the varied length and structure of news content, from short briefs to in-depth analyses, the framework integrates both standard Transformers (e.g., DeBERTa) and long-context architectures (e.g., Longformer). Advanced ensemble methods are employed to aggregate signals from these different architectures, aiming to resolve ambiguity in articles that reside at the intersection of categories like "Business" and "World."

2. **Adapting Large Language Models to the News Corpus**: To efficiently fine-tune large language models on the AG News corpus, we deeply integrate **Parameter-Efficient Fine-Tuning (PEFT)** techniques. **Advanced training** strategies are applied to address domain-specific challenges; for instance, Adversarial Training is leveraged to enhance model resilience against the subtle yet meaningful phrasal variations common in journalism, while **Knowledge Distillation** from models like GPT-4 enables the generation of nuanced, news-specific training labels or rationales.

3. **Enhancing Performance with External News Data**: In line with **Data-Centric AI** principles, we investigate the impact of enriching the AG News dataset. The framework facilitates **Domain-Adaptive Pretraining (DAPT)** on other large news corpora (e.g., Reuters, BBC News) to imbue the model with a broader understanding of journalistic language. Data augmentation is also used to address potential imbalances across news categories.

4. **Evaluation and Interpretability in the Context of News Classification**: To ensure that the classification of news articles is transparent and trustworthy, our evaluation protocol extends beyond standard metrics. We assess model **robustness** against journalistic paraphrasing and measure **efficiency** for real-world viability. Critically, **Explainable AI (XAI)** tools are used to pinpoint the specific keywords or sentences that drive a classification—for example, identifying financial terminology in a "Business" article.

In essence, this project transcends a singular solution for AG News to become a **structured experimental platform** for the community. By providing the means to conduct reproducible research, we hope to empower the field to move beyond a singular focus on leaderboards towards a more **principled**, **mechanistic understanding** of text classification systems. The completeness of this platform is demonstrated by its extension beyond research workflows to a **full MLOps pipeline**, thereby addressing the **research-to-production gap**.

## Model Architecture

![Pipeline Diagram](images/pipeline.png)

## Dataset

This project is centered on the **AG News (AG's Corpus of News Articles)** dataset, a canonical benchmark for topic classification first introduced by Zhang et al. (2015). The corpus consists of 120,000 training and 7,600 test samples, each comprising a concatenated title and description from news articles. These instances are distributed evenly across four high-level categories: **World, Sports, Business, and Science/Technology**.

Beyond its scale and balanced nature, the AG News dataset presents several salient challenges that make it a compelling testbed for advanced NLP methodologies. For instance:

- The inherent **semantic overlap** between categories (e.g., a technology article about a business merger) necessitates the sophisticated disambiguation capabilities of our proposed **ensemble models** and **contrastive learning** strategies.
- The stylistic variance and presence of concise, often ambiguous, text snippets motivate our investigation into **data augmentation** techniques and the robustness conferred by **adversarial training**.
- The standard instance length makes it a suitable baseline, but also highlights the need to evaluate how models like **DeBERTa** and **Longformer** generalize when exposed to longer, more context-rich documents.

Crucially, our project treats the AG News dataset not as an isolated resource, but as the **core component of a broader data ecosystem** designed to rigorously test our framework. This ecosystem is extended with:

1. **External News Corpora**: Large-scale datasets such as Reuters and BBC News are leveraged for **Domain-Adaptive Pretraining (DAPT)**, allowing models to acquire a richer understanding of journalistic language prior to fine-tuning on AG News.
2. **Systematic Data Augmentation**: A suite of augmentation techniques, from **back-translation** to **GPT-4-based paraphrasing**, is employed to create diverse training sets aimed at improving model generalization and data efficiency.
3. **Contrast and Adversarial Sets**: To move beyond standard I.I.D. evaluation, we generate specialized test sets to systematically measure model robustness and diagnose failure modes under controlled linguistic perturbations.

This multi-faceted data strategy, with AG News at its center, provides the empirical foundation for our systematic investigation. To ensure the reproducibility of this work, the core AG News dataset is made publicly accessible through multiple established channels, including the [Hugging Face Datasets library](https://huggingface.co/datasets/ag_news), the [TorchText loader](https://pytorch.org/text/stable/datasets.html#AG_NEWS), and the [original CSV source](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html). This accessibility ensures ready integration with modern deep learning frameworks such as **PyTorch** and **TensorFlow**.

## Installation

This project requires Python 3.9+ and the following libraries:

- `torch` – for model training and inference with PyTorch
- `transformers` – for loading and fine‑tuning DeBERTa‑v3 and Longformer via Hugging Face
- `datasets` – for accessing and handling the AG News dataset efficiently
- `evaluate` – for computing evaluation metrics like accuracy, precision, recall, and F1
- `peft` – for Low‑Rank Adaptation (LoRA) fine‑tuning
- `numpy` – for numerical operations
- `pandas` – for data manipulation and I/O
- `scikit‑learn` – for classification reports, cross‑validation, and utility functions
- `tqdm` – for progress bars during training, evaluation, and inference
- `matplotlib` – for plotting (attention heatmaps, training curves)
- `shap` – for SHAP‑based explainability (force plots, summary plots)
- `joblib` – for serializing stacking models and fast I/O
- `gradio` – for launching an interactive web demo
- `jupyterlab` – for interactive notebook exploration

Install dependencies via pip:

```bash
pip install transformers datasets torch evaluate peft numpy pandas scikit‑learn tqdm matplotlib shap joblib gradio jupyterlab
```

Or install them all at once with:

```bash
pip install -r requirements.txt
```

Install dependencies via conda:

```bash
conda env create -f environment.yml
conda activate agnews-classification
```

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
├── ROADMAP.md
├── setup.py
├── pyproject.toml
├── Makefile
├── .env.example
├── .env.test
├── .gitignore
├── .dockerignore
├── .editorconfig
├── .pre-commit-config.yaml
├── commitlint.config.js
│
├── requirements/
│   ├── base.txt
│   ├── ml.txt
│   ├── research.txt
│   ├── prod.txt
│   ├── dev.txt
│   ├── data.txt
│   ├── llm.txt
│   ├── ui.txt
│   ├── docs.txt
│   ├── robustness.txt
│   ├── minimal.txt
│   └── all.txt
│
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
│
├── .husky/
│   ├── pre-commit
│   └── commit-msg
│
├── images/
│   └── pipeline.png
│
├── configs/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── constants.py
│   │
│   ├── environments/
│   │   ├── dev.yaml
│   │   ├── staging.yaml
│   │   └── prod.yaml
│   │
│   ├── features/
│   │   └── feature_flags.yaml
│   │
│   ├── secrets/
│   │   └── secrets.template.yaml
│   │
│   ├── models/
│   │   ├── single/
│   │   │   ├── deberta_v3_xlarge.yaml
│   │   │   ├── roberta_large.yaml
│   │   │   ├── xlnet_large.yaml
│   │   │   ├── electra_large.yaml
│   │   │   ├── longformer_large.yaml
│   │   │   ├── gpt2_large.yaml
│   │   │   └── t5_large.yaml
│   │   └── ensemble/
│   │       ├── voting_ensemble.yaml
│   │       ├── stacking_xgboost.yaml
│   │       ├── stacking_catboost.yaml
│   │       ├── blending_advanced.yaml
│   │       └── bayesian_ensemble.yaml
│   │
│   ├── training/
│   │   ├── standard/
│   │   │   ├── base_training.yaml
│   │   │   ├── mixed_precision.yaml
│   │   │   └── distributed.yaml
│   │   ├── advanced/
│   │   │   ├── curriculum_learning.yaml
│   │   │   ├── adversarial_training.yaml
│   │   │   ├── multitask_learning.yaml
│   │   │   ├── contrastive_learning.yaml
│   │   │   ├── knowledge_distillation.yaml
│   │   │   ├── meta_learning.yaml
│   │   │   ├── prompt_based_tuning.yaml
│   │   │   ├── instruction_tuning.yaml
│   │   │   ├── multi_stage_training.yaml
│   │   │   └── gpt4_distillation.yaml
│   │   └── efficient/
│   │       ├── lora_peft.yaml
│   │       ├── qlora.yaml
│   │       ├── adapter_fusion.yaml
│   │       ├── prefix_tuning.yaml
│   │       └── prompt_tuning.yaml
│   │
│   ├── data/
│   │   ├── preprocessing/
│   │   │   ├── standard.yaml
│   │   │   ├── advanced.yaml
│   │   │   └── domain_specific.yaml
│   │   ├── augmentation/
│   │   │   ├── basic_augment.yaml
│   │   │   ├── back_translation.yaml
│   │   │   ├── paraphrase_generation.yaml
│   │   │   ├── mixup_strategies.yaml
│   │   │   ├── adversarial_augment.yaml
│   │   │   └── contrast_sets.yaml
│   │   ├── selection/
│   │   │   ├── coreset_selection.yaml
│   │   │   ├── influence_functions.yaml
│   │   │   └── active_selection.yaml
│   │   └── external/
│   │       ├── news_corpus.yaml
│   │       ├── wikipedia.yaml
│   │       ├── domain_adaptive.yaml
│   │       └── gpt4_generated.yaml
│   │
│   └── experiments/
│       ├── baselines/
│       │   ├── classical_ml.yaml
│       │   └── transformer_baseline.yaml
│       ├── ablations/
│       │   ├── model_size.yaml
│       │   ├── data_amount.yaml
│       │   ├── augmentation_impact.yaml
│       │   └── ensemble_components.yaml
│       ├── sota_attempts/
│       │   ├── phase1_single_models.yaml
│       │   ├── phase2_ensemble.yaml
│       │   ├── phase3_dapt.yaml
│       │   ├── phase4_final_sota.yaml
│       │   └── phase5_bleeding_edge.yaml
│       └── reproducibility/
│           ├── seeds.yaml
│           └── hardware_specs.yaml
│
├── data/
│   ├── raw/
│   │   ├── ag_news/
│   │   └── .gitkeep
│   ├── processed/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   └── stratified_folds/
│   ├── augmented/
│   │   ├── back_translated/
│   │   ├── paraphrased/
│   │   ├── synthetic/
│   │   ├── mixup/
│   │   ├── contrast_sets/
│   │   └── gpt4_augmented/
│   ├── external/
│   │   ├── news_corpus/
│   │   │   ├── cnn_dailymail/
│   │   │   ├── reuters/
│   │   │   ├── bbc_news/
│   │   │   └── reddit_news/
│   │   ├── pretrain_data/
│   │   └── distillation_data/
│   │       ├── gpt4_annotations/
│   │       └── teacher_predictions/
│   ├── pseudo_labeled/
│   ├── selected_subsets/
│   └── cache/
│
├── src/
│   ├── __init__.py
|   ├── __version__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── factory.py
│   │   ├── types.py
│   │   └── exceptions.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── rest/
│   │   │   ├── endpoints.py
│   │   │   ├── middleware.py
│   │   │   └── validators.py
│   │   ├── grpc/
│   │   │   ├── services.py
│   │   │   └── protos/
│   │   └── graphql/
│   │       ├── schema.py
│   │       └── resolvers.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prediction_service.py
│   │   ├── training_service.py
│   │   ├── data_service.py
│   │   └── model_management.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets/
│   │   │   ├── ag_news.py
│   │   │   ├── external_news.py
│   │   │   ├── combined_dataset.py
│   │   │   └── prompted_dataset.py
│   │   ├── preprocessing/
│   │   │   ├── text_cleaner.py
│   │   │   ├── tokenization.py
│   │   │   ├── feature_extraction.py
│   │   │   ├── sliding_window.py
│   │   │   └── prompt_formatter.py
│   │   ├── augmentation/
│   │   │   ├── base_augmenter.py
│   │   │   ├── back_translation.py
│   │   │   ├── paraphrase.py
│   │   │   ├── token_replacement.py
│   │   │   ├── mixup.py
│   │   │   ├── cutmix.py
│   │   │   ├── adversarial.py
│   │   │   └── contrast_set_generator.py
│   │   ├── sampling/
│   │   │   ├── balanced_sampler.py
│   │   │   ├── curriculum_sampler.py
│   │   │   ├── active_learning.py
│   │   │   ├── uncertainty_sampling.py
│   │   │   └── coreset_sampler.py
│   │   ├── selection/
│   │   │   ├── __init__.py
│   │   │   ├── influence_function.py
│   │   │   ├── gradient_matching.py
│   │   │   ├── diversity_selection.py
│   │   │   └── quality_filtering.py
│   │   └── loaders/
│   │       ├── dataloader.py
│   │       ├── dynamic_batching.py
│   │       └── prefetch_loader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── base_model.py
│   │   │   ├── model_wrapper.py
│   │   │   └── pooling_strategies.py
│   │   ├── transformers/
│   │   │   ├── deberta/
│   │   │   │   ├── deberta_v3.py
│   │   │   │   ├── deberta_sliding.py
│   │   │   │   └── deberta_hierarchical.py
│   │   │   ├── roberta/
│   │   │   │   ├── roberta_enhanced.py
│   │   │   │   └── roberta_domain.py
│   │   │   ├── xlnet/
│   │   │   │   └── xlnet_classifier.py
│   │   │   ├── electra/
│   │   │   │   └── electra_discriminator.py
│   │   │   ├── longformer/
│   │   │   │   └── longformer_global.py
│   │   │   └── generative/
│   │   │       ├── gpt2_classifier.py
│   │   │       └── t5_classifier.py
│   │   ├── prompt_based/
│   │   │   ├── __init__.py
│   │   │   ├── prompt_model.py
│   │   │   ├── soft_prompt.py
│   │   │   ├── instruction_model.py
│   │   │   └── template_manager.py
│   │   ├── efficient/
│   │   │   ├── lora/
│   │   │   │   ├── lora_model.py
│   │   │   │   ├── lora_config.py
│   │   │   │   └── lora_layers.py
│   │   │   ├── adapters/
│   │   │   │   ├── adapter_model.py
│   │   │   │   └── adapter_fusion.py
│   │   │   ├── quantization/
│   │   │   │   ├── int8_quantization.py
│   │   │   │   └── dynamic_quantization.py
│   │   │   └── pruning/
│   │   │       └── magnitude_pruning.py
│   │   ├── ensemble/
│   │   │   ├── base_ensemble.py
│   │   │   ├── voting/
│   │   │   │   ├── soft_voting.py
│   │   │   │   ├── weighted_voting.py
│   │   │   │   └── rank_averaging.py
│   │   │   ├── stacking/
│   │   │   │   ├── stacking_classifier.py
│   │   │   │   ├── meta_learners.py
│   │   │   │   └── cross_validation_stacking.py
│   │   │   ├── blending/
│   │   │   │   ├── blending_ensemble.py
│   │   │   │   └── dynamic_blending.py
│   │   │   └── advanced/
│   │   │       ├── bayesian_ensemble.py
│   │   │       ├── snapshot_ensemble.py
│   │   │       └── multi_level_ensemble.py
│   │   └── heads/
│   │       ├── classification_head.py
│   │       ├── multitask_head.py
│   │       ├── hierarchical_head.py
│   │       ├── attention_head.py
│   │       └── prompt_head.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainers/
│   │   │   ├── base_trainer.py
│   │   │   ├── standard_trainer.py
│   │   │   ├── distributed_trainer.py
│   │   │   ├── apex_trainer.py
│   │   │   ├── prompt_trainer.py
│   │   │   ├── instruction_trainer.py
│   │   │   └── multi_stage_trainer.py
│   │   ├── strategies/
│   │   │   ├── curriculum/
│   │   │   │   ├── curriculum_learning.py
│   │   │   │   ├── self_paced.py
│   │   │   │   └── competence_based.py
│   │   │   ├── adversarial/
│   │   │   │   ├── fgm.py
│   │   │   │   ├── pgd.py
│   │   │   │   └── freelb.py
│   │   │   ├── regularization/
│   │   │   │   ├── r_drop.py
│   │   │   │   ├── mixout.py
│   │   │   │   └── spectral_norm.py
│   │   │   ├── distillation/
│   │   │   │   ├── knowledge_distill.py
│   │   │   │   ├── feature_distill.py
│   │   │   │   ├── self_distill.py
│   │   │   │   └── gpt4_distill.py
│   │   │   ├── meta/
│   │   │   │   ├── maml.py
│   │   │   │   └── reptile.py
│   │   │   ├── prompt_based/
│   │   │   │   ├── prompt_tuning.py
│   │   │   │   ├── prefix_tuning.py
│   │   │   │   ├── p_tuning.py
│   │   │   │   └── soft_prompt_tuning.py
│   │   │   └── multi_stage/
│   │   │       ├── stage_manager.py
│   │   │       ├── progressive_training.py
│   │   │       └── iterative_refinement.py
│   │   ├── objectives/
│   │   │   ├── losses/
│   │   │   │   ├── focal_loss.py
│   │   │   │   ├── label_smoothing.py
│   │   │   │   ├── contrastive_loss.py
│   │   │   │   ├── triplet_loss.py
│   │   │   │   ├── custom_ce_loss.py
│   │   │   │   └── instruction_loss.py
│   │   │   └── regularizers/
│   │   │       ├── l2_regularizer.py
│   │   │       └── gradient_penalty.py
│   │   ├── optimization/
│   │   │   ├── optimizers/
│   │   │   │   ├── adamw_custom.py
│   │   │   │   ├── lamb.py
│   │   │   │   ├── lookahead.py
│   │   │   │   └── sam.py
│   │   │   ├── schedulers/
│   │   │   │   ├── cosine_warmup.py
│   │   │   │   ├── polynomial_decay.py
│   │   │   │   └── cyclic_scheduler.py
│   │   │   └── gradient/
│   │   │       ├── gradient_accumulation.py
│   │   │       └── gradient_clipping.py
│   │   └── callbacks/
│   │       ├── early_stopping.py
│   │       ├── model_checkpoint.py
│   │       ├── tensorboard_logger.py
│   │       ├── wandb_logger.py
│   │       └── learning_rate_monitor.py
│   │
│   ├── domain_adaptation/
│   │   ├── __init__.py
│   │   ├── pretraining/
│   │   │   ├── mlm_pretrain.py
│   │   │   ├── news_corpus_builder.py
│   │   │   └── adaptive_pretrain.py
│   │   ├── fine_tuning/
│   │   │   ├── gradual_unfreezing.py
│   │   │   └── discriminative_lr.py
│   │   └── pseudo_labeling/
│   │       ├── confidence_based.py
│   │       ├── uncertainty_filter.py
│   │       └── self_training.py
│   │
│   ├── distillation/
│   │   ├── __init__.py
│   │   ├── gpt4_api/
│   │   │   ├── api_client.py
│   │   │   ├── prompt_builder.py
│   │   │   └── response_parser.py
│   │   ├── teacher_models/
│   │   │   ├── gpt4_teacher.py
│   │   │   ├── ensemble_teacher.py
│   │   │   └── multi_teacher.py
│   │   └── distillation_data/
│   │       ├── data_generator.py
│   │       └── quality_filter.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── classification_metrics.py
│   │   │   ├── ensemble_metrics.py
│   │   │   ├── robustness_metrics.py
│   │   │   ├── efficiency_metrics.py
│   │   │   ├── fairness_metrics.py
│   │   │   ├── environmental_impact.py
│   │   │   └── contrast_consistency.py
│   │   ├── analysis/
│   │   │   ├── error_analysis.py
│   │   │   ├── confusion_analysis.py
│   │   │   ├── class_wise_analysis.py
│   │   │   ├── failure_case_analysis.py
│   │   │   ├── dataset_shift_analysis.py
│   │   │   ├── bias_analysis.py
│   │   │   └── contrast_set_analysis.py
│   │   ├── interpretability/
│   │   │   ├── attention_analysis.py
│   │   │   ├── shap_interpreter.py
│   │   │   ├── lime_interpreter.py
│   │   │   ├── integrated_gradients.py
│   │   │   ├── layer_wise_relevance.py
│   │   │   ├── probing_classifier.py
│   │   │   └── prompt_analysis.py
│   │   ├── statistical/
│   │   │   ├── significance_tests.py
│   │   │   ├── bootstrap_confidence.py
│   │   │   ├── mcnemar_test.py
│   │   │   └── effect_size.py
│   │   └── visualization/
│   │       ├── performance_plots.py
│   │       ├── learning_curves.py
│   │       ├── attention_heatmaps.py
│   │       ├── embedding_visualizer.py
│   │       └── report_generator.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictors/
│   │   │   ├── single_predictor.py
│   │   │   ├── batch_predictor.py
│   │   │   ├── streaming_predictor.py
│   │   │   ├── ensemble_predictor.py
│   │   │   └── prompt_predictor.py
│   │   ├── optimization/
│   │   │   ├── onnx_converter.py
│   │   │   ├── tensorrt_optimizer.py
│   │   │   ├── quantization_optimizer.py
│   │   │   └── pruning_optimizer.py
│   │   ├── serving/
│   │   │   ├── model_server.py
│   │   │   ├── batch_server.py
│   │   │   └── load_balancer.py
│   │   └── post_processing/
│   │       ├── confidence_calibration.py
│   │       ├── threshold_optimization.py
│   │       └── output_formatter.py
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
│       └── prompt_utils.py
│
├── experiments/
│   ├── __init__.py
│   ├── experiment_runner.py
│   ├── hyperparameter_search/
│   │   ├── optuna_search.py
│   │   ├── ray_tune_search.py
│   │   ├── hyperband.py
│   │   └── bayesian_optimization.py
│   ├── benchmarks/
│   │   ├── speed_benchmark.py
│   │   ├── memory_benchmark.py
│   │   ├── accuracy_benchmark.py
│   │   └── robustness_benchmark.py
│   ├── baselines/
│   │   ├── classical/
│   │   │   ├── naive_bayes.py
│   │   │   ├── svm_baseline.py
│   │   │   ├── random_forest.py
│   │   │   └── logistic_regression.py
│   │   └── neural/
│   │       ├── lstm_baseline.py
│   │       ├── cnn_baseline.py
│   │       └── bert_vanilla.py
│   ├── ablation_studies/
│   │   ├── component_ablation.py
│   │   ├── data_ablation.py
│   │   ├── model_size_ablation.py
│   │   ├── feature_ablation.py
│   │   └── prompt_ablation.py
│   ├── sota_experiments/
│   │   ├── single_model_sota.py
│   │   ├── ensemble_sota.py
│   │   ├── full_pipeline_sota.py
│   │   ├── production_sota.py
│   │   ├── prompt_based_sota.py
│   │   └── gpt4_distilled_sota.py
│   └── results/
│       ├── experiment_tracker.py
│       ├── result_aggregator.py
│       └── leaderboard_generator.py
│
├── research/
│   ├── papers/
│   │   ├── references.bib
│   │   └── related_work/
│   ├── experiments_log/
│   │   ├── daily_logs/
│   │   └── experiment_notes.md
│   ├── hypotheses/
│   │   ├── current_hypotheses.md
│   │   └── validation_results/
│   └── findings/
│       ├── key_insights.md
│       └── failed_experiments/
│
├── monitoring/
│   ├── dashboards/
│   │   ├── grafana/
│   │   └── prometheus/
│   ├── alerts/
│   │   ├── alert_rules.yaml
│   │   └── notification_config.yaml
│   ├── metrics/
│   │   ├── custom_metrics.py
│   │   └── metric_collectors.py
│   └── logs_analysis/
│       ├── log_parser.py
│       └── anomaly_detector.py
│
├── security/
│   ├── api_auth/
│   │   ├── jwt_handler.py
│   │   └── api_keys.py
│   ├── data_privacy/
│   │   ├── pii_detector.py
│   │   └── data_masking.py
│   ├── model_security/
│   │   ├── adversarial_defense.py
│   │   └── model_encryption.py
│   └── audit_logs/
│       ├── audit_logger.py
│       └── compliance_reports/
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
│   ├── redis/
│   │   └── redis_config.yaml
│   ├── memcached/
│   │   └── memcached_config.yaml
│   └── local/
│       └── disk_cache.py
│
├── load_testing/
│   ├── scenarios/
│   │   ├── basic_load.yaml
│   │   └── stress_test.yaml
│   ├── scripts/
│   │   ├── locust_test.py
│   │   └── k6_test.js
│   └── reports/
│       └── performance_report_template.md
│
├── backup/
│   ├── strategies/
│   │   ├── incremental_backup.yaml
│   │   └── full_backup.yaml
│   ├── scripts/
│   │   ├── backup_runner.sh
│   │   └── restore_runner.sh
│   └── recovery/
│       ├── disaster_recovery_plan.md
│       └── recovery_procedures/
│
├── quality/
│   ├── test_plans/
│   │   ├── unit_test_plan.md
│   │   └── integration_test_plan.md
│   ├── test_cases/
│   │   ├── manual_tests/
│   │   └── automated_tests/
│   ├── bug_reports/
│   │   └── bug_template.md
│   └── coverage/
│       └── coverage_reports/
│
├── quickstart/
│   ├── README.md
│   ├── minimal_example.py
│   ├── train_simple.py
│   ├── evaluate_simple.py
│   ├── demo_app.py
│   ├── colab_notebook.ipynb
│   └── docker_quickstart/
│       ├── Dockerfile
│       └── docker-compose.yml
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
│   └── evaluation/
│       └── metric_template.py
│
├── notebooks/
│   ├── tutorials/
│   │   ├── 00_environment_setup.ipynb
│   │   ├── 01_data_loading_basics.ipynb
│   │   ├── 02_preprocessing_tutorial.ipynb
│   │   ├── 03_model_training_basics.ipynb
│   │   ├── 04_evaluation_tutorial.ipynb
│   │   ├── 05_prompt_engineering.ipynb
│   │   └── 06_instruction_tuning.ipynb
│   ├── exploratory/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_data_statistics.ipynb
│   │   ├── 03_label_distribution.ipynb
│   │   ├── 04_text_length_analysis.ipynb
│   │   ├── 05_vocabulary_analysis.ipynb
│   │   └── 06_contrast_set_exploration.ipynb
│   ├── experiments/
│   │   ├── 01_baseline_experiments.ipynb
│   │   ├── 02_single_model_experiments.ipynb
│   │   ├── 03_ensemble_experiments.ipynb
│   │   ├── 04_ablation_studies.ipynb
│   │   ├── 05_sota_reproduction.ipynb
│   │   ├── 06_prompt_experiments.ipynb
│   │   └── 07_distillation_experiments.ipynb
│   ├── analysis/
│   │   ├── 01_error_analysis.ipynb
│   │   ├── 02_model_interpretability.ipynb
│   │   ├── 03_attention_visualization.ipynb
│   │   ├── 04_embedding_analysis.ipynb
│   │   └── 05_failure_cases.ipynb
│   ├── deployment/
│   │   ├── 01_model_optimization.ipynb
│   │   ├── 02_inference_pipeline.ipynb
│   │   └── 03_api_testing.ipynb
│   └── platform_specific/
│       ├── colab/
│       │   ├── quick_start_colab.ipynb
│       │   ├── full_training_colab.ipynb
│       │   └── inference_demo_colab.ipynb
│       ├── kaggle/
│       │   └── kaggle_submission.ipynb
│       └── sagemaker/
│           └── sagemaker_training.ipynb
│
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 01_Home.py
│   │   ├── 02_Single_Prediction.py
│   │   ├── 03_Batch_Analysis.py
│   │   ├── 04_Model_Comparison.py
│   │   ├── 05_Interpretability.py
│   │   ├── 06_Performance_Dashboard.py
│   │   ├── 07_Real_Time_Demo.py
│   │   ├── 08_Model_Selection.py
│   │   ├── 09_Documentation.py
│   │   └── 10_Prompt_Testing.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── prediction_component.py
│   │   ├── visualization_component.py
│   │   ├── model_selector.py
│   │   ├── file_uploader.py
│   │   ├── result_display.py
│   │   ├── performance_monitor.py
│   │   └── prompt_builder.py
│   ├── utils/
│   │   ├── session_manager.py
│   │   ├── caching.py
│   │   ├── theming.py
│   │   └── helpers.py
│   └── assets/
│       ├── css/
│       │   └── custom.css
│       ├── js/
│       │   └── custom.js
│       └── images/
│           ├── logo.png
│           └── banner.png
│
├── scripts/
│   ├── setup/
│   │   ├── download_all_data.py
│   │   ├── setup_environment.sh
│   │   ├── install_cuda.sh
│   │   ├── setup_colab.sh
│   │   └── verify_installation.py
│   ├── data_preparation/
│   │   ├── prepare_ag_news.py
│   │   ├── prepare_external_data.py
│   │   ├── create_augmented_data.py
│   │   ├── generate_pseudo_labels.py
│   │   ├── create_data_splits.py
│   │   ├── generate_contrast_sets.py
│   │   ├── select_quality_data.py
│   │   └── prepare_instruction_data.py
│   ├── training/
│   │   ├── train_all_models.sh
│   │   ├── train_single_model.py
│   │   ├── train_ensemble.py
│   │   ├── distributed_training.py
│   │   ├── resume_training.py
│   │   ├── train_with_prompts.py
│   │   ├── instruction_tuning.py
│   │   ├── multi_stage_training.py
│   │   └── distill_from_gpt4.py
│   ├── domain_adaptation/
│   │   ├── pretrain_on_news.py
│   │   ├── download_news_corpus.py
│   │   └── run_dapt.sh
│   ├── evaluation/
│   │   ├── evaluate_all_models.py
│   │   ├── generate_reports.py
│   │   ├── create_leaderboard.py
│   │   ├── statistical_analysis.py
│   │   └── evaluate_contrast_sets.py
│   ├── optimization/
│   │   ├── hyperparameter_search.py
│   │   ├── architecture_search.py
│   │   ├── ensemble_optimization.py
│   │   └── prompt_optimization.py
│   └── deployment/
│       ├── export_models.py
│       ├── optimize_for_inference.py
│       ├── create_docker_image.sh
│       └── deploy_to_cloud.py
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
│       ├── gpt4_prompts.txt
│       └── explanation_prompts.txt
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── data/
│   │   │   ├── test_preprocessing.py
│   │   │   ├── test_augmentation.py
│   │   │   ├── test_dataloader.py
│   │   │   └── test_contrast_sets.py
│   │   ├── models/
│   │   │   ├── test_transformers.py
│   │   │   ├── test_ensemble.py
│   │   │   ├── test_efficient.py
│   │   │   └── test_prompt_models.py
│   │   ├── training/
│   │   │   ├── test_trainers.py
│   │   │   ├── test_strategies.py
│   │   │   ├── test_callbacks.py
│   │   │   └── test_multi_stage.py
│   │   └── utils/
│   │       └── test_utilities.py
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_ensemble_pipeline.py
│   │   ├── test_inference_pipeline.py
│   │   ├── test_api_endpoints.py
│   │   └── test_prompt_pipeline.py
│   ├── performance/
│   │   ├── test_model_speed.py
│   │   ├── test_memory_usage.py
│   │   └── test_accuracy_benchmarks.py
│   └── fixtures/
│       ├── sample_data.py
│       ├── mock_models.py
│       └── test_configs.py
│
├── outputs/
│   ├── models/
│   │   ├── checkpoints/
│   │   ├── pretrained/
│   │   ├── fine_tuned/
│   │   ├── ensembles/
│   │   ├── optimized/
│   │   ├── exported/
│   │   ├── prompted/
│   │   └── distilled/
│   ├── results/
│   │   ├── experiments/
│   │   ├── ablations/
│   │   ├── benchmarks/
│   │   └── reports/
│   ├── analysis/
│   │   ├── error_analysis/
│   │   ├── interpretability/
│   │   └── statistical/
│   ├── logs/
│   │   ├── training/
│   │   ├── tensorboard/
│   │   ├── wandb/
│   │   └── mlflow/
│   └── artifacts/
│       ├── figures/
│       ├── tables/
│       └── presentations/
│
├── docs/
│   ├── index.md
│   ├── getting_started/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── troubleshooting.md
│   ├── user_guide/
│   │   ├── data_preparation.md
│   │   ├── model_training.md
│   │   ├── evaluation.md
│   │   ├── deployment.md
│   │   ├── prompt_engineering.md
│   │   └── advanced_techniques.md
│   ├── developer_guide/
│   │   ├── architecture.md
│   │   ├── adding_models.md
│   │   ├── custom_datasets.md
│   │   └── contributing.md
│   ├── api_reference/
│   │   ├── data_api.md
│   │   ├── models_api.md
│   │   ├── training_api.md
│   │   └── evaluation_api.md
│   ├── tutorials/
│   │   ├── basic_usage.md
│   │   ├── advanced_features.md
│   │   └── best_practices.md
│   ├── architecture/
│   │   ├── decisions/
│   │   │   ├── 001-model-selection.md
│   │   │   └── 002-ensemble-strategy.md
│   │   ├── diagrams/
│   │   │   ├── system-overview.puml
│   │   │   └── data-flow.puml
│   │   └── patterns/
│   │       ├── factory-pattern.md
│   │       └── strategy-pattern.md
│   ├── operations/
│   │   ├── runbooks/
│   │   │   ├── deployment.md
│   │   │   └── troubleshooting.md
│   │   ├── sops/
│   │   │   ├── model-update.md
│   │   │   └── data-refresh.md
│   │   └── incidents/
│   │       └── incident-response.md
│   ├── case_studies/
│   │   ├── production-deployment.md
│   │   └── performance-optimization.md
│   └── _static/
│       └── custom.css
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── Dockerfile.gpu
│   │   ├── docker-compose.yml
│   │   └── .dockerignore
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   ├── configmap.yaml
│   │   └── hpa.yaml
│   ├── cloud/
│   │   ├── aws/
│   │   │   ├── sagemaker/
│   │   │   ├── lambda/
│   │   │   └── ecs/
│   │   ├── gcp/
│   │   │   ├── vertex-ai/
│   │   │   └── cloud-run/
│   │   └── azure/
│   │       └── ml-studio/
│   ├── edge/
│   │   ├── mobile/
│   │   │   ├── tflite/
│   │   │   └── coreml/
│   │   └── iot/
│   │       └── nvidia-jetson/
│   └── serverless/
│       ├── functions/
│       └── api-gateway/
│
├── benchmarks/
│   ├── accuracy/
│   │   ├── model_comparison.json
│   │   └── ensemble_results.json
│   ├── speed/
│   │   ├── inference_benchmarks.json
│   │   └── training_benchmarks.json
│   ├── efficiency/
│   │   ├── memory_usage.json
│   │   └── energy_consumption.json
│   └── robustness/
│       ├── adversarial_results.json
│       ├── ood_detection.json
│       └── contrast_set_results.json
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── cd.yml
│   │   ├── tests.yml
│   │   ├── docker-publish.yml
│   │   ├── documentation.yml
│   │   └── benchmarks.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
│
├── .vscode/
│   ├── settings.json
│   ├── launch.json
│   ├── tasks.json
│   └── extensions.json
│
├── ci/
│   ├── run_tests.sh
│   ├── run_benchmarks.sh
│   ├── build_docker.sh
│   └── deploy.sh
│
└── tools/
    ├── profiling/
    │   ├── memory_profiler.py
    │   └── speed_profiler.py
    ├── debugging/
    │   ├── model_debugger.py
    │   └── data_validator.py
    └── visualization/
        ├── training_monitor.py
        └── result_plotter.py
```

## Usage



## Evaluation Metrics

To assess the model’s performance on the AG News classification task, we evaluate it using standard classification metrics:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

### Evaluation Code

Use the following code snippet to compute metrics during training or evaluation:

```python
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=label_names, digits=4))
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted')
    }
```

**Note**: Replace label_names with your actual class labels, for example:

```python
label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
```

### Sample Results

| Class        | Precision | Recall | F1-Score   |
| ------------ | --------- | ------ | ---------- |
| World        | 0.94      | 0.93   | 0.94       |
| Sports       | 0.97      | 0.96   | 0.96       |
| Business     | 0.93      | 0.93   | 0.93       |
| Sci/Tech     | 0.93      | 0.94   | 0.93       |
| **Macro**    | 0.94      | 0.94   | 0.94       |
| **Weighted** | 0.94      | 0.94   | 0.94       |
| **Accuracy** |           |        | **0.9402** |

These scores indicate that the BERT-based model performs consistently well across all four categories in AG News.

