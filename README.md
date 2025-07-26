# AG News Text Classification

## Introduction

This project investigates the problem of **multi-class text classification** using the **AG News dataset**, a well-established benchmark dataset comprising English-language news articles categorized into four thematic classes: *World*, *Sports*, *Business*, and *Science/Technology*. The central objective is to design and evaluate a high-performance classification framework that leverages **state-of-the-art Transformer architectures**, incorporating strategies to address both the **limitations of input length** and the **efficiency of fine-tuning** in large-scale language models.

A fundamental limitation in Transformer-based models, such as BERT and its variants, lies in their constrained **maximum input sequence length** (typically 512 tokens), which poses significant challenges in classifying **long-form text** - a common characteristic in real-world documents. To circumvent this issue, the proposed architecture integrates a **Sliding Window mechanism** with **DeBERTa-v3**, enabling the model to process extended sequences through overlapping textual segments while maintaining global contextual coherence.

Concurrently, the use of **Longformer** - an architecture specifically engineered for extended attention spans (up to 4096 tokens) - facilitates direct encoding of long-range dependencies without segmentation. This dual-model approach enables robust contextual representation across both short and long textual inputs.

To optimize both computational efficiency and generalization capability, this work adopts **LoRA (Low-Rank Adaptation)**, a paradigm within **Parameter-Efficient Fine-Tuning (PEFT)**. By introducing trainable low-rank matrices into attention layers while freezing the bulk of pretrained parameters, LoRA significantly reduces the number of trainable parameters during fine-tuning, enabling **efficient adaptation on limited hardware resources** without compromising predictive performance.

Moreover, the framework integrates a **logit-level ensemble strategy**, aggregating the outputs of DeBERTa-v3 and Longformer via soft-voting. This ensemble approach seeks to synergize the localized precision of DeBERTa with the global modeling capacity of Longformer, resulting in a more robust and generalizable classifier.

In pursuit of greater transparency and accountability in model behavior, the project further incorporates **Error Analysis** and **Explainable AI (XAI)** methodologies. Post-hoc interpretability tools such as attention heatmaps and logit attribution are employed to analyze model predictions, diagnose failure cases, and guide iterative improvements through targeted data and architecture refinement.

In addition to the core architecture, further performance gains may be achieved through advanced extensions such as **domain-adaptive pretraining (DAPT)**, **confidence-based pseudo-labeling**, **stacking ensembles with meta-learners**, **k-fold cross-validation**, and **targeted data augmentation**. These strategies aim to **enhance robustness**, **reduce variance**, and **align model priors** more closely with the target domain.

**The pipeline encompasses the following components:**

- **Preprocessing**: Advanced tokenization, normalization, and window-based input segmentation.
- **Modeling**: Fine-tuning of `microsoft/deberta-v3-large` and `allenai/longformer-large-4096` with LoRA via the Hugging Face PEFT framework.
- **Ensembling**: Logit-level aggregation across models to enhance robustness and reduce variance.
- **Evaluation**: Comprehensive reporting of Accuracy, Precision, Recall, and F1-Score across all classes.
- **Analysis**: Qualitative and quantitative error investigation, along with model interpretability via XAI techniques.

By integrating recent advances in **transformer modeling**, **efficient fine-tuning**, and **model interpretability**, this project sets forth a replicable and scalable NLP pipeline. The framework not only surpasses classical baselines such as Naive Bayes and Support Vector Machines, but also provides a blueprint for future work in **long-form document** classification under constrained computational environments.

All components are developed using the Hugging Face `transformers`, `datasets`, `evaluate`, and `peft` libraries, ensuring **modularity**, **reproducibility**, and **applicability to a wide range of real-world classification tasks**.

## Model Architecture

![Pipeline Diagram](images/pipeline.png)

## Dataset

The **AG News dataset**, introduced by **Xiang Zhang, Junbo Zhao, and Yann LeCun in 2015**, is a well-established benchmark corpus for topic classification in natural language processing (NLP). It was curated as part of the **ComeToMyHead** academic project and consists of news articles collected from over 2,000 news sources over a period exceeding one year.

The dataset is organized into four high-level topical categories:

- **World**
- **Sports**
- **Business**
- **Science/Technology**

Each instance comprises a concise **title** and **description** of a news article, together forming the input text for classification. This design supports both short-form and long-form input handling, making it particularly well-suited for evaluating models such as **DeBERTa-v3**, **Longformer**, and those fine-tuned via **LoRA (Low-Rank Adaptation)** for efficient long-sequence modeling.

The dataset is **balanced across categories** and comes with a predefined split:

- **Training set**: 120,000 samples (30,000 per class)
- **Test set**: 7,600 samples (1,900 per class)

AG News captures various real-world challenges in text classification, including:

- **Semantic ambiguity across topic boundaries**
- **Domain overlap and concept drift**
- **Stylistic variation and differences in textual length**

The dataset is publicly available via the [Hugging Face Datasets library](https://huggingface.co/datasets/ag_news), [TorchText loader](https://pytorch.org/text/stable/datasets.html#AG_NEWS), and the [original CSV source](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), making it readily integrable with **PyTorch** and **TensorFlow** pipelines.

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
├── setup.py
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
│
├── images/
│   └── pipeline.png
│
├── configs/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── constants.py
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
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── factory.py
│   │   ├── types.py
│   │   └── exceptions.py
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
│   │   │   └── contrast_consistency.py
│   │   ├── analysis/
│   │   │   ├── error_analysis.py
│   │   │   ├── confusion_analysis.py
│   │   │   ├── class_wise_analysis.py
│   │   │   ├── failure_case_analysis.py
│   │   │   ├── dataset_shift_analysis.py
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

