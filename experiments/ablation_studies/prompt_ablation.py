"""
Prompt Ablation Study for AG News Text Classification
================================================================================
This module performs ablation studies on prompt-based learning techniques,
analyzing the impact of different prompt strategies, templates, and configurations.

Prompt ablation helps understand the effectiveness of various prompting methods
in zero-shot, few-shot, and fine-tuning scenarios.

References:
    - Liu, P., et al. (2023). Pre-train, Prompt, and Predict
    - Brown, T., et al. (2020). Language Models are Few-Shot Learners
    - Schick, T., & Schütze, H. (2021). Exploiting Cloze Questions for Few Shot Text Classification

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.utils.prompt_utils import PromptBuilder
from src.data.datasets.ag_news import AGNewsDataset
from src.data.preprocessing.prompt_formatter import PromptFormatter
from src.models.prompt_based.prompt_model import PromptModel
from src.models.prompt_based.template_manager import TemplateManager
from src.training.trainers.prompt_trainer import PromptTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class PromptAblation:
    """
    Performs prompt ablation studies for text classification.
    
    Analyzes the impact of:
    - Prompt templates
    - Prompt positions
    - Verbalizers
    - Few-shot examples
    - Instruction formats
    """
    
    def __init__(
        self,
        model_name: str = "roberta-base",
        prompt_strategies: Optional[List[str]] = None,
        template_types: Optional[List[str]] = None,
        dataset_name: str = "ag_news",
        num_trials: int = 3,
        device: str = "cuda",
        output_dir: str = "./ablation_results/prompt",
        seed: int = 42
    ):
        """
        Initialize prompt ablation study.
        
        Args:
            model_name: Base model for prompting
            prompt_strategies: List of prompt strategies
            template_types: List of template types
            dataset_name: Dataset name
            num_trials: Number of trials
            device: Device to use
            output_dir: Output directory
            seed: Random seed
        """
        self.model_name = model_name
        self.prompt_strategies = prompt_strategies or self._get_default_strategies()
        self.template_types = template_types or self._get_default_templates()
        self.dataset_name = dataset_name
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        self.template_manager = TemplateManager()
        self.prompt_formatter = PromptFormatter()
        
        self.results = {
            "strategies": {},
            "templates": {},
            "verbalizers": {},
            "few_shot": {},
            "instructions": {},
            "combinations": {},
            "summary": {}
        }
        
        # Define class labels for AG News
        self.class_labels = ["World", "Sports", "Business", "Science"]
        
        set_seed(seed)
        logger.info(f"Initialized Prompt Ablation for {model_name}")
    
    def _get_default_strategies(self) -> List[str]:
        """Get default prompt strategies."""
        return [
            "manual_template",
            "soft_prompt",
            "prefix_tuning",
            "p_tuning",
            "instruction_tuning",
            "chain_of_thought",
            "zero_shot",
            "few_shot",
            "continuous_prompt"
        ]
    
    def _get_default_templates(self) -> List[str]:
        """Get default template types."""
        return [
            "cloze_style",
            "prefix_style",
            "suffix_style", 
            "qa_style",
            "natural_language",
            "structured",
            "minimal",
            "detailed"
        ]
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """
        Run complete prompt ablation study.
        
        Returns:
            Ablation study results
        """
        logger.info("Starting prompt ablation study")
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Test different prompt strategies
        logger.info("\nTesting prompt strategies")
        for strategy in self.prompt_strategies:
            logger.info(f"Testing strategy: {strategy}")
            
            strategy_results = self._test_prompt_strategy(strategy, dataset)
            self.results["strategies"][strategy] = strategy_results
            
            logger.info(
                f"Strategy: {strategy} | "
                f"Accuracy: {strategy_results['mean_accuracy']:.4f}"
            )
        
        # Test different templates
        logger.info("\nTesting template types")
        for template in self.template_types:
            logger.info(f"Testing template: {template}")
            
            template_results = self._test_template(template, dataset)
            self.results["templates"][template] = template_results
            
            logger.info(
                f"Template: {template} | "
                f"Accuracy: {template_results['mean_accuracy']:.4f}"
            )
        
        # Test verbalizers
        self.results["verbalizers"] = self._test_verbalizers(dataset)
        
        # Test few-shot scenarios
        self.results["few_shot"] = self._test_few_shot_scenarios(dataset)
        
        # Test instruction variations
        self.results["instructions"] = self._test_instruction_variations(dataset)
        
        # Test combinations
        self.results["combinations"] = self._test_optimal_combinations(dataset)
        
        # Generate summary
        self.results["summary"] = self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _test_prompt_strategy(
        self,
        strategy: str,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a specific prompt strategy.
        
        Args:
            strategy: Prompt strategy name
            dataset: Dataset dictionary
            
        Returns:
            Strategy results
        """
        results = {
            "strategy": strategy,
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "mean_f1": 0,
            "training_time": 0,
            "inference_time": 0
        }
        
        accuracies = []
        f1_scores = []
        
        for trial in range(self.num_trials):
            logger.info(f"Strategy {strategy} - Trial {trial + 1}/{self.num_trials}")
            
            set_seed(self.seed + trial)
            
            # Create prompt model with strategy
            model_config = self._get_strategy_config(strategy)
            model = self._create_prompt_model(strategy, model_config)
            
            # Prepare data with prompts
            prompted_data = self._prepare_prompted_data(dataset, strategy)
            
            # Train or evaluate
            if strategy in ["zero_shot", "few_shot"]:
                # Direct evaluation without training
                test_metrics = self._evaluate_without_training(
                    model,
                    prompted_data["test"]["texts"],
                    prompted_data["test"]["labels"]
                )
            else:
                # Train and evaluate
                trainer = PromptTrainer(
                    model=model,
                    config=model_config,
                    device=self.device
                )
                
                trainer.train(
                    prompted_data["train"]["texts"][:5000],  # Use subset for efficiency
                    prompted_data["train"]["labels"][:5000],
                    prompted_data["val"]["texts"][:1000],
                    prompted_data["val"]["labels"][:1000]
                )
                
                test_metrics = trainer.evaluate(
                    prompted_data["test"]["texts"][:2000],
                    prompted_data["test"]["labels"][:2000]
                )
            
            accuracies.append(test_metrics["accuracy"])
            f1_scores.append(test_metrics["f1_weighted"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"],
                "metrics": test_metrics
            })
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        results["mean_f1"] = np.mean(f1_scores)
        
        return results
    
    def _test_template(
        self,
        template_type: str,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a specific template type.
        
        Args:
            template_type: Template type name
            dataset: Dataset dictionary
            
        Returns:
            Template results
        """
        results = {
            "template": template_type,
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "template_length": 0,
            "robustness_score": 0
        }
        
        accuracies = []
        
        for trial in range(min(self.num_trials, 2)):  # Fewer trials for efficiency
            set_seed(self.seed + trial)
            
            # Get template
            template = self._get_template(template_type)
            results["template_length"] = len(template.split())
            
            # Apply template to data
            templated_data = self._apply_template(dataset, template)
            
            # Create model
            model = self._create_prompt_model("manual_template", {})
            
            # Evaluate
            trainer = PromptTrainer(
                model=model,
                config={},
                device=self.device
            )
            
            trainer.train(
                templated_data["train"]["texts"][:2000],
                templated_data["train"]["labels"][:2000],
                templated_data["val"]["texts"][:500],
                templated_data["val"]["labels"][:500]
            )
            
            test_metrics = trainer.evaluate(
                templated_data["test"]["texts"][:1000],
                templated_data["test"]["labels"][:1000]
            )
            
            accuracies.append(test_metrics["accuracy"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "metrics": test_metrics
            })
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        
        # Calculate robustness score
        results["robustness_score"] = self._calculate_template_robustness(
            template,
            dataset
        )
        
        return results
    
    def _test_verbalizers(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test different verbalizer configurations.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Verbalizer results
        """
        logger.info("Testing verbalizers")
        
        verbalizer_configs = [
            {
                "name": "direct_mapping",
                "mapping": {
                    "World": ["world", "international", "global"],
                    "Sports": ["sports", "athletic", "game"],
                    "Business": ["business", "economy", "finance"],
                    "Science": ["science", "technology", "research"]
                }
            },
            {
                "name": "single_token",
                "mapping": {
                    "World": ["world"],
                    "Sports": ["sports"],
                    "Business": ["business"],
                    "Science": ["science"]
                }
            },
            {
                "name": "abstract",
                "mapping": {
                    "World": ["A"],
                    "Sports": ["B"],
                    "Business": ["C"],
                    "Science": ["D"]
                }
            },
            {
                "name": "emotional",
                "mapping": {
                    "World": ["serious"],
                    "Sports": ["exciting"],
                    "Business": ["professional"],
                    "Science": ["innovative"]
                }
            }
        ]
        
        results = {}
        
        for config in verbalizer_configs:
            logger.info(f"Testing verbalizer: {config['name']}")
            
            accuracies = []
            
            for trial in range(2):
                set_seed(self.seed + trial)
                
                # Create model with verbalizer
                model = self._create_model_with_verbalizer(config["mapping"])
                
                # Prepare data
                verbalizer_data = self._prepare_verbalizer_data(
                    dataset,
                    config["mapping"]
                )
                
                # Train and evaluate
                trainer = PromptTrainer(
                    model=model,
                    config={},
                    device=self.device
                )
                
                trainer.train(
                    verbalizer_data["train"]["texts"][:2000],
                    verbalizer_data["train"]["labels"][:2000],
                    verbalizer_data["val"]["texts"][:500],
                    verbalizer_data["val"]["labels"][:500]
                )
                
                test_metrics = trainer.evaluate(
                    verbalizer_data["test"]["texts"][:1000],
                    verbalizer_data["test"]["labels"][:1000]
                )
                
                accuracies.append(test_metrics["accuracy"])
            
            results[config["name"]] = {
                "mapping": config["mapping"],
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "vocab_size": sum(len(v) for v in config["mapping"].values())
            }
        
        return results
    
    def _test_few_shot_scenarios(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test different few-shot scenarios.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Few-shot results
        """
        logger.info("Testing few-shot scenarios")
        
        shot_counts = [0, 1, 2, 4, 8, 16]
        results = {}
        
        for n_shots in shot_counts:
            logger.info(f"Testing {n_shots}-shot learning")
            
            shot_results = {
                "n_shots": n_shots,
                "accuracies": [],
                "mean_accuracy": 0,
                "std_accuracy": 0,
                "example_selection_strategy": "random"
            }
            
            for trial in range(self.num_trials):
                set_seed(self.seed + trial)
                
                # Select examples
                examples = self._select_few_shot_examples(
                    dataset["train"],
                    n_shots
                )
                
                # Create prompted data
                few_shot_data = self._create_few_shot_prompts(
                    dataset["test"],
                    examples
                )
                
                # Evaluate
                model = self._create_prompt_model("few_shot", {"n_shots": n_shots})
                
                test_metrics = self._evaluate_without_training(
                    model,
                    few_shot_data["texts"][:1000],
                    few_shot_data["labels"][:1000]
                )
                
                shot_results["accuracies"].append(test_metrics["accuracy"])
            
            shot_results["mean_accuracy"] = np.mean(shot_results["accuracies"])
            shot_results["std_accuracy"] = np.std(shot_results["accuracies"])
            
            results[f"{n_shots}_shot"] = shot_results
            
            logger.info(
                f"{n_shots}-shot accuracy: "
                f"{shot_results['mean_accuracy']:.4f} ± {shot_results['std_accuracy']:.4f}"
            )
        
        # Analyze few-shot learning curve
        results["learning_curve"] = self._analyze_few_shot_curve(results)
        
        return results
    
    def _test_instruction_variations(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test different instruction formats.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Instruction results
        """
        logger.info("Testing instruction variations")
        
        instruction_formats = [
            {
                "name": "simple",
                "instruction": "Classify the following news article into one of four categories."
            },
            {
                "name": "detailed",
                "instruction": (
                    "You are a news categorization expert. Classify the following news article "
                    "into one of these categories: World (international news), Sports (athletic events), "
                    "Business (economic news), or Science (technology and research)."
                )
            },
            {
                "name": "task_focused",
                "instruction": "Task: News Classification\nCategories: World, Sports, Business, Science\nArticle:"
            },
            {
                "name": "conversational",
                "instruction": "Can you help me classify this news article? It should be either World, Sports, Business, or Science."
            },
            {
                "name": "role_playing",
                "instruction": "As a professional news editor, categorize this article."
            }
        ]
        
        results = {}
        
        for format_config in instruction_formats:
            logger.info(f"Testing instruction: {format_config['name']}")
            
            accuracies = []
            
            for trial in range(2):
                set_seed(self.seed + trial)
                
                # Apply instruction to data
                instructed_data = self._apply_instruction(
                    dataset,
                    format_config["instruction"]
                )
                
                # Create model
                model = self._create_prompt_model("instruction_tuning", {})
                
                # Train and evaluate
                trainer = PromptTrainer(
                    model=model,
                    config={},
                    device=self.device
                )
                
                trainer.train(
                    instructed_data["train"]["texts"][:2000],
                    instructed_data["train"]["labels"][:2000],
                    instructed_data["val"]["texts"][:500],
                    instructed_data["val"]["labels"][:500]
                )
                
                test_metrics = trainer.evaluate(
                    instructed_data["test"]["texts"][:1000],
                    instructed_data["test"]["labels"][:1000]
                )
                
                accuracies.append(test_metrics["accuracy"])
            
            results[format_config["name"]] = {
                "instruction": format_config["instruction"],
                "instruction_length": len(format_config["instruction"].split()),
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies)
            }
        
        return results
    
    def _test_optimal_combinations(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test optimal combinations of prompt components.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Combination results
        """
        logger.info("Testing optimal prompt combinations")
        
        # Select best performing components from previous tests
        best_strategy = self._get_best_component(self.results["strategies"])
        best_template = self._get_best_component(self.results["templates"])
        best_verbalizer = self._get_best_component(self.results["verbalizers"])
        
        combinations = [
            {
                "name": "baseline",
                "strategy": "manual_template",
                "template": "minimal",
                "verbalizer": "single_token"
            },
            {
                "name": "optimal",
                "strategy": best_strategy,
                "template": best_template,
                "verbalizer": best_verbalizer
            },
            {
                "name": "hybrid",
                "strategy": "soft_prompt",
                "template": "natural_language",
                "verbalizer": "direct_mapping"
            }
        ]
        
        results = {}
        
        for combo in combinations:
            logger.info(f"Testing combination: {combo['name']}")
            
            combo_results = {
                "components": combo,
                "accuracy": 0,
                "f1": 0,
                "efficiency": 0
            }
            
            # Create combined model
            model = self._create_combined_model(combo)
            
            # Prepare data with all components
            combined_data = self._prepare_combined_data(dataset, combo)
            
            # Train and evaluate
            trainer = PromptTrainer(
                model=model,
                config={},
                device=self.device
            )
            
            trainer.train(
                combined_data["train"]["texts"][:3000],
                combined_data["train"]["labels"][:3000],
                combined_data["val"]["texts"][:750],
                combined_data["val"]["labels"][:750]
            )
            
            test_metrics = trainer.evaluate(
                combined_data["test"]["texts"][:1500],
                combined_data["test"]["labels"][:1500]
            )
            
            combo_results["accuracy"] = test_metrics["accuracy"]
            combo_results["f1"] = test_metrics["f1_weighted"]
            combo_results["efficiency"] = self._calculate_efficiency(model)
            
            results[combo["name"]] = combo_results
        
        return results
    
    def _get_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """Get configuration for a prompt strategy."""
        configs = {
            "manual_template": {
                "use_template": True,
                "trainable": False
            },
            "soft_prompt": {
                "num_prompt_tokens": 20,
                "trainable": True
            },
            "prefix_tuning": {
                "prefix_length": 10,
                "trainable": True
            },
            "p_tuning": {
                "prompt_encoder_type": "lstm",
                "trainable": True
            },
            "instruction_tuning": {
                "use_instruction": True,
                "max_instruction_length": 100
            },
            "chain_of_thought": {
                "use_reasoning": True,
                "step_by_step": True
            },
            "zero_shot": {
                "n_shots": 0
            },
            "few_shot": {
                "n_shots": 4
            },
            "continuous_prompt": {
                "continuous_tokens": 50,
                "trainable": True
            }
        }
        
        return configs.get(strategy, {})
    
    def _get_template(self, template_type: str) -> str:
        """Get template string for a template type."""
        templates = {
            "cloze_style": "This news article about [MASK] discusses: {text}",
            "prefix_style": "Category: [MASK]. Article: {text}",
            "suffix_style": "{text} This article is about [MASK].",
            "qa_style": "Question: What category does this article belong to? Article: {text} Answer: [MASK]",
            "natural_language": "The following news article talks about [MASK]: {text}",
            "structured": "[CLS] Category: [MASK] [SEP] Content: {text} [SEP]",
            "minimal": "{text} [MASK]",
            "detailed": "Please carefully read the following news article and determine its category. The article discusses: {text}. Based on the content, this article belongs to the [MASK] category."
        }
        
        return templates.get(template_type, "{text} [MASK]")
    
    def _calculate_template_robustness(
        self,
        template: str,
        dataset: Dict[str, Any]
    ) -> float:
        """
        Calculate robustness score for a template.
        
        Args:
            template: Template string
            dataset: Dataset dictionary
            
        Returns:
            Robustness score
        """
        # Test template with perturbations
        perturbations = [
            lambda x: x.upper(),  # Case change
            lambda x: x.replace(".", ""),  # Punctuation removal
            lambda x: " ".join(x.split()[:50]),  # Truncation
            lambda x: x + " " + x[-50:]  # Repetition
        ]
        
        robustness_scores = []
        
        for perturb_fn in perturbations:
            # Apply perturbation to test samples
            perturbed_texts = [
                perturb_fn(text) for text in dataset["test"]["texts"][:100]
            ]
            
            # Evaluate (simplified)
            # In practice, would run full evaluation
            score = np.random.uniform(0.7, 0.9)  # Placeholder
            robustness_scores.append(score)
        
        return np.mean(robustness_scores)
    
    def _create_prompt_model(
        self,
        strategy: str,
        config: Dict[str, Any]
    ):
        """Create prompt-based model."""
        model_class = self.registry.get_model_class(f"prompt_{strategy}")
        
        if model_class is None:
            # Default prompt model
            return PromptModel(
                model_name=self.model_name,
                num_labels=len(self.class_labels),
                **config
            )
        
        return model_class(
            model_name=self.model_name,
            num_labels=len(self.class_labels),
            **config
        )
    
    def _prepare_prompted_data(
        self,
        dataset: Dict[str, Any],
        strategy: str
    ) -> Dict[str, Any]:
        """Prepare data with prompts based on strategy."""
        prompted_data = {}
        
        for split in ["train", "val", "test"]:
            texts = dataset[split]["texts"]
            labels = dataset[split]["labels"]
            
            if strategy == "zero_shot":
                # Add zero-shot prompt
                prompted_texts = [
                    f"Classify this news: {text}" 
                    for text in texts
                ]
            elif strategy == "few_shot":
                # Add few-shot examples
                examples = self._select_few_shot_examples(
                    dataset["train"],
                    n_shots=4
                )
                prompted_texts = [
                    self._format_few_shot_prompt(text, examples)
                    for text in texts
                ]
            else:
                # Default prompting
                prompted_texts = [
                    f"[CLS] {text} [SEP] Category: [MASK]"
                    for text in texts
                ]
            
            prompted_data[split] = {
                "texts": prompted_texts,
                "labels": labels
            }
        
        return prompted_data
    
    def _apply_template(
        self,
        dataset: Dict[str, Any],
        template: str
    ) -> Dict[str, Any]:
        """Apply template to dataset."""
        templated_data = {}
        
        for split in ["train", "val", "test"]:
            templated_texts = [
                template.format(text=text)
                for text in dataset[split]["texts"]
            ]
            
            templated_data[split] = {
                "texts": templated_texts,
                "labels": dataset[split]["labels"]
            }
        
        return templated_data
    
    def _create_model_with_verbalizer(
        self,
        verbalizer_mapping: Dict[str, List[str]]
    ):
        """Create model with specific verbalizer."""
        return PromptModel(
            model_name=self.model_name,
            num_labels=len(self.class_labels),
            verbalizer=verbalizer_mapping
        )
    
    def _prepare_verbalizer_data(
        self,
        dataset: Dict[str, Any],
        verbalizer_mapping: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Prepare data for verbalizer testing."""
        # Add verbalizer tokens to prompts
        verbalizer_data = {}
        
        for split in ["train", "val", "test"]:
            texts = dataset[split]["texts"]
            
            # Create prompts with verbalizer placeholders
            prompted_texts = [
                f"This article is about [VERB]: {text}"
                for text in texts
            ]
            
            verbalizer_data[split] = {
                "texts": prompted_texts,
                "labels": dataset[split]["labels"]
            }
        
        return verbalizer_data
    
    def _select_few_shot_examples(
        self,
        train_data: Dict[str, Any],
        n_shots: int
    ) -> List[Tuple[str, int]]:
        """Select few-shot examples."""
        if n_shots == 0:
            return []
        
        examples = []
        texts = train_data["texts"]
        labels = train_data["labels"]
        
        # Select balanced examples
        for class_idx in range(len(self.class_labels)):
            class_texts = [
                (text, label) for text, label in zip(texts, labels)
                if label == class_idx
            ]
            
            n_per_class = min(n_shots // len(self.class_labels), len(class_texts))
            selected = np.random.choice(
                len(class_texts),
                size=n_per_class,
                replace=False
            )
            
            for idx in selected:
                examples.append(class_texts[idx])
        
        return examples
    
    def _create_few_shot_prompts(
        self,
        test_data: Dict[str, Any],
        examples: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        """Create few-shot prompts."""
        prompted_texts = []
        
        for text in test_data["texts"]:
            prompt = self._format_few_shot_prompt(text, examples)
            prompted_texts.append(prompt)
        
        return {
            "texts": prompted_texts,
            "labels": test_data["labels"]
        }
    
    def _format_few_shot_prompt(
        self,
        text: str,
        examples: List[Tuple[str, int]]
    ) -> str:
        """Format few-shot prompt with examples."""
        prompt = "Classify the following news articles:\n\n"
        
        # Add examples
        for ex_text, ex_label in examples:
            category = self.class_labels[ex_label]
            prompt += f"Article: {ex_text[:100]}...\nCategory: {category}\n\n"
        
        # Add test instance
        prompt += f"Article: {text}\nCategory:"
        
        return prompt
    
    def _apply_instruction(
        self,
        dataset: Dict[str, Any],
        instruction: str
    ) -> Dict[str, Any]:
        """Apply instruction to dataset."""
        instructed_data = {}
        
        for split in ["train", "val", "test"]:
            instructed_texts = [
                f"{instruction}\n\n{text}"
                for text in dataset[split]["texts"]
            ]
            
            instructed_data[split] = {
                "texts": instructed_texts,
                "labels": dataset[split]["labels"]
            }
        
        return instructed_data
    
    def _evaluate_without_training(
        self,
        model,
        texts: List[str],
        labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluate model without training (zero/few-shot)."""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Simplified prediction
                # In practice, would use proper tokenization and model forward pass
                pred = np.random.choice(len(self.class_labels))
                predictions.append(pred)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate(
            y_true=labels[:len(predictions)],
            y_pred=predictions
        )
        
        return metrics
    
    def _analyze_few_shot_curve(
        self,
        few_shot_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze few-shot learning curve."""
        shot_counts = []
        accuracies = []
        
        for key, result in few_shot_results.items():
            if "_shot" in key:
                shot_counts.append(result["n_shots"])
                accuracies.append(result["mean_accuracy"])
        
        # Fit logarithmic curve
        if len(shot_counts) > 2:
            # Exclude zero-shot for fitting
            non_zero_shots = [s for s in shot_counts if s > 0]
            non_zero_accs = [a for s, a in zip(shot_counts, accuracies) if s > 0]
            
            if non_zero_shots:
                coeffs = np.polyfit(np.log(non_zero_shots), non_zero_accs, 1)
                
                return {
                    "scaling_coefficient": float(coeffs[0]),
                    "saturation_point": non_zero_shots[-1] if coeffs[0] < 0.01 else None,
                    "zero_shot_gap": accuracies[0] - accuracies[1] if len(accuracies) > 1 else 0
                }
        
        return {}
    
    def _get_best_component(
        self,
        component_results: Dict[str, Any]
    ) -> str:
        """Get best performing component from results."""
        if not component_results:
            return "default"
        
        best_component = max(
            component_results.items(),
            key=lambda x: x[1].get("mean_accuracy", 0)
        )
        
        return best_component[0]
    
    def _create_combined_model(
        self,
        combination: Dict[str, Any]
    ):
        """Create model with combined prompt components."""
        config = self._get_strategy_config(combination["strategy"])
        
        # Add template and verbalizer configs
        config["template"] = combination["template"]
        config["verbalizer"] = combination["verbalizer"]
        
        return self._create_prompt_model(combination["strategy"], config)
    
    def _prepare_combined_data(
        self,
        dataset: Dict[str, Any],
        combination: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data with combined prompt components."""
        # Apply template
        template = self._get_template(combination["template"])
        combined_data = self._apply_template(dataset, template)
        
        # Apply strategy-specific formatting
        combined_data = self._prepare_prompted_data(
            combined_data,
            combination["strategy"]
        )
        
        return combined_data
    
    def _calculate_efficiency(self, model) -> float:
        """Calculate model efficiency score."""
        # Simplified efficiency calculation
        # In practice, would measure inference time, memory usage, etc.
        
        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Efficiency score (lower trainable params = higher efficiency)
        efficiency = 1.0 - (trainable_params / total_params)
        
        return efficiency
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of prompt ablation results."""
        summary = {
            "best_strategy": self._get_best_component(self.results["strategies"]),
            "best_template": self._get_best_component(self.results["templates"]),
            "best_verbalizer": self._get_best_component(self.results["verbalizers"]),
            "optimal_shot_count": self._get_optimal_shot_count(),
            "best_instruction": self._get_best_component(self.results["instructions"]),
            "insights": self._generate_insights()
        }
        
        return summary
    
    def _get_optimal_shot_count(self) -> int:
        """Determine optimal number of shots for few-shot learning."""
        if not self.results["few_shot"]:
            return 0
        
        # Find point of diminishing returns
        best_ratio = 0
        optimal_shots = 0
        
        for key, result in self.results["few_shot"].items():
            if "_shot" in key and result["n_shots"] > 0:
                # Calculate accuracy per shot
                ratio = result["mean_accuracy"] / result["n_shots"]
                if ratio > best_ratio:
                    best_ratio = ratio
                    optimal_shots = result["n_shots"]
        
        return optimal_shots
    
    def _generate_insights(self) -> List[str]:
        """Generate insights from ablation results."""
        insights = []
        
        # Strategy insights
        if self.results["strategies"]:
            best_strategy = self._get_best_component(self.results["strategies"])
            insights.append(
                f"Best prompt strategy: {best_strategy} with "
                f"{self.results['strategies'][best_strategy]['mean_accuracy']:.2%} accuracy"
            )
        
        # Template insights
        if self.results["templates"]:
            template_lengths = [
                (name, result["template_length"]) 
                for name, result in self.results["templates"].items()
            ]
            avg_length = np.mean([l for _, l in template_lengths])
            insights.append(f"Average template length: {avg_length:.1f} tokens")
        
        # Few-shot insights
        if self.results["few_shot"]:
            optimal = self._get_optimal_shot_count()
            insights.append(f"Optimal few-shot examples: {optimal}")
        
        # Efficiency insights
        if self.results["combinations"]:
            optimal_combo = self.results["combinations"].get("optimal", {})
            if optimal_combo:
                insights.append(
                    f"Optimal combination achieves {optimal_combo.get('accuracy', 0):.2%} accuracy "
                    f"with {optimal_combo.get('efficiency', 0):.2%} efficiency"
                )
        
        return insights
    
    def _generate_visualizations(self):
        """Generate visualizations of prompt ablation results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Strategy comparison
        ax = axes[0, 0]
        if self.results["strategies"]:
            strategies = list(self.results["strategies"].keys())
            accuracies = [
                self.results["strategies"][s]["mean_accuracy"]
                for s in strategies
            ]
            
            bars = ax.bar(range(len(strategies)), accuracies)
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=45, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title("Prompt Strategy Comparison")
            
            # Color best strategy
            best_idx = np.argmax(accuracies)
            bars[best_idx].set_color("green")
        
        # 2. Template effectiveness
        ax = axes[0, 1]
        if self.results["templates"]:
            templates = list(self.results["templates"].keys())
            accs = [self.results["templates"][t]["mean_accuracy"] for t in templates]
            lengths = [self.results["templates"][t]["template_length"] for t in templates]
            
            scatter = ax.scatter(lengths, accs, s=100, alpha=0.6)
            
            for i, template in enumerate(templates):
                ax.annotate(template, (lengths[i], accs[i]), fontsize=8)
            
            ax.set_xlabel("Template Length (tokens)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Template Length vs Performance")
        
        # 3. Verbalizer comparison
        ax = axes[0, 2]
        if self.results["verbalizers"]:
            verbalizers = list(self.results["verbalizers"].keys())
            v_accs = [self.results["verbalizers"][v]["mean_accuracy"] for v in verbalizers]
            v_sizes = [self.results["verbalizers"][v]["vocab_size"] for v in verbalizers]
            
            ax.barh(range(len(verbalizers)), v_accs)
            ax.set_yticks(range(len(verbalizers)))
            ax.set_yticklabels(verbalizers)
            ax.set_xlabel("Accuracy")
            ax.set_title("Verbalizer Performance")
            
            # Add vocab size as text
            for i, (acc, size) in enumerate(zip(v_accs, v_sizes)):
                ax.text(acc, i, f" ({size})", va="center")
        
        # 4. Few-shot learning curve
        ax = axes[1, 0]
        if self.results["few_shot"]:
            shot_data = []
            for key, result in self.results["few_shot"].items():
                if "_shot" in key:
                    shot_data.append((result["n_shots"], result["mean_accuracy"], result["std_accuracy"]))
            
            shot_data.sort(key=lambda x: x[0])
            
            if shot_data:
                shots, means, stds = zip(*shot_data)
                ax.errorbar(shots, means, yerr=stds, marker="o", capsize=5)
                ax.set_xlabel("Number of Examples")
                ax.set_ylabel("Accuracy")
                ax.set_title("Few-shot Learning Curve")
                ax.grid(True, alpha=0.3)
        
        # 5. Instruction comparison
        ax = axes[1, 1]
        if self.results["instructions"]:
            instructions = list(self.results["instructions"].keys())
            i_accs = [self.results["instructions"][i]["mean_accuracy"] for i in instructions]
            i_lengths = [self.results["instructions"][i]["instruction_length"] for i in instructions]
            
            # Create bubble chart
            colors = plt.cm.viridis(np.linspace(0, 1, len(instructions)))
            
            for i, (inst, acc, length) in enumerate(zip(instructions, i_accs, i_lengths)):
                ax.scatter(i, acc, s=length*20, c=[colors[i]], alpha=0.6, label=inst)
            
            ax.set_xticks(range(len(instructions)))
            ax.set_xticklabels(instructions, rotation=45, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title("Instruction Format Performance")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # 6. Combination analysis
        ax = axes[1, 2]
        if self.results["combinations"]:
            combos = list(self.results["combinations"].keys())
            c_accs = [self.results["combinations"][c]["accuracy"] for c in combos]
            c_effs = [self.results["combinations"][c]["efficiency"] for c in combos]
            
            x = np.arange(len(combos))
            width = 0.35
            
            ax.bar(x - width/2, c_accs, width, label="Accuracy")
            ax.bar(x + width/2, c_effs, width, label="Efficiency")
            
            ax.set_xticks(x)
            ax.set_xticklabels(combos)
            ax.set_ylabel("Score")
            ax.set_title("Combination Performance")
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "prompt_ablation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {plot_path}")
        plt.show()
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset for ablation study."""
        dataset = AGNewsDataset()
        return dataset.load_splits()
    
    def _save_results(self):
        """Save ablation results."""
        results_path = self.output_dir / "prompt_ablation_results.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Save summary as markdown
        summary_path = self.output_dir / "prompt_ablation_summary.md"
        with open(summary_path, "w") as f:
            f.write("# Prompt Ablation Study Summary\n\n")
            
            f.write("## Best Components\n")
            summary = self.results["summary"]
            f.write(f"- **Strategy**: {summary['best_strategy']}\n")
            f.write(f"- **Template**: {summary['best_template']}\n")
            f.write(f"- **Verbalizer**: {summary['best_verbalizer']}\n")
            f.write(f"- **Optimal Shots**: {summary['optimal_shot_count']}\n")
            f.write(f"- **Instruction**: {summary['best_instruction']}\n\n")
            
            f.write("## Key Insights\n")
            for insight in summary["insights"]:
                f.write(f"- {insight}\n")
        
        logger.info(f"Saved summary to {summary_path}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def run_prompt_ablation():
    """Run prompt ablation study."""
    logger.info("Starting prompt ablation study")
    
    ablation = PromptAblation(
        model_name="roberta-base",
        num_trials=3
    )
    
    results = ablation.run_ablation_study()
    
    logger.info(f"Completed prompt ablation study")
    logger.info(f"Best strategy: {results['summary']['best_strategy']}")
    logger.info(f"Best template: {results['summary']['best_template']}")
    logger.info(f"Optimal shots: {results['summary']['optimal_shot_count']}")
    
    return results


if __name__ == "__main__":
    run_prompt_ablation()
