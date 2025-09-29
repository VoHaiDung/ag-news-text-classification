#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Instruction-Following Data Preparation Script for AG News Text Classification
================================================================================
This script prepares training data in instruction-following format for fine-tuning
large language models on the AG News classification task, implementing state-of-the-art
instruction tuning techniques. It converts traditional supervised learning data into
conversational instruction-response pairs that enable models to follow natural language
instructions for classification tasks.

The instruction formatting approach enables leveraging powerful pretrained language
models through instruction tuning, improving zero-shot and few-shot performance while
maintaining strong supervised learning capabilities. This method has shown remarkable
success in making models more generalizable and user-friendly.

References:
    - Wei, J. et al. (2022): Finetuned Language Models are Zero-Shot Learners (FLAN)
    - Wang, Y. et al. (2022): Self-Instruct - Aligning Language Models with Self-Generated Instructions
    - Ouyang, L. et al. (2022): Training language models to follow instructions with human feedback
    - Sanh, V. et al. (2022): Multitask Prompted Training Enables Zero-Shot Task Generalization (T0)
    - Chung, H.W. et al. (2022): Scaling Instruction-Finetuned Language Models
    - Longpre, S. et al. (2023): The Flan Collection - Designing Data and Methods for Effective Instruction Tuning

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import random
from dataclasses import dataclass, field
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.data.preprocessing.prompt_formatter import PromptFormatter, PromptFormatterConfig
from configs.constants import DATA_DIR, AG_NEWS_CLASSES, ID_TO_LABEL

logger = setup_logging(__name__)


def create_instructions() -> List[str]:
    """
    Create diverse instruction templates for classification
    
    Generates a variety of instruction phrasings following the diversity principles
    from Wei et al. (2022) FLAN paper, which showed that instruction diversity
    significantly improves model generalization. Each instruction variant helps
    the model understand different ways users might phrase the same task.
    
    The instructions cover different linguistic styles:
    - Direct commands ("Classify...")
    - Questions ("What is...")
    - Analytical requests ("Analyze...")
    - Contextual descriptions ("Based on...")
    
    Returns:
        List of instruction template strings with varied phrasings
    """
    instructions = [
        # Direct classification instructions
        "Classify the following news article into one of four categories: World, Sports, Business, or Sci/Tech.",
        "Determine which category this news article belongs to: World, Sports, Business, or Sci/Tech.",
        
        # Question-based instructions
        "What is the main topic of this news article? Choose from: World, Sports, Business, or Sci/Tech.",
        "Which news category does this article fall under? Options: World, Sports, Business, Sci/Tech.",
        
        # Task-oriented instructions
        "Read the article below and identify its topic category from these options: World, Sports, Business, or Sci/Tech.",
        "Categorize this news content into the appropriate section: World, Sports, Business, or Sci/Tech.",
        
        # Analytical instructions
        "Analyze the article and select the most fitting category from: World, Sports, Business, or Sci/Tech.",
        "Based on the content, determine if this is a World, Sports, Business, or Sci/Tech article.",
        
        # Contextual instructions
        "Identify whether this article is about World news, Sports, Business, or Science/Technology.",
        "This is a news article. Please classify it as World, Sports, Business, or Sci/Tech news.",
        
        # Editorial-style instructions
        "As a news editor, categorize this article into: World, Sports, Business, or Sci/Tech.",
        "Help organize this news article by assigning it to World, Sports, Business, or Sci/Tech category.",
        
        # Zero-shot style instructions
        "Given the following news article, determine its category. The categories are: World (international news), Sports (athletic events), Business (economic/corporate news), or Sci/Tech (science and technology).",
        "You are a news categorization system. Classify the following article as World, Sports, Business, or Sci/Tech.",
    ]
    return instructions


def create_explanations() -> Dict[int, List[str]]:
    """
    Create diverse explanations for each category
    
    Generates multiple explanation templates per category following the
    chain-of-thought prompting approach from Wei et al. (2022) and the
    explanation-based learning from Camburu et al. (2018). Explanations
    help models learn reasoning patterns.
    
    Returns:
        Dictionary mapping label IDs to lists of explanation templates
    """
    explanations = {
        0: [  # World
            "This article discusses international events, politics, or global affairs.",
            "The content covers world news, including international relations and global events.",
            "This piece focuses on international developments, foreign policy, or worldwide issues.",
            "The article reports on global matters, international politics, or world events.",
        ],
        1: [  # Sports  
            "This article covers sports, athletics, games, or sporting events.",
            "The content is about athletic competitions, sports news, or recreational activities.",
            "This piece discusses sports teams, athletes, games, or sporting achievements.",
            "The article focuses on sports events, athletic performance, or competitive games.",
        ],
        2: [  # Business
            "This article is about business, finance, economics, or corporate news.",
            "The content covers economic matters, business operations, or financial markets.",
            "This piece discusses corporate activities, market trends, or economic policies.",
            "The article reports on business developments, financial news, or economic indicators.",
        ],
        3: [  # Sci/Tech
            "This article relates to science, technology, or technical innovations.",
            "The content covers scientific discoveries, technological advances, or research.",
            "This piece discusses technology products, scientific research, or technical developments.",
            "The article focuses on scientific breakthroughs, tech industry news, or innovations.",
        ]
    }
    return explanations


def format_as_instruction(
    text: str,
    label: int,
    instruction: str,
    add_explanation: bool = False,
    explanation_style: str = "standard"
) -> Dict[str, str]:
    """
    Format a single sample as an instruction-following example
    
    Converts traditional (text, label) pairs into instruction-response format
    following the template structure from Wei et al. (2022) FLAN and
    Wang et al. (2022) Self-Instruct papers. This format enables models
    to understand tasks through natural language instructions.
    
    Args:
        text: The input news article text
        label: The ground truth label (0-3)
        instruction: The instruction template to use
        add_explanation: Whether to include reasoning explanation
        explanation_style: Style of explanation ("standard", "detailed", "cot")
        
    Returns:
        Dictionary with "instruction", "input", and "output" fields
    """
    
    # Get the category name
    output = ID_TO_LABEL[label]
    
    # Add explanation if requested
    if add_explanation:
        explanations = create_explanations()
        
        if explanation_style == "standard":
            # Simple explanation
            explanation = random.choice(explanations[label])
            output += f"\nExplanation: {explanation}"
            
        elif explanation_style == "detailed":
            # More detailed explanation with reasoning
            explanation = random.choice(explanations[label])
            output = f"Category: {output}\n"
            output += f"Reasoning: {explanation}\n"
            output += "This classification is based on the key topics and themes present in the article."
            
        elif explanation_style == "cot":
            # Chain-of-thought style explanation
            output = f"Let me analyze this article step by step:\n"
            output += f"1. First, I'll identify the main topic.\n"
            output += f"2. The article appears to be about {ID_TO_LABEL[label].lower()} related content.\n"
            output += f"3. {random.choice(explanations[label])}\n"
            output += f"4. Therefore, the category is: {ID_TO_LABEL[label]}"
    
    return {
        "instruction": instruction,
        "input": text,
        "output": output
    }


def prepare_instruction_dataset(
    df: pd.DataFrame,
    add_explanations: bool = False,
    use_diverse_instructions: bool = True,
    explanation_style: str = "standard",
    max_samples: Optional[int] = None,
    balance_classes: bool = False
) -> List[Dict[str, str]]:
    """
    Prepare complete instruction-following dataset
    
    Converts a DataFrame of classification data into instruction-following format
    implementing techniques from FLAN (Wei et al., 2022) and T0 (Sanh et al., 2022)
    for effective instruction tuning. Supports various formatting options to
    create diverse training examples.
    
    Args:
        df: DataFrame containing 'text' and 'label' columns
        add_explanations: Whether to add explanations to outputs
        use_diverse_instructions: Whether to use varied instruction templates
        explanation_style: Style of explanations to generate
        max_samples: Optional limit on number of samples
        balance_classes: Whether to balance samples across classes
        
    Returns:
        List of instruction-formatted dictionaries
    """
    logger.info(f"Preparing instruction dataset from {len(df)} samples")
    
    # Balance classes if requested
    if balance_classes:
        logger.info("Balancing classes...")
        min_class_count = df['label'].value_counts().min()
        balanced_dfs = []
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            balanced_dfs.append(label_df.sample(n=min_class_count, random_state=42))
        df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Balanced dataset to {len(df)} samples")
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"Limited to {max_samples} samples")
    
    # Get instruction templates
    instructions = create_instructions() if use_diverse_instructions else [create_instructions()[0]]
    
    instruction_data = []
    
    for idx, row in df.iterrows():
        # Select instruction (cycle through if using diverse instructions)
        instruction = instructions[idx % len(instructions)]
        
        # Format as instruction
        formatted = format_as_instruction(
            row['text'],
            row['label'],
            instruction,
            add_explanations,
            explanation_style
        )
        
        # Add metadata for tracking
        formatted['metadata'] = {
            'original_label': row['label'],
            'category': ID_TO_LABEL[row['label']],
            'instruction_template_idx': idx % len(instructions),
            'has_explanation': add_explanations
        }
        
        instruction_data.append(formatted)
    
    # Log statistics
    logger.info(f"Created {len(instruction_data)} instruction examples")
    if use_diverse_instructions:
        logger.info(f"Used {len(instructions)} different instruction templates")
    
    return instruction_data


def augment_with_few_shot_examples(
    instruction_data: List[Dict[str, str]],
    num_shots: int = 3,
    include_negative: bool = False
) -> List[Dict[str, str]]:
    """
    Augment instructions with few-shot examples
    
    Implements few-shot prompting following Brown et al. (2020) GPT-3 paper
    and Min et al. (2022) on few-shot learning. Adds demonstration examples
    to instructions to improve in-context learning.
    
    Args:
        instruction_data: List of instruction examples
        num_shots: Number of demonstration examples to include
        include_negative: Whether to include incorrect examples
        
    Returns:
        Augmented instruction data with few-shot examples
    """
    logger.info(f"Augmenting with {num_shots}-shot examples")
    
    augmented_data = []
    
    # Group by category for sampling
    category_examples = {i: [] for i in range(4)}
    for item in instruction_data:
        label = item['metadata']['original_label']
        category_examples[label].append(item)
    
    for item in instruction_data:
        # Sample few-shot examples
        examples = []
        for category in range(4):
            if len(category_examples[category]) > 0:
                sample = random.choice(category_examples[category])
                if sample != item:  # Don't include the item itself
                    examples.append(sample)
                    if len(examples) >= num_shots:
                        break
        
        if examples:
            # Create few-shot prompt
            few_shot_text = "Here are some examples:\n\n"
            for ex in examples:
                few_shot_text += f"Input: {ex['input'][:100]}...\n"
                few_shot_text += f"Output: {ex['output']}\n\n"
            
            # Prepend to instruction
            item['instruction'] = few_shot_text + item['instruction']
        
        augmented_data.append(item)
    
    return augmented_data


def save_instruction_data(
    data: List[Dict[str, str]],
    output_path: Path,
    format: str = "jsonl",
    include_metadata: bool = True
):
    """
    Save instruction data in specified format
    
    Persists the instruction-formatted data following common formats used
    in instruction tuning pipelines. Supports JSONL for streaming and
    JSON for complete dataset loading.
    
    Args:
        data: List of instruction dictionaries
        output_path: Path for output file
        format: Output format ("jsonl" or "json")
        include_metadata: Whether to include metadata in output
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove metadata if not needed
    if not include_metadata:
        data = [{k: v for k, v in item.items() if k != 'metadata'} for item in data]
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
    elif format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} instruction samples to {output_path}")
    
    # Save statistics
    stats_path = output_path.with_suffix('.stats.json')
    stats = {
        'total_samples': len(data),
        'timestamp': datetime.now().isoformat(),
        'format': format,
        'includes_metadata': include_metadata,
        'instruction_types': len(set(item['instruction'][:50] for item in data)),
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to {stats_path}")


def main():
    """
    Main entry point for instruction data preparation
    
    Orchestrates the complete instruction formatting pipeline with comprehensive
    options for creating diverse, high-quality instruction-tuning datasets following
    state-of-the-art practices from FLAN, T0, and InstructGPT.
    """
    parser = argparse.ArgumentParser(
        description="Prepare instruction-following format data for fine-tuning LLMs"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "processed" / "train.csv",
        help="Path to training data CSV file"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "processed" / "train_instructions.jsonl",
        help="Output path for instruction data"
    )
    
    parser.add_argument(
        "--add-explanations",
        action="store_true",
        help="Add reasoning explanations to outputs"
    )
    
    parser.add_argument(
        "--explanation-style",
        choices=["standard", "detailed", "cot"],
        default="standard",
        help="Style of explanations (standard/detailed/chain-of-thought)"
    )
    
    parser.add_argument(
        "--diverse-instructions",
        action="store_true",
        help="Use diverse instruction templates for variety"
    )
    
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help="Number of few-shot examples to include (0 for none)"
    )
    
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance samples across classes"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (jsonl for streaming, json for complete)"
    )
    
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata in output for analysis"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Log class distribution
    class_dist = df['label'].value_counts().sort_index()
    logger.info(f"Class distribution: {dict(class_dist)}")
    
    # Prepare instruction data
    instruction_data = prepare_instruction_dataset(
        df,
        add_explanations=args.add_explanations,
        use_diverse_instructions=args.diverse_instructions,
        explanation_style=args.explanation_style,
        max_samples=args.max_samples,
        balance_classes=args.balance_classes
    )
    
    # Add few-shot examples if requested
    if args.few_shot > 0:
        instruction_data = augment_with_few_shot_examples(
            instruction_data,
            num_shots=args.few_shot
        )
    
    # Save instruction data
    save_instruction_data(
        instruction_data, 
        args.output_path, 
        args.format,
        args.include_metadata
    )
    
    logger.info("Instruction data preparation complete!")


if __name__ == "__main__":
    main()
