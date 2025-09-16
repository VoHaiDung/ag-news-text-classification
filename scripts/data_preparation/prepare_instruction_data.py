#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare Instruction-Following Data
===================================

Prepares data in instruction-following format for fine-tuning.

Following instruction tuning from:
- Wei et al. (2022): "Finetuned Language Models are Zero-Shot Learners"
- Wang et al. (2022): "Self-Instruct"

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.data.preprocessing.prompt_formatter import PromptFormatter, PromptFormatterConfig
from configs.constants import DATA_DIR, AG_NEWS_CLASSES, ID_TO_LABEL

logger = setup_logging(__name__)

def create_instructions() -> List[str]:
    """Create diverse instruction templates."""
    instructions = [
        "Classify the following news article into one of four categories: World, Sports, Business, or Sci/Tech.",
        "Determine which category this news article belongs to.",
        "Read the article and identify its topic category.",
        "What is the main topic of this news article?",
        "Categorize this news content into the appropriate section.",
        "Analyze the article and select the most fitting category.",
        "Based on the content, which news category does this belong to?",
        "Identify whether this article is about World news, Sports, Business, or Science/Technology.",
    ]
    return instructions

def format_as_instruction(
    text: str,
    label: int,
    instruction: str,
    add_explanation: bool = False
) -> Dict[str, str]:
    """Format single sample as instruction."""
    
    # Create output
    output = ID_TO_LABEL[label]
    
    # Add explanation if requested
    if add_explanation:
        explanations = {
            0: "This article discusses international events, politics, or global affairs.",
            1: "This article covers sports, athletics, games, or sporting events.",
            2: "This article is about business, finance, economics, or corporate news.",
            3: "This article relates to science, technology, or technical innovations."
        }
        output += f"\nExplanation: {explanations[label]}"
    
    return {
        "instruction": instruction,
        "input": text,
        "output": output
    }

def prepare_instruction_dataset(
    df: pd.DataFrame,
    add_explanations: bool = False,
    use_diverse_instructions: bool = True
) -> List[Dict[str, str]]:
    """Prepare instruction-following dataset."""
    
    instructions = create_instructions() if use_diverse_instructions else [create_instructions()[0]]
    
    instruction_data = []
    
    for idx, row in df.iterrows():
        # Select instruction
        instruction = instructions[idx % len(instructions)]
        
        # Format as instruction
        formatted = format_as_instruction(
            row['text'],
            row['label'],
            instruction,
            add_explanations
        )
        
        instruction_data.append(formatted)
    
    return instruction_data

def save_instruction_data(
    data: List[Dict[str, str]],
    output_path: Path,
    format: str = "jsonl"
):
    """Save instruction data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
    elif format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(data)} instruction samples to {output_path}")

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Prepare instruction data")
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "processed" / "train.csv",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "processed" / "train_instructions.jsonl",
        help="Output path"
    )
    
    parser.add_argument(
        "--add-explanations",
        action="store_true",
        help="Add explanations to outputs"
    )
    
    parser.add_argument(
        "--diverse-instructions",
        action="store_true",
        help="Use diverse instruction templates"
    )
    
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Prepare instruction data
    instruction_data = prepare_instruction_dataset(
        df,
        args.add_explanations,
        args.diverse_instructions
    )
    
    # Save instruction data
    save_instruction_data(instruction_data, args.output_path, args.format)
    
    logger.info("Instruction data preparation complete!")

if __name__ == "__main__":
    main()
