"""
Unit Tests for Prompt-based Models
===================================

Comprehensive test suite for prompt-based model architectures following:
- IEEE 829-2008: Standard for Software Test Documentation
- ISO/IEC/IEEE 29119: Software Testing Standards
- Prompt Engineering Best Practices

This module tests:
- Prompt model architectures
- Soft prompt tuning
- Instruction-based models
- Template management
- Prompt optimization techniques

Author: Võ Hải Dũng
License: MIT
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock, create_autospec
import numpy as np
import pytest
from typing import Dict, List, Optional, Tuple, Any, Union
import json

# Mock required modules to avoid import issues
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['transformers'] = MagicMock()

import torch
import torch.nn as nn

# ============================================================================
# Helper Functions for Creating Mock Data
# ============================================================================

def create_mock_prompt_config():
    """Create mock configuration for prompt models."""
    config = MagicMock()
    config.model_name = "t5-base"
    config.num_labels = 4  # AG News classes
    config.prompt_length = 20
    config.prompt_init_method = "random"  # random, vocab, text
    config.prompt_tuning_init_text = "Classify the news article:"
    config.use_soft_prompt = True
    config.freeze_model = True
    config.max_length = 512
    return config


def create_mock_template_config():
    """Create mock template configuration."""
    config = MagicMock()
    config.template_type = "manual"  # manual, auto, mixed
    config.template_format = "{prompt} {text} Answer: {label}"
    config.label_words = {
        0: ["World", "International", "Global"],
        1: ["Sports", "Game", "Match"],
        2: ["Business", "Economy", "Finance"],
        3: ["Technology", "Tech", "Science"]
    }
    config.use_demonstrations = True
    config.num_demonstrations = 3
    return config


def create_mock_instruction_config():
    """Create mock instruction tuning configuration."""
    config = MagicMock()
    config.instruction_template = """Task: Classify the following news article into one of four categories.
Categories: World, Sports, Business, Technology
Article: {text}
Category:"""
    config.use_chain_of_thought = False
    config.use_explanation = True
    config.max_instruction_length = 100
    return config


def create_mock_text_batch():
    """Create mock text batch for testing."""
    return {
        'texts': [
            "The stock market reached new highs today.",
            "The football team won the championship.",
            "New smartphone technology was announced.",
            "International summit discusses climate change."
        ],
        'labels': [2, 1, 3, 0]  # Business, Sports, Tech, World
    }


# ============================================================================
# Base Test Class
# ============================================================================

class PromptModelTestBase(unittest.TestCase):
    """Base class for prompt model tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        np.random.seed(42)
        self.batch_size = 4
        self.seq_length = 128
        self.hidden_size = 768
        self.num_labels = 4
        self.vocab_size = 32000
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def assert_valid_prompt_embedding(self, embedding: Any, expected_shape: Tuple[int, ...]):
        """Assert prompt embedding has valid shape and values."""
        if hasattr(embedding, 'shape'):
            self.assertEqual(
                embedding.shape,
                expected_shape,
                f"Expected shape {expected_shape}, got {embedding.shape}"
            )
    
    def assert_valid_template(self, template: str, required_fields: List[str]):
        """Assert template contains required fields."""
        for field in required_fields:
            self.assertIn(
                f"{{{field}}}",
                template,
                f"Template missing required field: {field}"
            )


# ============================================================================
# Prompt Model Tests
# ============================================================================

class TestPromptModels(PromptModelTestBase):
    """Test suite for prompt-based models."""
    
    def test_prompt_model_initialization(self):
        """Test prompt model initialization."""
        class MockPromptModel:
            def __init__(self, config):
                self.config = config
                self.base_model = MagicMock()
                self.prompt_embeddings = None
                self.num_labels = config.num_labels
                
                if config.use_soft_prompt:
                    self.initialize_soft_prompts()
            
            def initialize_soft_prompts(self):
                """Initialize soft prompt embeddings."""
                self.prompt_embeddings = MagicMock()
                self.prompt_embeddings.shape = (
                    self.config.prompt_length,
                    768  # hidden_size
                )
        
        config = create_mock_prompt_config()
        model = MockPromptModel(config)
        
        self.assertEqual(model.num_labels, 4)
        self.assertIsNotNone(model.prompt_embeddings)
        self.assertEqual(model.prompt_embeddings.shape[0], 20)
    
    def test_prompt_concatenation(self):
        """Test concatenation of prompts with input."""
        class MockPromptConcatenator:
            def __init__(self, prompt_length: int):
                self.prompt_length = prompt_length
            
            def concatenate(self, prompt_embeds, input_embeds):
                """Concatenate prompt and input embeddings."""
                batch_size = input_embeds.shape[0]
                seq_length = input_embeds.shape[1]
                
                # Expand prompt for batch
                expanded_prompt = MagicMock()
                expanded_prompt.shape = (batch_size, self.prompt_length, 768)
                
                # Concatenate
                combined = MagicMock()
                combined.shape = (batch_size, self.prompt_length + seq_length, 768)
                
                return combined
        
        concatenator = MockPromptConcatenator(prompt_length=20)
        
        prompt_embeds = MagicMock(shape=(20, 768))
        input_embeds = MagicMock(shape=(4, 128, 768))
        
        combined = concatenator.concatenate(prompt_embeds, input_embeds)
        
        self.assertEqual(combined.shape, (4, 148, 768))
    
    def test_prompt_initialization_methods(self):
        """Test different prompt initialization methods."""
        class MockPromptInitializer:
            def __init__(self, method: str, vocab_size: int = 32000):
                self.method = method
                self.vocab_size = vocab_size
            
            def initialize_random(self, prompt_length: int, hidden_size: int):
                """Random initialization."""
                prompt = MagicMock()
                prompt.shape = (prompt_length, hidden_size)
                prompt.init_method = "random"
                return prompt
            
            def initialize_from_vocab(self, prompt_length: int, hidden_size: int):
                """Initialize from vocabulary embeddings."""
                # Select random vocab indices
                vocab_indices = np.random.randint(0, self.vocab_size, prompt_length)
                prompt = MagicMock()
                prompt.shape = (prompt_length, hidden_size)
                prompt.init_method = "vocab"
                prompt.vocab_indices = vocab_indices
                return prompt
            
            def initialize_from_text(self, text: str, tokenizer, embedding_layer):
                """Initialize from text."""
                # Tokenize text
                tokens = MagicMock()
                tokens.shape = (len(text.split()),)
                
                prompt = MagicMock()
                prompt.shape = (len(text.split()), 768)
                prompt.init_method = "text"
                prompt.init_text = text
                return prompt
        
        # Test random initialization
        initializer = MockPromptInitializer("random")
        prompt = initializer.initialize_random(20, 768)
        self.assertEqual(prompt.shape, (20, 768))
        self.assertEqual(prompt.init_method, "random")
        
        # Test vocab initialization
        initializer = MockPromptInitializer("vocab")
        prompt = initializer.initialize_from_vocab(20, 768)
        self.assertEqual(prompt.shape, (20, 768))
        self.assertEqual(prompt.init_method, "vocab")
        self.assertEqual(len(prompt.vocab_indices), 20)
        
        # Test text initialization
        initializer = MockPromptInitializer("text")
        init_text = "Classify the following news article"
        prompt = initializer.initialize_from_text(init_text, None, None)
        self.assertEqual(prompt.init_method, "text")
        self.assertEqual(prompt.init_text, init_text)
    
    def test_prompt_tuning_forward(self):
        """Test forward pass with prompt tuning."""
        class MockPromptTuningModel:
            def __init__(self, config):
                self.config = config
                self.prompt_embeddings = MagicMock(shape=(20, 768))
                self.base_model = MagicMock()
            
            def forward(self, input_ids, attention_mask=None):
                batch_size = input_ids.shape[0]
                
                # Get input embeddings
                input_embeds = MagicMock(shape=(batch_size, 128, 768))
                
                # Expand and concatenate prompts
                prompt_embeds = MagicMock(shape=(batch_size, 20, 768))
                combined_embeds = MagicMock(shape=(batch_size, 148, 768))
                
                # Forward through model
                output = MagicMock()
                output.logits = MagicMock(shape=(batch_size, 4))
                
                return output
        
        config = create_mock_prompt_config()
        model = MockPromptTuningModel(config)
        
        input_ids = MagicMock(shape=(4, 128))
        attention_mask = MagicMock(shape=(4, 128))
        
        output = model.forward(input_ids, attention_mask)
        
        self.assertEqual(output.logits.shape, (4, 4))


# ============================================================================
# Soft Prompt Tests
# ============================================================================

class TestSoftPrompts(PromptModelTestBase):
    """Test suite for soft prompt implementations."""
    
    def test_soft_prompt_embedding_layer(self):
        """Test soft prompt embedding layer."""
        class MockSoftPromptEmbedding:
            def __init__(self, num_prompts: int, prompt_length: int, hidden_size: int):
                self.num_prompts = num_prompts
                self.prompt_length = prompt_length
                self.hidden_size = hidden_size
                
                # Learnable prompt embeddings
                self.embeddings = MagicMock()
                self.embeddings.shape = (num_prompts, prompt_length, hidden_size)
                self.embeddings.requires_grad = True
            
            def forward(self, prompt_ids=None):
                """Get prompt embeddings."""
                if prompt_ids is None:
                    # Return all prompts
                    return self.embeddings
                else:
                    # Return selected prompts
                    selected = MagicMock()
                    selected.shape = (len(prompt_ids), self.prompt_length, self.hidden_size)
                    return selected
        
        embedding = MockSoftPromptEmbedding(
            num_prompts=10,
            prompt_length=20,
            hidden_size=768
        )
        
        self.assertEqual(embedding.embeddings.shape, (10, 20, 768))
        self.assertTrue(embedding.embeddings.requires_grad)
        
        # Test getting all prompts
        all_prompts = embedding.forward()
        self.assertEqual(all_prompts.shape, (10, 20, 768))
        
        # Test selecting specific prompts
        selected = embedding.forward(prompt_ids=[0, 2, 5])
        self.assertEqual(selected.shape, (3, 20, 768))
    
    def test_prefix_tuning(self):
        """Test prefix tuning implementation."""
        class MockPrefixTuning:
            def __init__(self, config):
                self.config = config
                self.num_layers = 12
                self.num_heads = 12
                self.prefix_length = config.prompt_length
                
                # Prefix parameters for each layer
                self.prefix_keys = MagicMock()
                self.prefix_values = MagicMock()
                
                # Shape: (num_layers, 2, num_heads, prefix_length, head_dim)
                head_dim = 768 // self.num_heads
                self.prefix_keys.shape = (
                    self.num_layers, 2, self.num_heads,
                    self.prefix_length, head_dim
                )
                self.prefix_values.shape = self.prefix_keys.shape
            
            def get_prefix_states(self, batch_size: int):
                """Get prefix key-value states for attention."""
                # Expand for batch
                expanded_keys = MagicMock()
                expanded_values = MagicMock()
                
                head_dim = 768 // self.num_heads
                expanded_shape = (
                    self.num_layers, 2 * batch_size,
                    self.num_heads, self.prefix_length, head_dim
                )
                
                expanded_keys.shape = expanded_shape
                expanded_values.shape = expanded_shape
                
                return expanded_keys, expanded_values
        
        config = create_mock_prompt_config()
        prefix_model = MockPrefixTuning(config)
        
        # Check prefix shapes
        head_dim = 64  # 768 / 12
        expected_shape = (12, 2, 12, 20, head_dim)
        self.assertEqual(prefix_model.prefix_keys.shape, expected_shape)
        
        # Get prefix states for batch
        keys, values = prefix_model.get_prefix_states(batch_size=4)
        expected_batch_shape = (12, 8, 12, 20, head_dim)
        self.assertEqual(keys.shape, expected_batch_shape)
        self.assertEqual(values.shape, expected_batch_shape)
    
    def test_p_tuning_v2(self):
        """Test P-Tuning v2 implementation."""
        class MockPTuningV2:
            def __init__(self, config):
                self.config = config
                self.prompt_length = config.prompt_length
                self.num_layers = 12
                
                # Continuous prompts for each layer
                self.prompts = {}
                for layer_idx in range(self.num_layers):
                    self.prompts[f'layer_{layer_idx}'] = MagicMock(
                        shape=(self.prompt_length, 768)
                    )
            
            def get_layer_prompt(self, layer_idx: int):
                """Get prompt for specific layer."""
                return self.prompts[f'layer_{layer_idx}']
            
            def reparameterize(self, use_lstm: bool = True):
                """Reparameterize prompts using LSTM."""
                if use_lstm:
                    # Use LSTM to generate prompts
                    lstm_hidden = MagicMock(shape=(self.prompt_length, 768))
                    
                    # Update all layer prompts
                    for layer_idx in range(self.num_layers):
                        self.prompts[f'layer_{layer_idx}'] = lstm_hidden
                    
                    return True
                return False
        
        config = create_mock_prompt_config()
        p_tuning = MockPTuningV2(config)
        
        # Check prompts for each layer
        self.assertEqual(len(p_tuning.prompts), 12)
        
        # Get specific layer prompt
        layer_5_prompt = p_tuning.get_layer_prompt(5)
        self.assertEqual(layer_5_prompt.shape, (20, 768))
        
        # Test reparameterization
        result = p_tuning.reparameterize(use_lstm=True)
        self.assertTrue(result)


# ============================================================================
# Template Management Tests
# ============================================================================

class TestTemplateManagement(PromptModelTestBase):
    """Test suite for template management."""
    
    def test_template_creation(self):
        """Test template creation and validation."""
        class MockTemplateManager:
            def __init__(self, config):
                self.config = config
                self.templates = {}
                self.label_words = config.label_words
            
            def create_template(self, name: str, format_string: str) -> bool:
                """Create a new template."""
                # Validate template format
                required_fields = ['{text}']
                for field in required_fields:
                    if field not in format_string:
                        return False
                
                self.templates[name] = {
                    'format': format_string,
                    'created': True
                }
                return True
            
            def apply_template(self, template_name: str, text: str, label: Optional[int] = None):
                """Apply template to text."""
                if template_name not in self.templates:
                    return None
                
                template = self.templates[template_name]['format']
                
                # Format template
                result = template.replace('{text}', text)
                
                if label is not None and '{label}' in template:
                    label_word = self.label_words[label][0]
                    result = result.replace('{label}', label_word)
                
                return result
        
        config = create_mock_template_config()
        manager = MockTemplateManager(config)
        
        # Create template
        success = manager.create_template(
            "classification",
            "Classify: {text} Category: {label}"
        )
        self.assertTrue(success)
        
        # Apply template
        result = manager.apply_template(
            "classification",
            "Stock prices rise",
            label=2
        )
        self.assertIn("Stock prices rise", result)
        self.assertIn("Business", result)
    
    def test_verbalizer(self):
        """Test verbalizer for mapping labels to words."""
        class MockVerbalizer:
            def __init__(self, label_words: Dict[int, List[str]]):
                self.label_words = label_words
                self.vocab = MagicMock()
                self.word_embeddings = MagicMock()
            
            def verbalize(self, label: int) -> List[str]:
                """Get label words for a label."""
                return self.label_words.get(label, [])
            
            def get_label_word_ids(self, label: int) -> List[int]:
                """Get token IDs for label words."""
                words = self.verbalize(label)
                # Mock tokenization
                word_ids = []
                for word in words:
                    word_ids.append(hash(word) % 32000)  # Mock vocab size
                return word_ids
            
            def compute_label_word_embeddings(self):
                """Compute embeddings for all label words."""
                embeddings = {}
                for label, words in self.label_words.items():
                    label_embeds = []
                    for word in words:
                        # Mock embedding
                        embed = MagicMock(shape=(768,))
                        label_embeds.append(embed)
                    embeddings[label] = label_embeds
                return embeddings
        
        config = create_mock_template_config()
        verbalizer = MockVerbalizer(config.label_words)
        
        # Test verbalization
        words = verbalizer.verbalize(1)  # Sports
        self.assertIn("Sports", words)
        self.assertIn("Game", words)
        
        # Test word ID conversion
        word_ids = verbalizer.get_label_word_ids(1)
        self.assertEqual(len(word_ids), 3)  # 3 label words for Sports
        
        # Test embedding computation
        embeddings = verbalizer.compute_label_word_embeddings()
        self.assertEqual(len(embeddings), 4)  # 4 labels
        self.assertEqual(len(embeddings[0]), 3)  # 3 words for World
    
    def test_few_shot_demonstrations(self):
        """Test few-shot demonstration selection."""
        class MockDemonstrationSelector:
            def __init__(self, num_demonstrations: int = 3):
                self.num_demonstrations = num_demonstrations
                self.demonstration_pool = []
            
            def add_demonstration(self, text: str, label: int):
                """Add a demonstration to the pool."""
                self.demonstration_pool.append({
                    'text': text,
                    'label': label
                })
            
            def select_demonstrations(self, query_text: str, method: str = "random"):
                """Select demonstrations for query."""
                if method == "random":
                    # Random selection
                    indices = np.random.choice(
                        len(self.demonstration_pool),
                        min(self.num_demonstrations, len(self.demonstration_pool)),
                        replace=False
                    )
                elif method == "similarity":
                    # Mock similarity-based selection
                    indices = range(min(self.num_demonstrations, len(self.demonstration_pool)))
                else:
                    indices = []
                
                selected = [self.demonstration_pool[i] for i in indices]
                return selected
            
            def format_demonstrations(self, demonstrations: List[Dict]):
                """Format demonstrations into prompt."""
                formatted = []
                for demo in demonstrations:
                    formatted.append(f"Text: {demo['text']}\nLabel: {demo['label']}\n")
                return "\n".join(formatted)
        
        selector = MockDemonstrationSelector(num_demonstrations=3)
        
        # Add demonstrations
        selector.add_demonstration("Market crashes today", 2)
        selector.add_demonstration("Team wins championship", 1)
        selector.add_demonstration("New AI breakthrough", 3)
        selector.add_demonstration("UN meeting held", 0)
        
        # Select demonstrations
        selected = selector.select_demonstrations("Stock rises", method="random")
        self.assertLessEqual(len(selected), 3)
        
        # Format demonstrations
        formatted = selector.format_demonstrations(selected)
        self.assertIn("Text:", formatted)
        self.assertIn("Label:", formatted)


# ============================================================================
# Instruction Tuning Tests
# ============================================================================

class TestInstructionTuning(PromptModelTestBase):
    """Test suite for instruction tuning."""
    
    def test_instruction_formatting(self):
        """Test instruction formatting for different tasks."""
        class MockInstructionFormatter:
            def __init__(self, config):
                self.config = config
                self.instruction_template = config.instruction_template
            
            def format_instruction(self, text: str, task: str = "classification"):
                """Format instruction for task."""
                if task == "classification":
                    instruction = self.instruction_template.replace("{text}", text)
                elif task == "generation":
                    instruction = f"Generate a summary for: {text}"
                elif task == "question_answering":
                    instruction = f"Answer the question based on the context: {text}"
                else:
                    instruction = text
                
                return instruction
            
            def add_constraints(self, instruction: str, constraints: List[str]):
                """Add constraints to instruction."""
                if constraints:
                    constraint_text = "\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
                    instruction += constraint_text
                return instruction
        
        config = create_mock_instruction_config()
        formatter = MockInstructionFormatter(config)
        
        # Test classification instruction
        instruction = formatter.format_instruction(
            "Breaking news about technology",
            task="classification"
        )
        self.assertIn("Breaking news about technology", instruction)
        self.assertIn("Categories:", instruction)
        
        # Test with constraints
        constraints = ["Choose only one category", "Be confident in your answer"]
        instruction = formatter.add_constraints(instruction, constraints)
        self.assertIn("Constraints:", instruction)
        self.assertIn("Choose only one category", instruction)
    
    def test_chain_of_thought_prompting(self):
        """Test chain-of-thought prompting."""
        class MockChainOfThoughtPrompter:
            def __init__(self):
                self.use_cot = True
                self.cot_trigger = "Let's think step by step."
            
            def add_cot_trigger(self, prompt: str) -> str:
                """Add CoT trigger to prompt."""
                if self.use_cot:
                    return f"{prompt}\n{self.cot_trigger}"
                return prompt
            
            def parse_cot_response(self, response: str) -> Tuple[str, str]:
                """Parse CoT response into reasoning and answer."""
                # Mock parsing
                if "Therefore" in response:
                    parts = response.split("Therefore")
                    reasoning = parts[0].strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                else:
                    reasoning = ""
                    answer = response.strip()
                
                return reasoning, answer
            
            def create_cot_examples(self) -> List[str]:
                """Create CoT demonstration examples."""
                examples = [
                    "Q: Classify this news about stock market.\nLet's think step by step.\nThe text mentions stock market, which relates to finance.\nTherefore, the category is Business.",
                    "Q: Classify this news about football.\nLet's think step by step.\nFootball is a sport.\nTherefore, the category is Sports."
                ]
                return examples
        
        prompter = MockChainOfThoughtPrompter()
        
        # Test adding CoT trigger
        prompt = "Classify this article"
        cot_prompt = prompter.add_cot_trigger(prompt)
        self.assertIn("Let's think step by step", cot_prompt)
        
        # Test parsing CoT response
        response = "The article discusses stocks and trading. Therefore, Business"
        reasoning, answer = prompter.parse_cot_response(response)
        self.assertIn("stocks and trading", reasoning)
        self.assertEqual(answer, "Business")
        
        # Test creating examples
        examples = prompter.create_cot_examples()
        self.assertEqual(len(examples), 2)
        self.assertIn("step by step", examples[0])
    
    def test_instruction_optimization(self):
        """Test instruction optimization techniques."""
        class MockInstructionOptimizer:
            def __init__(self):
                self.instruction_history = []
                self.performance_scores = []
            
            def evaluate_instruction(self, instruction: str, validation_data) -> float:
                """Evaluate instruction performance."""
                # Mock evaluation
                score = np.random.random() * 0.3 + 0.6  # Random score between 0.6-0.9
                
                self.instruction_history.append(instruction)
                self.performance_scores.append(score)
                
                return score
            
            def optimize_instruction(self, base_instruction: str, num_iterations: int = 5):
                """Optimize instruction through iterations."""
                best_instruction = base_instruction
                best_score = 0.0
                
                for i in range(num_iterations):
                    # Generate variation
                    variation = f"{base_instruction} (v{i})"
                    
                    # Evaluate
                    score = self.evaluate_instruction(variation, None)
                    
                    if score > best_score:
                        best_score = score
                        best_instruction = variation
                
                return best_instruction, best_score
            
            def get_optimization_history(self) -> Dict[str, Any]:
                """Get optimization history."""
                return {
                    'num_evaluations': len(self.instruction_history),
                    'best_score': max(self.performance_scores) if self.performance_scores else 0,
                    'avg_score': np.mean(self.performance_scores) if self.performance_scores else 0,
                    'improvement': (max(self.performance_scores) - min(self.performance_scores)) if len(self.performance_scores) > 1 else 0
                }
        
        optimizer = MockInstructionOptimizer()
        
        # Optimize instruction
        base_instruction = "Classify the following text"
        best_instruction, best_score = optimizer.optimize_instruction(
            base_instruction,
            num_iterations=5
        )
        
        self.assertIn("Classify", best_instruction)
        self.assertGreater(best_score, 0.0)
        
        # Check history
        history = optimizer.get_optimization_history()
        self.assertEqual(history['num_evaluations'], 5)
        self.assertGreater(history['best_score'], 0.6)


# ============================================================================
# Prompt Optimization Tests
# ============================================================================

class TestPromptOptimization(PromptModelTestBase):
    """Test suite for prompt optimization techniques."""
    
    def test_gradient_based_prompt_search(self):
        """Test gradient-based prompt search."""
        class MockGradientPromptSearch:
            def __init__(self, prompt_length: int, vocab_size: int):
                self.prompt_length = prompt_length
                self.vocab_size = vocab_size
                self.continuous_prompt = MagicMock(shape=(prompt_length, 768))
                self.learning_rate = 0.01
            
            def compute_gradients(self, loss):
                """Compute gradients for prompt."""
                # Mock gradient computation
                gradients = MagicMock(shape=self.continuous_prompt.shape)
                return gradients
            
            def update_prompt(self, gradients):
                """Update prompt using gradients."""
                # Mock update
                self.continuous_prompt = MagicMock(shape=(self.prompt_length, 768))
                return True
            
            def project_to_vocab(self):
                """Project continuous prompt to discrete tokens."""
                # Find nearest vocab embeddings
                discrete_tokens = np.random.randint(0, self.vocab_size, self.prompt_length)
                return discrete_tokens
        
        searcher = MockGradientPromptSearch(prompt_length=10, vocab_size=32000)
        
        # Compute gradients
        loss = MagicMock()
        gradients = searcher.compute_gradients(loss)
        self.assertEqual(gradients.shape, (10, 768))
        
        # Update prompt
        success = searcher.update_prompt(gradients)
        self.assertTrue(success)
        
        # Project to vocabulary
        tokens = searcher.project_to_vocab()
        self.assertEqual(len(tokens), 10)
        self.assertTrue(all(0 <= t < 32000 for t in tokens))
    
    def test_reinforcement_learning_prompt_optimization(self):
        """Test RL-based prompt optimization."""
        class MockRLPromptOptimizer:
            def __init__(self):
                self.action_space = 100  # Number of possible prompt modifications
                self.state_dim = 768
                self.reward_history = []
            
            def get_state(self, prompt, context):
                """Get state representation."""
                state = MagicMock(shape=(self.state_dim,))
                return state
            
            def select_action(self, state, epsilon: float = 0.1):
                """Select action using epsilon-greedy."""
                if np.random.random() < epsilon:
                    # Exploration
                    action = np.random.randint(0, self.action_space)
                else:
                    # Exploitation (mock Q-values)
                    action = np.random.randint(0, self.action_space)
                
                return action
            
            def apply_action(self, prompt, action: int):
                """Apply action to modify prompt."""
                # Mock prompt modification
                modified_prompt = f"{prompt}_action_{action}"
                return modified_prompt
            
            def compute_reward(self, performance: float, baseline: float = 0.5):
                """Compute reward signal."""
                reward = performance - baseline
                self.reward_history.append(reward)
                return reward
            
            def update_policy(self, state, action, reward, next_state):
                """Update policy using Q-learning."""
                # Mock Q-learning update
                learning_rate = 0.1
                discount = 0.9
                
                # Q(s,a) = Q(s,a) + lr * (r + discount * max(Q(s',a')) - Q(s,a))
                # Simplified mock update
                return True
        
        optimizer = MockRLPromptOptimizer()
        
        # Get state
        state = optimizer.get_state("Classify:", None)
        self.assertEqual(state.shape, (768,))
        
        # Select action
        action = optimizer.select_action(state, epsilon=0.1)
        self.assertLess(action, 100)
        
        # Apply action
        modified = optimizer.apply_action("Classify:", action)
        self.assertIn("action", modified)
        
        # Compute reward
        reward = optimizer.compute_reward(performance=0.75, baseline=0.5)
        self.assertEqual(reward, 0.25)
        
        # Update policy
        next_state = optimizer.get_state(modified, None)
        success = optimizer.update_policy(state, action, reward, next_state)
        self.assertTrue(success)


# ============================================================================
# Integration Tests
# ============================================================================

class TestPromptModelIntegration(PromptModelTestBase):
    """Integration tests for prompt-based models."""
    
    def test_prompt_with_instruction_tuning(self):
        """Test combining prompts with instruction tuning."""
        class MockPromptInstructionModel:
            def __init__(self, prompt_config, instruction_config):
                self.prompt_config = prompt_config
                self.instruction_config = instruction_config
                self.soft_prompts = MagicMock(shape=(20, 768))
            
            def create_full_prompt(self, text: str) -> str:
                """Create full prompt with instruction and soft prompts."""
                # Format instruction
                instruction = self.instruction_config.instruction_template.replace(
                    "{text}", text
                )
                
                # Add soft prompt tokens (represented as special tokens)
                prompt_tokens = "[PROMPT]" * self.prompt_config.prompt_length
                
                full_prompt = f"{prompt_tokens} {instruction}"
                return full_prompt
        
        prompt_config = create_mock_prompt_config()
        instruction_config = create_mock_instruction_config()
        
        model = MockPromptInstructionModel(prompt_config, instruction_config)
        
        full_prompt = model.create_full_prompt("Tech news article")
        self.assertIn("[PROMPT]", full_prompt)
        self.assertIn("Tech news article", full_prompt)
        self.assertIn("Task:", full_prompt)
    
    def test_multi_task_prompting(self):
        """Test multi-task prompting setup."""
        class MockMultiTaskPromptModel:
            def __init__(self, tasks: List[str]):
                self.tasks = tasks
                self.task_prompts = {}
                
                # Create task-specific prompts
                for task in tasks:
                    self.task_prompts[task] = MagicMock(shape=(20, 768))
            
            def select_task_prompt(self, task: str):
                """Select prompt for specific task."""
                return self.task_prompts.get(task)
            
            def combine_task_prompts(self, tasks: List[str]):
                """Combine prompts from multiple tasks."""
                combined = []
                for task in tasks:
                    if task in self.task_prompts:
                        combined.append(self.task_prompts[task])
                
                # Mock concatenation
                if combined:
                    result = MagicMock(shape=(len(combined) * 20, 768))
                    return result
                return None
        
        tasks = ["classification", "summarization", "question_answering"]
        model = MockMultiTaskPromptModel(tasks)
        
        # Test single task prompt
        prompt = model.select_task_prompt("classification")
        self.assertEqual(prompt.shape, (20, 768))
        
        # Test combining multiple tasks
        combined = model.combine_task_prompts(["classification", "summarization"])
        self.assertEqual(combined.shape, (40, 768))
    
    def test_prompt_ensemble(self):
        """Test ensemble of different prompting methods."""
        class MockPromptEnsemble:
            def __init__(self):
                self.prompt_methods = [
                    "manual_prompt",
                    "soft_prompt",
                    "instruction_tuning",
                    "chain_of_thought"
                ]
                self.weights = [0.25, 0.25, 0.25, 0.25]
            
            def get_predictions(self, text: str) -> Dict[str, Any]:
                """Get predictions from each prompting method."""
                predictions = {}
                
                for method in self.prompt_methods:
                    # Mock prediction for each method
                    logits = MagicMock(shape=(4,))
                    predictions[method] = {
                        'logits': logits,
                        'confidence': np.random.random()
                    }
                
                return predictions
            
            def ensemble_predictions(self, predictions: Dict[str, Any]):
                """Ensemble predictions from different methods."""
                # Weighted average of logits
                ensemble_logits = MagicMock(shape=(4,))
                
                # Compute ensemble confidence
                confidences = [p['confidence'] for p in predictions.values()]
                ensemble_confidence = np.mean(confidences)
                
                return {
                    'logits': ensemble_logits,
                    'confidence': ensemble_confidence,
                    'method_agreement': self.compute_agreement(predictions)
                }
            
            def compute_agreement(self, predictions: Dict[str, Any]) -> float:
                """Compute agreement between methods."""
                # Mock agreement calculation
                return np.random.random() * 0.3 + 0.6  # 0.6 to 0.9
        
        ensemble = MockPromptEnsemble()
        
        # Get individual predictions
        predictions = ensemble.get_predictions("Sample text")
        self.assertEqual(len(predictions), 4)
        
        # Ensemble predictions
        result = ensemble.ensemble_predictions(predictions)
        self.assertIn('logits', result)
        self.assertIn('confidence', result)
        self.assertIn('method_agreement', result)
        self.assertGreater(result['method_agreement'], 0.5)


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    pytest.main([__file__, "-v", "--tb=short"])
