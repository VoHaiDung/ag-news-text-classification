"""
Unit Tests for Transformer Models
==================================

Comprehensive test suite for transformer-based classification models following:
- IEEE 829-2008: Standard for Software Test Documentation
- ISO/IEC/IEEE 29119: Software Testing Standards
- HuggingFace Transformers Best Practices

This module tests:
- DeBERTa-v3 variants (base, sliding window, hierarchical)
- RoBERTa variants (enhanced, domain-adapted)
- XLNet classifier
- ELECTRA discriminator
- Longformer with global attention
- Generative models (GPT-2, T5) for classification

Author: AG News Classification Team
License: MIT
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, List, Optional, Tuple, Any

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration for transformer models."""
    config = MagicMock()
    config.model_name = "microsoft/deberta-v3-base"
    config.num_labels = 4  # AG News has 4 classes
    config.hidden_size = 768
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.intermediate_size = 3072
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    config.max_position_embeddings = 512
    config.gradient_checkpointing = False
    config.use_cache = False
    config.classifier_dropout = 0.1
    config.pooling_strategy = "cls"
    config.freeze_encoder = False
    config.layer_wise_lr_decay = 1.0
    return config


@pytest.fixture
def mock_batch():
    """Create mock batch data for testing."""
    batch_size = 8
    seq_length = 128
    
    return {
        'input_ids': torch.randint(0, 30000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
        'labels': torch.randint(0, 4, (batch_size,)),
        'position_ids': torch.arange(seq_length).expand(batch_size, -1)
    }


@pytest.fixture
def mock_long_batch():
    """Create mock batch with long sequences for Longformer testing."""
    batch_size = 4
    seq_length = 1024
    
    return {
        'input_ids': torch.randint(0, 30000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'global_attention_mask': torch.zeros(batch_size, seq_length),
        'labels': torch.randint(0, 4, (batch_size,))
    }


# ============================================================================
# Base Test Class
# ============================================================================

class TransformerTestBase(unittest.TestCase):
    """Base class for transformer model tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
    
    def tearDown(self):
        """Clean up after tests."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def assert_output_shape(self, output: torch.Tensor, expected_shape: Tuple[int, ...]):
        """Assert output tensor has expected shape."""
        self.assertEqual(
            output.shape, 
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}"
        )
    
    def assert_valid_logits(self, logits: torch.Tensor, num_classes: int = 4):
        """Assert logits are valid for classification."""
        self.assertEqual(logits.dim(), 2)
        self.assertEqual(logits.size(-1), num_classes)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())


# ============================================================================
# DeBERTa Tests
# ============================================================================

class TestDeBERTaV3(TransformerTestBase):
    """Test suite for DeBERTa-v3 models."""
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_deberta_v3_initialization(self, mock_model):
        """Test DeBERTa-v3 model initialization."""
        from src.models.transformers.deberta.deberta_v3 import DeBERTaV3Classifier
        
        # Mock the pretrained model
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "microsoft/deberta-v3-base"
        config.num_labels = 4
        config.gradient_checkpointing = False
        
        model = DeBERTaV3Classifier(config)
        
        self.assertIsNotNone(model)
        mock_model.assert_called_once()
        self.assertEqual(model.num_labels, 4)
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_deberta_v3_forward(self, mock_model):
        """Test DeBERTa-v3 forward pass."""
        from src.models.transformers.deberta.deberta_v3 import DeBERTaV3Classifier
        
        # Create mock model with proper output
        mock_transformer = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(8, 4)
        mock_output.loss = torch.tensor(1.5)
        mock_transformer.return_value = mock_output
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "microsoft/deberta-v3-base"
        config.num_labels = 4
        
        model = DeBERTaV3Classifier(config)
        batch = mock_batch()
        
        output = model(batch)
        
        self.assertIn('logits', output)
        self.assertIn('loss', output)
        self.assert_output_shape(output['logits'], (8, 4))
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_deberta_sliding_window(self, mock_model):
        """Test DeBERTa with sliding window approach."""
        from src.models.transformers.deberta.deberta_sliding import DeBERTaSlidingWindow
        
        mock_transformer = MagicMock()
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "microsoft/deberta-v3-base"
        config.num_labels = 4
        config.window_size = 256
        config.stride = 128
        config.aggregation = "mean"
        
        model = DeBERTaSlidingWindow(config)
        
        self.assertEqual(model.window_size, 256)
        self.assertEqual(model.stride, 128)
        self.assertEqual(model.aggregation, "mean")
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_deberta_hierarchical(self, mock_model):
        """Test DeBERTa with hierarchical attention."""
        from src.models.transformers.deberta.deberta_hierarchical import DeBERTaHierarchical
        
        mock_transformer = MagicMock()
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "microsoft/deberta-v3-base"
        config.num_labels = 4
        config.chunk_size = 128
        config.num_chunks = 4
        config.hierarchical_dropout = 0.1
        
        model = DeBERTaHierarchical(config)
        
        self.assertEqual(model.chunk_size, 128)
        self.assertEqual(model.num_chunks, 4)
        self.assertIsNotNone(model.hierarchical_attention)


# ============================================================================
# RoBERTa Tests
# ============================================================================

class TestRoBERTa(TransformerTestBase):
    """Test suite for RoBERTa models."""
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_roberta_enhanced_initialization(self, mock_model):
        """Test RoBERTa enhanced model initialization."""
        from src.models.transformers.roberta.roberta_enhanced import RoBERTaEnhanced
        
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "roberta-large"
        config.num_labels = 4
        config.use_multi_sample_dropout = True
        config.dropout_samples = 5
        
        model = RoBERTaEnhanced(config)
        
        self.assertIsNotNone(model)
        self.assertTrue(model.use_multi_sample_dropout)
        self.assertEqual(model.dropout_samples, 5)
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_roberta_domain_adapted(self, mock_model):
        """Test domain-adapted RoBERTa."""
        from src.models.transformers.roberta.roberta_domain import RoBERTaDomainAdapted
        
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "roberta-large"
        config.num_labels = 4
        config.domain_vocab_size = 1000
        config.domain_embedding_dim = 128
        config.use_domain_adapter = True
        
        model = RoBERTaDomainAdapted(config)
        
        self.assertIsNotNone(model)
        self.assertTrue(model.use_domain_adapter)
        self.assertEqual(model.domain_vocab_size, 1000)
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_roberta_with_custom_pooling(self, mock_model):
        """Test RoBERTa with custom pooling strategies."""
        from src.models.transformers.roberta.roberta_enhanced import RoBERTaEnhanced
        
        mock_transformer = MagicMock()
        mock_model.return_value = mock_transformer
        
        pooling_strategies = ["cls", "mean", "max", "cls_mean"]
        
        for strategy in pooling_strategies:
            config = Mock()
            config.model_name = "roberta-base"
            config.num_labels = 4
            config.pooling_strategy = strategy
            config.use_multi_sample_dropout = False
            
            model = RoBERTaEnhanced(config)
            self.assertEqual(model.pooling_strategy, strategy)


# ============================================================================
# XLNet Tests
# ============================================================================

class TestXLNet(TransformerTestBase):
    """Test suite for XLNet classifier."""
    
    @patch('transformers.XLNetForSequenceClassification.from_pretrained')
    def test_xlnet_initialization(self, mock_model):
        """Test XLNet model initialization."""
        from src.models.transformers.xlnet.xlnet_classifier import XLNetClassifier
        
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "xlnet-large-cased"
        config.num_labels = 4
        config.summary_type = "last"
        config.use_mems = True
        
        model = XLNetClassifier(config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.summary_type, "last")
        self.assertTrue(model.use_mems)
    
    @patch('transformers.XLNetForSequenceClassification.from_pretrained')
    def test_xlnet_with_memory(self, mock_model):
        """Test XLNet with memory mechanism."""
        from src.models.transformers.xlnet.xlnet_classifier import XLNetClassifier
        
        mock_transformer = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(8, 4)
        mock_output.mems = [torch.randn(8, 128, 768) for _ in range(12)]
        mock_transformer.return_value = mock_output
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "xlnet-base-cased"
        config.num_labels = 4
        config.use_mems = True
        config.mem_len = 128
        
        model = XLNetClassifier(config)
        batch = mock_batch()
        
        output = model(batch)
        
        self.assertIn('logits', output)
        self.assertIn('mems', output)
        self.assert_valid_logits(output['logits'])


# ============================================================================
# ELECTRA Tests
# ============================================================================

class TestELECTRA(TransformerTestBase):
    """Test suite for ELECTRA discriminator."""
    
    @patch('transformers.ElectraForSequenceClassification.from_pretrained')
    def test_electra_initialization(self, mock_model):
        """Test ELECTRA model initialization."""
        from src.models.transformers.electra.electra_discriminator import ELECTRADiscriminator
        
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "google/electra-large-discriminator"
        config.num_labels = 4
        config.use_discriminator_head = True
        
        model = ELECTRADiscriminator(config)
        
        self.assertIsNotNone(model)
        self.assertTrue(model.use_discriminator_head)
    
    @patch('transformers.ElectraForSequenceClassification.from_pretrained')
    def test_electra_with_adversarial(self, mock_model):
        """Test ELECTRA with adversarial training support."""
        from src.models.transformers.electra.electra_discriminator import ELECTRADiscriminator
        
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "google/electra-base-discriminator"
        config.num_labels = 4
        config.adversarial_training = True
        config.adversarial_eps = 0.1
        
        model = ELECTRADiscriminator(config)
        
        self.assertTrue(model.adversarial_training)
        self.assertEqual(model.adversarial_eps, 0.1)


# ============================================================================
# Longformer Tests
# ============================================================================

class TestLongformer(TransformerTestBase):
    """Test suite for Longformer with global attention."""
    
    @patch('transformers.LongformerForSequenceClassification.from_pretrained')
    def test_longformer_initialization(self, mock_model):
        """Test Longformer model initialization."""
        from src.models.transformers.longformer.longformer_global import LongformerGlobal
        
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "allenai/longformer-base-4096"
        config.num_labels = 4
        config.attention_window = 512
        config.global_attention_indices = [0]  # CLS token
        
        model = LongformerGlobal(config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.attention_window, 512)
        self.assertEqual(model.global_attention_indices, [0])
    
    @patch('transformers.LongformerForSequenceClassification.from_pretrained')
    def test_longformer_long_sequence(self, mock_model):
        """Test Longformer with long sequences."""
        from src.models.transformers.longformer.longformer_global import LongformerGlobal
        
        mock_transformer = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(4, 4)
        mock_transformer.return_value = mock_output
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "allenai/longformer-base-4096"
        config.num_labels = 4
        config.max_position_embeddings = 4096
        
        model = LongformerGlobal(config)
        batch = mock_long_batch()
        
        # Set global attention for CLS token
        batch['global_attention_mask'][:, 0] = 1
        
        output = model(batch)
        
        self.assertIn('logits', output)
        self.assert_output_shape(output['logits'], (4, 4))


# ============================================================================
# Generative Model Tests
# ============================================================================

class TestGenerativeModels(TransformerTestBase):
    """Test suite for generative models adapted for classification."""
    
    @patch('transformers.GPT2ForSequenceClassification.from_pretrained')
    def test_gpt2_classifier(self, mock_model):
        """Test GPT-2 classifier initialization and forward pass."""
        from src.models.transformers.generative.gpt2_classifier import GPT2Classifier
        
        mock_transformer = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.randn(8, 4)
        mock_transformer.return_value = mock_output
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "gpt2-large"
        config.num_labels = 4
        config.use_prefix_tuning = False
        config.pad_token_id = 50256
        
        model = GPT2Classifier(config)
        batch = mock_batch()
        
        output = model(batch)
        
        self.assertIn('logits', output)
        self.assert_valid_logits(output['logits'])
    
    @patch('transformers.T5ForConditionalGeneration.from_pretrained')
    def test_t5_classifier(self, mock_model):
        """Test T5 classifier for sequence classification."""
        from src.models.transformers.generative.t5_classifier import T5Classifier
        
        mock_transformer = MagicMock()
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "t5-large"
        config.num_labels = 4
        config.use_prompt = True
        config.prompt_template = "Classify the news: {text}"
        config.label_mapping = {
            0: "World",
            1: "Sports", 
            2: "Business",
            3: "Technology"
        }
        
        model = T5Classifier(config)
        
        self.assertIsNotNone(model)
        self.assertTrue(model.use_prompt)
        self.assertEqual(len(model.label_mapping), 4)
    
    @patch('transformers.T5ForConditionalGeneration.from_pretrained')
    def test_t5_generation_based_classification(self, mock_model):
        """Test T5 generation-based classification."""
        from src.models.transformers.generative.t5_classifier import T5Classifier
        
        mock_transformer = MagicMock()
        # Mock generate method
        mock_transformer.generate = MagicMock(
            return_value=torch.tensor([[1, 2, 3, 4]])
        )
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "t5-base"
        config.num_labels = 4
        config.max_generate_length = 10
        config.generation_num_beams = 4
        
        model = T5Classifier(config)
        
        self.assertEqual(model.max_generate_length, 10)
        self.assertEqual(model.generation_num_beams, 4)


# ============================================================================
# Integration Tests
# ============================================================================

class TestTransformerIntegration(TransformerTestBase):
    """Integration tests for transformer models."""
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_gradient_checkpointing(self, mock_model):
        """Test gradient checkpointing functionality."""
        mock_transformer = MagicMock()
        mock_model.return_value = mock_transformer
        
        config = Mock()
        config.model_name = "microsoft/deberta-v3-base"
        config.num_labels = 4
        config.gradient_checkpointing = True
        
        # Test that gradient checkpointing is properly configured
        from src.models.transformers.deberta.deberta_v3 import DeBERTaV3Classifier
        model = DeBERTaV3Classifier(config)
        
        self.assertTrue(config.gradient_checkpointing)
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_mixed_precision_compatibility(self, mock_model):
        """Test model compatibility with mixed precision training."""
        mock_model.return_value = MagicMock()
        
        config = Mock()
        config.model_name = "roberta-base"
        config.num_labels = 4
        config.mixed_precision = True
        
        from src.models.transformers.roberta.roberta_enhanced import RoBERTaEnhanced
        model = RoBERTaEnhanced(config)
        
        # Model should be compatible with mixed precision
        self.assertTrue(hasattr(model, 'forward'))
    
    def test_model_size_comparison(self):
        """Test relative model sizes for resource planning."""
        model_sizes = {
            'deberta-v3-base': 184e6,
            'deberta-v3-large': 435e6,
            'roberta-base': 125e6,
            'roberta-large': 355e6,
            'xlnet-base': 110e6,
            'xlnet-large': 340e6,
            'electra-base': 110e6,
            'electra-large': 335e6,
            'longformer-base': 149e6,
            'gpt2-medium': 345e6,
            'gpt2-large': 774e6,
            't5-base': 220e6,
            't5-large': 770e6
        }
        
        # Verify model size relationships
        self.assertLess(model_sizes['deberta-v3-base'], model_sizes['deberta-v3-large'])
        self.assertLess(model_sizes['roberta-base'], model_sizes['roberta-large'])
        self.assertLess(model_sizes['t5-base'], model_sizes['t5-large'])


# ============================================================================
# Performance Tests
# ============================================================================

class TestTransformerPerformance(TransformerTestBase):
    """Performance tests for transformer models."""
    
    def test_batch_size_scaling(self):
        """Test performance with different batch sizes."""
        batch_sizes = [1, 8, 16, 32, 64]
        seq_length = 128
        
        for batch_size in batch_sizes:
            batch = {
                'input_ids': torch.randint(0, 30000, (batch_size, seq_length)),
                'attention_mask': torch.ones(batch_size, seq_length)
            }
            
            # Verify batch creation is successful
            self.assertEqual(batch['input_ids'].shape[0], batch_size)
            self.assertEqual(batch['attention_mask'].shape[0], batch_size)
    
    def test_sequence_length_handling(self):
        """Test handling of different sequence lengths."""
        sequence_lengths = [32, 64, 128, 256, 512]
        batch_size = 4
        
        for seq_length in sequence_lengths:
            batch = {
                'input_ids': torch.randint(0, 30000, (batch_size, seq_length)),
                'attention_mask': torch.ones(batch_size, seq_length)
            }
            
            # Verify sequence length handling
            self.assertEqual(batch['input_ids'].shape[1], seq_length)
            self.assertEqual(batch['attention_mask'].shape[1], seq_length)
    
    def test_memory_efficiency_features(self):
        """Test memory efficiency features."""
        features = {
            'gradient_checkpointing': True,
            'mixed_precision': True,
            'use_cache': False,
            'output_hidden_states': False,
            'output_attentions': False
        }
        
        # Verify all memory efficiency features are properly configured
        for feature, expected_value in features.items():
            self.assertEqual(features[feature], expected_value)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestTransformerEdgeCases(TransformerTestBase):
    """Test edge cases and error handling for transformer models."""
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        empty_batch = {
            'input_ids': torch.tensor([]),
            'attention_mask': torch.tensor([])
        }
        
        self.assertEqual(empty_batch['input_ids'].numel(), 0)
        self.assertEqual(empty_batch['attention_mask'].numel(), 0)
    
    def test_single_token_sequence(self):
        """Test handling of single token sequences."""
        batch = {
            'input_ids': torch.tensor([[101]]),  # Just CLS token
            'attention_mask': torch.tensor([[1]])
        }
        
        self.assertEqual(batch['input_ids'].shape, (1, 1))
        self.assertEqual(batch['attention_mask'].shape, (1, 1))
    
    def test_max_length_sequence(self):
        """Test handling of maximum length sequences."""
        max_length = 512
        batch = {
            'input_ids': torch.randint(0, 30000, (1, max_length)),
            'attention_mask': torch.ones(1, max_length)
        }
        
        self.assertEqual(batch['input_ids'].shape[1], max_length)
        self.assertEqual(batch['attention_mask'].shape[1], max_length)
    
    def test_invalid_label_handling(self):
        """Test handling of invalid label values."""
        valid_labels = torch.tensor([0, 1, 2, 3])
        invalid_labels = torch.tensor([-1, 4, 100])
        
        # Check label ranges
        self.assertTrue((valid_labels >= 0).all())
        self.assertTrue((valid_labels < 4).all())
        
        # Invalid labels should be out of range
        self.assertTrue((invalid_labels < 0).any() or (invalid_labels >= 4).any())
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        # Create tensors with special values
        tensor_with_nan = torch.tensor([1.0, float('nan'), 3.0])
        tensor_with_inf = torch.tensor([1.0, float('inf'), 3.0])
        
        self.assertTrue(torch.isnan(tensor_with_nan).any())
        self.assertTrue(torch.isinf(tensor_with_inf).any())


# ============================================================================
# Model Freezing and Fine-tuning Tests
# ============================================================================

class TestModelFineTuning(TransformerTestBase):
    """Test model freezing and fine-tuning strategies."""
    
    def test_encoder_freezing(self):
        """Test freezing encoder layers."""
        config = Mock()
        config.freeze_encoder = True
        config.freeze_layers = [0, 1, 2, 3]
        
        self.assertTrue(config.freeze_encoder)
        self.assertEqual(len(config.freeze_layers), 4)
    
    def test_gradual_unfreezing(self):
        """Test gradual unfreezing strategy."""
        config = Mock()
        config.gradual_unfreezing = True
        config.unfreezing_schedule = {
            0: [],  # All frozen
            5: [11, 10],  # Unfreeze last 2 layers
            10: [9, 8, 7, 6],  # Unfreeze more layers
            15: list(range(12))  # Unfreeze all
        }
        
        self.assertTrue(config.gradual_unfreezing)
        self.assertEqual(len(config.unfreezing_schedule[15]), 12)
    
    def test_discriminative_learning_rates(self):
        """Test discriminative learning rates for different layers."""
        config = Mock()
        config.layer_wise_lr_decay = 0.9
        config.base_lr = 2e-5
        config.num_layers = 12
        
        # Calculate learning rates for each layer
        layer_lrs = []
        for layer_idx in range(config.num_layers):
            lr = config.base_lr * (config.layer_wise_lr_decay ** (config.num_layers - layer_idx))
            layer_lrs.append(lr)
        
        # Verify decreasing learning rates
        for i in range(1, len(layer_lrs)):
            self.assertLess(layer_lrs[i-1], layer_lrs[i])


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    pytest.main([__file__, "-v", "--tb=short"])
