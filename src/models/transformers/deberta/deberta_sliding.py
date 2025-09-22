"""
Sliding Window DeBERTa for Long Document Classification
========================================================

Implementation of DeBERTa with sliding window mechanism for processing documents
longer than the model's maximum sequence length, based on:
- He et al. (2021): "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
- Beltagy et al. (2020): "Longformer: The Long-Document Transformer"
- Pappagari et al. (2019): "Hierarchical Transformers for Long Document Classification"

The sliding window approach splits long documents into overlapping segments,
processes each independently, and aggregates the results.

Mathematical Foundation:
Document D = [w₁, w₂, ..., wₙ] is split into windows W₁, W₂, ..., Wₖ
where Wᵢ = [w_{i*s}, ..., w_{i*s+l}] with stride s and length l
Final prediction: P(y|D) = Aggregate({P(y|Wᵢ)})

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DebertaV2Model,
    DebertaV2Config,
    DebertaV2Tokenizer,
    DebertaV2PreTrainedModel
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingStrategy, create_pooling_layer
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window DeBERTa"""
    
    # Model configuration
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 4
    
    # Window configuration
    window_size: int = 256  # Size of each window
    stride: int = 128  # Stride between windows
    max_windows: int = 16  # Maximum number of windows
    
    # Aggregation strategy
    aggregation: str = "weighted_avg"  # "mean", "max", "weighted_avg", "attention"
    position_weighting: bool = True  # Weight windows by position
    
    # Pooling configuration  
    pooling_strategy: str = "cls"  # "cls", "mean", "max", "attention"
    
    # Training configuration
    freeze_embeddings: bool = False
    freeze_layers: int = 0
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False
    
    # Window interaction
    use_cross_window_attention: bool = False
    share_window_encoder: bool = True  # Share encoder across windows
    
    # Performance optimization
    process_windows_parallel: bool = True
    cache_window_embeddings: bool = False
    
    # Advanced features
    use_confidence_weighting: bool = True
    confidence_temperature: float = 1.0
    dynamic_window_size: bool = False  # Adjust window size based on content


class WindowAggregator(nn.Module):
    """
    Aggregates predictions from multiple windows.
    
    Implements various aggregation strategies including:
    - Simple averaging
    - Weighted averaging based on position or confidence
    - Attention-based aggregation
    - Hierarchical aggregation
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        config: SlidingWindowConfig
    ):
        """
        Initialize window aggregator.
        
        Args:
            hidden_size: Hidden dimension size
            num_labels: Number of output labels
            config: Sliding window configuration
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.config = config
        
        if config.aggregation == "attention":
            # Attention-based aggregation
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True
            )
            
            # Query vector for aggregation
            self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
            
        elif config.aggregation == "weighted_avg":
            # Learnable position weights
            self.position_weights = nn.Parameter(torch.ones(config.max_windows))
            
            if config.use_confidence_weighting:
                # Network to predict confidence from hidden states
                self.confidence_predictor = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(hidden_size // 2, 1),
                    nn.Sigmoid()
                )
        
        # Optional cross-window interaction
        if config.use_cross_window_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True
            )
            
        logger.info(f"Initialized WindowAggregator with {config.aggregation} strategy")
    
    def forward(
        self,
        window_outputs: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Aggregate window outputs.
        
        Args:
            window_outputs: Window representations [batch_size, num_windows, hidden_size]
            window_mask: Valid windows mask [batch_size, num_windows]
            return_weights: Whether to return aggregation weights
            
        Returns:
            Aggregated output and optionally weights
        """
        batch_size, num_windows, hidden_size = window_outputs.shape
        
        # Apply cross-window attention if configured
        if self.config.use_cross_window_attention:
            window_outputs = self._apply_cross_attention(window_outputs, window_mask)
        
        # Aggregate based on strategy
        if self.config.aggregation == "mean":
            # Simple averaging
            if window_mask is not None:
                mask_expanded = window_mask.unsqueeze(-1).expand_as(window_outputs)
                sum_outputs = (window_outputs * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                aggregated = sum_outputs / count
            else:
                aggregated = window_outputs.mean(dim=1)
            weights = None
            
        elif self.config.aggregation == "max":
            # Max pooling
            if window_mask is not None:
                window_outputs = window_outputs.masked_fill(
                    ~window_mask.unsqueeze(-1), float('-inf')
                )
            aggregated = window_outputs.max(dim=1)[0]
            weights = None
            
        elif self.config.aggregation == "weighted_avg":
            # Weighted averaging
            weights = self._compute_weights(window_outputs, window_mask)
            weights_expanded = weights.unsqueeze(-1)
            aggregated = (window_outputs * weights_expanded).sum(dim=1)
            
        elif self.config.aggregation == "attention":
            # Attention-based aggregation
            query = self.query.expand(batch_size, -1, -1)
            
            if window_mask is not None:
                attended, weights = self.attention(
                    query, window_outputs, window_outputs,
                    key_padding_mask=~window_mask
                )
            else:
                attended, weights = self.attention(
                    query, window_outputs, window_outputs
                )
            
            aggregated = attended.squeeze(1)
            weights = weights.squeeze(1)
            
        else:
            # Default to mean
            aggregated = window_outputs.mean(dim=1)
            weights = None
        
        if return_weights and weights is not None:
            return aggregated, weights
        return aggregated
    
    def _compute_weights(
        self,
        window_outputs: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute aggregation weights.
        
        Args:
            window_outputs: Window representations
            window_mask: Valid windows mask
            
        Returns:
            Normalized weights [batch_size, num_windows]
        """
        batch_size, num_windows = window_outputs.shape[:2]
        
        # Start with position weights
        if self.config.position_weighting:
            weights = self.position_weights[:num_windows].unsqueeze(0).expand(batch_size, -1)
        else:
            weights = torch.ones(batch_size, num_windows, device=window_outputs.device)
        
        # Add confidence weighting if configured
        if self.config.use_confidence_weighting and hasattr(self, 'confidence_predictor'):
            confidence = self.confidence_predictor(window_outputs).squeeze(-1)
            confidence = confidence / self.config.confidence_temperature
            weights = weights * confidence
        
        # Apply mask
        if window_mask is not None:
            weights = weights.masked_fill(~window_mask, 0)
        
        # Normalize
        weights = F.softmax(weights, dim=-1)
        
        return weights
    
    def _apply_cross_attention(
        self,
        window_outputs: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention between windows.
        
        Args:
            window_outputs: Window representations
            window_mask: Valid windows mask
            
        Returns:
            Updated window representations
        """
        if window_mask is not None:
            attended, _ = self.cross_attention(
                window_outputs, window_outputs, window_outputs,
                key_padding_mask=~window_mask
            )
        else:
            attended, _ = self.cross_attention(
                window_outputs, window_outputs, window_outputs
            )
        
        # Residual connection
        window_outputs = window_outputs + attended
        
        return window_outputs


@MODELS.register("deberta_sliding")
class SlidingWindowDeBERTa(AGNewsBaseModel):
    """
    DeBERTa with sliding window for long documents.
    
    Processes long documents by:
    1. Splitting into overlapping windows
    2. Encoding each window with DeBERTa
    3. Aggregating window representations
    4. Making final prediction
    
    This approach enables processing of documents that exceed
    the model's maximum sequence length while maintaining
    context through overlapping windows.
    """
    
    def __init__(self, config: Optional[SlidingWindowConfig] = None):
        """
        Initialize sliding window DeBERTa.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or SlidingWindowConfig()
        
        # Initialize tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.model_name)
        
        # Initialize DeBERTa encoder
        self._init_encoder()
        
        # Initialize pooling layer
        self.pooler = create_pooling_layer(
            PoolingStrategy(self.config.pooling_strategy),
            self.hidden_size
        )
        
        # Initialize aggregator
        self.aggregator = WindowAggregator(
            self.hidden_size,
            self.config.num_labels,
            self.config
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_size, self.config.num_labels)
        )
        
        # Apply freezing if configured
        self._apply_freezing()
        
        # Cache for window embeddings
        if self.config.cache_window_embeddings:
            self.window_cache = {}
        
        logger.info(
            f"Initialized SlidingWindowDeBERTa: "
            f"window_size={config.window_size}, stride={config.stride}"
        )
    
    def _init_encoder(self):
        """Initialize DeBERTa encoder"""
        # Load configuration
        deberta_config = DebertaV2Config.from_pretrained(self.config.model_name)
        
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            deberta_config.gradient_checkpointing = True
        
        # Load model
        self.deberta = DebertaV2Model.from_pretrained(
            self.config.model_name,
            config=deberta_config
        )
        
        self.hidden_size = self.deberta.config.hidden_size
    
    def _apply_freezing(self):
        """Apply parameter freezing"""
        if self.config.freeze_embeddings:
            for param in self.deberta.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze embedding parameters")
        
        if self.config.freeze_layers > 0:
            layers_to_freeze = self.deberta.encoder.layer[:self.config.freeze_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Froze {self.config.freeze_layers} encoder layers")
    
    def _create_windows(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sliding windows from input.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Tuple of (window_ids, window_masks)
        """
        batch_size, seq_length = input_ids.shape
        window_size = self.config.window_size
        stride = self.config.stride
        
        # Calculate number of windows
        num_windows = min(
            (seq_length - window_size) // stride + 1,
            self.config.max_windows
        )
        
        # Create window indices
        windows_ids = []
        windows_masks = []
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = min(start_idx + window_size, seq_length)
            
            # Extract window
            window_ids = input_ids[:, start_idx:end_idx]
            window_mask = attention_mask[:, start_idx:end_idx]
            
            # Pad if necessary
            if window_ids.shape[1] < window_size:
                pad_length = window_size - window_ids.shape[1]
                window_ids = F.pad(window_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
                window_mask = F.pad(window_mask, (0, pad_length), value=0)
            
            windows_ids.append(window_ids)
            windows_masks.append(window_mask)
        
        # Stack windows
        windows_ids = torch.stack(windows_ids, dim=1)  # [batch, num_windows, window_size]
        windows_masks = torch.stack(windows_masks, dim=1)
        
        return windows_ids, windows_masks
    
    def _encode_windows(
        self,
        windows_ids: torch.Tensor,
        windows_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode each window with DeBERTa.
        
        Args:
            windows_ids: Window token IDs [batch, num_windows, window_size]
            windows_masks: Window attention masks
            
        Returns:
            Window representations [batch, num_windows, hidden_size]
        """
        batch_size, num_windows, window_size = windows_ids.shape
        
        if self.config.process_windows_parallel and self.config.share_window_encoder:
            # Process all windows in parallel
            # Reshape to [batch * num_windows, window_size]
            flat_ids = windows_ids.view(-1, window_size)
            flat_masks = windows_masks.view(-1, window_size)
            
            # Encode
            outputs = self.deberta(
                input_ids=flat_ids,
                attention_mask=flat_masks
            )
            
            # Pool
            window_reprs = self.pooler(outputs.last_hidden_state, flat_masks)
            
            # Reshape back to [batch, num_windows, hidden_size]
            window_reprs = window_reprs.view(batch_size, num_windows, self.hidden_size)
            
        else:
            # Process windows sequentially
            window_reprs = []
            
            for i in range(num_windows):
                window_ids = windows_ids[:, i, :]
                window_mask = windows_masks[:, i, :]
                
                # Check cache if enabled
                if self.config.cache_window_embeddings:
                    cache_key = (window_ids.cpu().numpy().tobytes(), window_mask.cpu().numpy().tobytes())
                    if cache_key in self.window_cache:
                        window_repr = self.window_cache[cache_key]
                    else:
                        outputs = self.deberta(
                            input_ids=window_ids,
                            attention_mask=window_mask
                        )
                        window_repr = self.pooler(outputs.last_hidden_state, window_mask)
                        self.window_cache[cache_key] = window_repr
                else:
                    outputs = self.deberta(
                        input_ids=window_ids,
                        attention_mask=window_mask
                    )
                    window_repr = self.pooler(outputs.last_hidden_state, window_mask)
                
                window_reprs.append(window_repr)
            
            window_reprs = torch.stack(window_reprs, dim=1)
        
        return window_reprs
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_window_weights: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through sliding window DeBERTa.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            labels: Target labels
            return_window_weights: Whether to return window aggregation weights
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with predictions
        """
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create sliding windows
        windows_ids, windows_masks = self._create_windows(input_ids, attention_mask)
        
        # Encode windows
        window_representations = self._encode_windows(windows_ids, windows_masks)
        
        # Create window validity mask
        window_valid_mask = windows_masks.sum(dim=2) > 0
        
        # Aggregate window representations
        if return_window_weights:
            aggregated, window_weights = self.aggregator(
                window_representations,
                window_valid_mask,
                return_weights=True
            )
        else:
            aggregated = self.aggregator(
                window_representations,
                window_valid_mask,
                return_weights=False
            )
            window_weights = None
        
        # Classification
        logits = self.classifier(aggregated)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        # Prepare outputs
        outputs = ModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=aggregated,
            metadata={
                'num_windows': int(window_valid_mask.sum(dim=1).mean().item()),
                'window_size': self.config.window_size,
                'stride': self.config.stride,
                'aggregation': self.config.aggregation
            }
        )
        
        if return_window_weights and window_weights is not None:
            outputs.attentions = window_weights
            outputs.metadata['window_weights'] = window_weights.cpu().numpy()
        
        return outputs
    
    def get_window_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get importance scores for each window.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Window importance scores
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask,
                return_window_weights=True
            )
            
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                return outputs.attentions
            else:
                # Return uniform weights if not available
                batch_size = input_ids.shape[0]
                num_windows = self.config.max_windows
                return torch.ones(batch_size, num_windows) / num_windows
