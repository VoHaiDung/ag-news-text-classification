"""
Dynamic Blending Ensemble for AG News Classification
=====================================================

Implementation of dynamic blending that adapts combination weights based on
input characteristics, following:
- Wolpert (1992): "Stacked Generalization"
- Džeroski & Ženko (2004): "Is Combining Classifiers with Stacking Better than Selecting the Best One?"
- Rooney et al. (2004): "Dynamic Integration of Regression Models"

Dynamic blending learns instance-specific weights for combining base models,
allowing the ensemble to adapt to different regions of the input space.

Mathematical Foundation:
w(x) = g(f(x)) where g is the gating network and f extracts features from x
H(x) = Σ w_i(x) * h_i(x) where h_i are base model predictions

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.ensemble.base_ensemble import BaseEnsemble
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class GatingNetwork(nn.Module):
    """
    Gating network that produces instance-specific weights.
    
    The network learns to predict optimal combination weights
    based on input features or intermediate representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_models: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        use_attention: bool = False
    ):
        """
        Initialize gating network.
        
        Args:
            input_dim: Input feature dimension
            num_models: Number of models to weight
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_attention: Use attention mechanism
        """
        super().__init__()
        
        self.num_models = num_models
        self.use_attention = use_attention
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        if use_attention:
            # Attention-based gating
            self.attention = nn.MultiheadAttention(
                embed_dim=prev_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.model_embeddings = nn.Parameter(
                torch.randn(num_models, prev_dim)
            )
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_models)
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))
        
        logger.info(f"Initialized gating network for {num_models} models")
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate instance-specific weights.
        
        Args:
            features: Input features [batch_size, input_dim]
            return_attention: Return attention weights
            
        Returns:
            Model weights [batch_size, num_models]
        """
        # Extract features
        hidden = self.feature_extractor(features)
        
        if self.use_attention:
            # Use attention to compute weights
            batch_size = features.shape[0]
            
            # Expand model embeddings for batch
            model_embeds = self.model_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Query with hidden features
            query = hidden.unsqueeze(1)  # [batch, 1, hidden]
            
            # Attention
            attended, attention_weights = self.attention(
                query, model_embeds, model_embeds
            )
            
            # Flatten for output
            hidden = attended.squeeze(1)
        else:
            attention_weights = None
        
        # Generate weights
        logits = self.output_layer(hidden)
        
        # Apply temperature and softmax
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        if return_attention and attention_weights is not None:
            return weights, attention_weights
        
        return weights


class FeatureExtractor(nn.Module):
    """
    Extract features for gating network from various sources.
    
    Can extract features from:
    - Raw inputs
    - Intermediate model representations
    - Prediction statistics
    """
    
    def __init__(
        self,
        extract_from: str = "predictions",  # "input", "hidden", "predictions", "combined"
        feature_dim: int = 256,
        num_classes: int = 4
    ):
        """
        Initialize feature extractor.
        
        Args:
            extract_from: Source of features
            feature_dim: Output feature dimension
            num_classes: Number of classes
        """
        super().__init__()
        
        self.extract_from = extract_from
        self.feature_dim = feature_dim
        
        if extract_from == "predictions":
            # Extract from prediction statistics
            # Features: mean, std, entropy, top-2 diff, etc.
            stats_dim = num_classes * 3 + 5  # Approximate dimension
            self.projection = nn.Linear(stats_dim, feature_dim)
            
        elif extract_from == "combined":
            # Combine multiple sources
            self.input_proj = nn.Linear(768, feature_dim // 2)  # Assuming BERT-like
            self.pred_proj = nn.Linear(num_classes * 3, feature_dim // 2)
            
        else:
            # Direct projection
            self.projection = nn.Linear(768, feature_dim)
    
    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from available sources.
        
        Args:
            inputs: Input embeddings
            hidden_states: Hidden representations
            predictions: Model predictions
            
        Returns:
            Extracted features
        """
        if self.extract_from == "predictions" and predictions is not None:
            # Extract statistics from predictions
            features = self._extract_prediction_stats(predictions)
            features = self.projection(features)
            
        elif self.extract_from == "combined":
            features_list = []
            
            if inputs is not None:
                features_list.append(self.input_proj(inputs.mean(dim=1)))
            
            if predictions is not None:
                pred_stats = self._extract_prediction_stats(predictions)
                features_list.append(self.pred_proj(pred_stats))
            
            features = torch.cat(features_list, dim=-1)
            
        else:
            # Use hidden states or inputs
            source = hidden_states if hidden_states is not None else inputs
            if source is not None:
                # Pool over sequence dimension if needed
                if source.dim() == 3:
                    source = source.mean(dim=1)
                features = self.projection(source)
            else:
                raise ValueError("No valid feature source available")
        
        return features
    
    def _extract_prediction_stats(self, predictions: torch.Tensor) -> torch.Tensor:
        """Extract statistical features from predictions"""
        # predictions shape: [n_models, batch_size, n_classes]
        
        # Mean and std across models
        mean_preds = predictions.mean(dim=0)
        std_preds = predictions.std(dim=0)
        
        # Entropy of average predictions
        entropy = -(mean_preds * torch.log(mean_preds + 1e-8)).sum(dim=-1, keepdim=True)
        
        # Maximum prediction confidence
        max_conf = mean_preds.max(dim=-1, keepdim=True)[0]
        
        # Difference between top-2 predictions
        sorted_preds = mean_preds.sort(dim=-1, descending=True)[0]
        top2_diff = (sorted_preds[:, 0] - sorted_preds[:, 1]).unsqueeze(-1)
        
        # Model disagreement (std of predictions)
        disagreement = std_preds.mean(dim=-1, keepdim=True)
        
        # Concatenate all features
        features = torch.cat([
            mean_preds.flatten(1),
            std_preds.flatten(1),
            entropy,
            max_conf,
            top2_diff,
            disagreement
        ], dim=-1)
        
        return features


@dataclass
class DynamicBlendingConfig:
    """Configuration for dynamic blending ensemble"""
    
    # Gating network configuration
    gating_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    gating_dropout: float = 0.2
    use_attention_gating: bool = False
    
    # Feature extraction
    feature_source: str = "predictions"  # "input", "hidden", "predictions", "combined"
    feature_dim: int = 256
    
    # Training configuration
    gating_lr: float = 1e-3
    gating_weight_decay: float = 1e-4
    pretrain_gating: bool = True
    pretrain_epochs: int = 5
    
    # Blending strategy
    blend_logits: bool = False  # Blend logits vs probabilities
    residual_connection: bool = False  # Add residual from best model
    
    # Regularization
    weight_entropy_reg: float = 0.01  # Encourage diverse weights
    weight_smoothness_reg: float = 0.01  # Encourage smooth weight changes
    
    # Adaptive learning
    adapt_online: bool = False  # Adapt weights during inference
    adaptation_rate: float = 0.01
    
    # Monitoring
    track_weight_distribution: bool = True
    log_interval: int = 100


@MODELS.register("dynamic_blending")
class DynamicBlendingEnsemble(BaseEnsemble):
    """
    Dynamic Blending Ensemble with Instance-Specific Weights.
    
    This ensemble learns to predict optimal combination weights for each
    input instance, allowing it to adapt to different data characteristics
    and leverage the strengths of different models in different regions
    of the input space.
    
    The gating network can use various features including:
    - Input representations
    - Model predictions
    - Uncertainty estimates
    - Hidden states
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[DynamicBlendingConfig] = None
    ):
        """
        Initialize dynamic blending ensemble.
        
        Args:
            models: List of base models
            config: Ensemble configuration
        """
        super().__init__(models)
        
        self.config = config or DynamicBlendingConfig()
        self.n_models = len(models)
        
        # Initialize components
        self._init_gating_network()
        self._init_feature_extractor()
        
        # Statistics tracking
        self.weight_history = []
        self.adaptation_stats = {
            'weight_entropy': [],
            'weight_variance': [],
            'dominant_model_freq': {}
        }
        
        logger.info(
            f"Initialized Dynamic Blending Ensemble with {self.n_models} models"
        )
    
    def _init_gating_network(self):
        """Initialize the gating network"""
        self.gating_network = GatingNetwork(
            input_dim=self.config.feature_dim,
            num_models=self.n_models,
            hidden_dims=self.config.gating_hidden_dims,
            dropout=self.config.gating_dropout,
            use_attention=self.config.use_attention_gating
        )
        
        # Separate optimizer for gating network
        self.gating_optimizer = torch.optim.AdamW(
            self.gating_network.parameters(),
            lr=self.config.gating_lr,
            weight_decay=self.config.gating_weight_decay
        )
    
    def _init_feature_extractor(self):
        """Initialize feature extractor"""
        self.feature_extractor = FeatureExtractor(
            extract_from=self.config.feature_source,
            feature_dim=self.config.feature_dim,
            num_classes=4  # AG News classes
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass with dynamic blending.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_weights: Return blending weights
            **kwargs: Additional arguments
            
        Returns:
            Dynamically blended predictions
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Collect predictions and features from all models
        all_predictions = []
        all_logits = []
        all_hidden = []
        
        for model in self.models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs = torch.ones(batch_size, 4, device=device) / 4
                    logits = torch.zeros(batch_size, 4, device=device)
                
                all_predictions.append(probs)
                all_logits.append(logits)
                
                # Collect hidden states if available
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    all_hidden.append(outputs.hidden_states)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions)  # [n_models, batch, classes]
        all_logits = torch.stack(all_logits)
        
        # Extract features for gating
        features = self.feature_extractor(
            inputs=None,  # Could pass input embeddings
            hidden_states=all_hidden[0] if all_hidden else None,
            predictions=all_predictions
        )
        
        # Get dynamic weights from gating network
        blend_weights = self.gating_network(features)  # [batch, n_models]
        
        # Apply weights to predictions
        if self.config.blend_logits:
            # Blend logits
            blend_weights = blend_weights.unsqueeze(2)  # [batch, n_models, 1]
            blended = torch.sum(
                all_logits.permute(1, 0, 2) * blend_weights,
                dim=1
            )
            ensemble_logits = blended
        else:
            # Blend probabilities
            blend_weights = blend_weights.unsqueeze(2)  # [batch, n_models, 1]
            blended = torch.sum(
                all_predictions.permute(1, 0, 2) * blend_weights,
                dim=1
            )
            ensemble_logits = torch.log(blended + 1e-8)
        
        # Optional residual connection
        if self.config.residual_connection:
            # Add residual from best average model
            best_idx = all_predictions.mean(dim=(1, 2)).argmax()
            ensemble_logits = ensemble_logits + 0.1 * all_logits[best_idx]
        
        # Calculate loss
        loss = None
        if labels is not None:
            # Main loss
            ensemble_loss = F.cross_entropy(ensemble_logits, labels)
            
            # Regularization losses
            reg_loss = 0
            
            # Entropy regularization (encourage diverse weights)
            if self.config.weight_entropy_reg > 0:
                weight_entropy = -(blend_weights * torch.log(blend_weights + 1e-8)).sum(dim=1).mean()
                reg_loss += self.config.weight_entropy_reg * (-weight_entropy)  # Maximize entropy
            
            # Smoothness regularization
            if self.config.weight_smoothness_reg > 0 and batch_size > 1:
                weight_diff = torch.diff(blend_weights, dim=0)
                smoothness_loss = torch.norm(weight_diff, p=2)
                reg_loss += self.config.weight_smoothness_reg * smoothness_loss
            
            loss = ensemble_loss + reg_loss
            
            # Update gating network if training
            if self.training and self.config.pretrain_gating:
                self.gating_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.gating_optimizer.step()
        
        # Update statistics
        self._update_statistics(blend_weights)
        
        # Prepare metadata
        metadata = {
            'blend_weights': blend_weights.squeeze(2).detach().cpu().numpy(),
            'weight_entropy': self._calculate_weight_entropy(blend_weights),
            'dominant_model': blend_weights.squeeze(2).argmax(dim=1).mode()[0].item()
        }
        
        if return_weights:
            metadata['weight_distribution'] = self._analyze_weight_distribution(blend_weights)
        
        return ModelOutputs(
            loss=loss,
            logits=ensemble_logits,
            metadata=metadata
        )
    
    def _update_statistics(self, weights: torch.Tensor):
        """Update blending statistics"""
        # Calculate entropy
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        self.adaptation_stats['weight_entropy'].append(entropy.item())
        
        # Calculate variance
        variance = weights.var(dim=1).mean()
        self.adaptation_stats['weight_variance'].append(variance.item())
        
        # Track dominant model
        dominant = weights.squeeze(2).argmax(dim=1)
        for idx in dominant.cpu().numpy():
            self.adaptation_stats['dominant_model_freq'][idx] = \
                self.adaptation_stats['dominant_model_freq'].get(idx, 0) + 1
    
    def _calculate_weight_entropy(self, weights: torch.Tensor) -> float:
        """Calculate entropy of weight distribution"""
        weights = weights.squeeze(2)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        return entropy.item()
    
    def _analyze_weight_distribution(self, weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze weight distribution patterns"""
        weights = weights.squeeze(2).detach().cpu().numpy()
        
        return {
            'mean_weights': weights.mean(axis=0).tolist(),
            'std_weights': weights.std(axis=0).tolist(),
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'weight_range': weights.max() - weights.min(),
            'effective_models': 1.0 / np.sum(weights.mean(axis=0) ** 2)  # Inverse HHI
        }
    
    def adapt_online(self, features: torch.Tensor, feedback: torch.Tensor):
        """
        Online adaptation of gating network.
        
        Args:
            features: Input features
            feedback: Feedback signal (e.g., prediction confidence)
        """
        if not self.config.adapt_online:
            return
        
        # Simple gradient update based on feedback
        weights = self.gating_network(features)
        
        # Update based on feedback (higher feedback = reinforce current weights)
        loss = -torch.mean(feedback * torch.log(weights.max(dim=1)[0]))
        
        self.gating_optimizer.zero_grad()
        loss.backward()
        
        # Scale gradients by adaptation rate
        for param in self.gating_network.parameters():
            if param.grad is not None:
                param.grad *= self.config.adaptation_rate
        
        self.gating_optimizer.step()
        
        logger.debug("Performed online adaptation of gating network")
