"""
Feature Distillation Implementation for AG News Text Classification
====================================================================

This module implements feature-based knowledge distillation for transferring
intermediate representations from teacher to student models.

Mathematical Foundation:
------------------------
Feature Distillation Loss: L = α * L_CE + β * L_feature
where L_feature = MSE(f_s(x), f_t(x)) or other distance metrics

References:
- Romero et al. (2015): "FitNets: Hints for Thin Deep Nets"
- Zagoruyko & Komodakis (2017): "Paying More Attention to Attention"
- Tian et al. (2020): "Contrastive Representation Distillation"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureDistillConfig:
    """Configuration for feature distillation."""
    
    # Distillation weights
    task_weight: float = 0.5
    feature_weight: float = 0.5
    attention_weight: float = 0.1
    
    # Feature matching
    feature_layers: List[str] = None
    use_adaptation: bool = True
    distance_metric: str = 'mse'  # mse, cosine, l1
    
    # Attention transfer
    use_attention_transfer: bool = True
    attention_type: str = 'spatial'  # spatial, channel
    
    # Advanced options
    use_fsp: bool = False  # Flow of Solution Procedure
    use_rkd: bool = False  # Relational Knowledge Distillation
    temperature: float = 4.0


class FeatureDistillation(nn.Module):
    """
    Feature-based knowledge distillation.
    
    Transfers intermediate representations and attention maps
    from teacher to student models.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[FeatureDistillConfig] = None
    ):
        """
        Initialize feature distillation.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            config: Distillation configuration
        """
        super().__init__()
        
        self.teacher = teacher_model
        self.student = student_model
        self.config = config or FeatureDistillConfig()
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Setup feature extractors
        self.teacher_features = {}
        self.student_features = {}
        self._register_hooks()
        
        # Feature adaptation layers
        if self.config.use_adaptation:
            self.adaptation_layers = self._create_adaptation_layers()
    
    def _register_hooks(self):
        """Register forward hooks to extract features."""
        def get_activation(name, storage):
            def hook(model, input, output):
                storage[name] = output
            return hook
        
        # Register hooks for specified layers
        if self.config.feature_layers:
            for layer_name in self.config.feature_layers:
                # Teacher hooks
                teacher_layer = self._get_layer(self.teacher, layer_name)
                if teacher_layer:
                    teacher_layer.register_forward_hook(
                        get_activation(layer_name, self.teacher_features)
                    )
                
                # Student hooks
                student_layer = self._get_layer(self.student, layer_name)
                if student_layer:
                    student_layer.register_forward_hook(
                        get_activation(layer_name, self.student_features)
                    )
    
    def _get_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from model."""
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def _create_adaptation_layers(self) -> nn.ModuleDict:
        """Create adaptation layers for dimension matching."""
        adaptation_layers = nn.ModuleDict()
        
        # This is a simplified version - in practice, you'd need to
        # determine dimensions dynamically
        for layer_name in self.config.feature_layers or []:
            adaptation_layers[layer_name] = nn.Linear(768, 768)
        
        return adaptation_layers
    
    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature distillation loss.
        
        Args:
            inputs: Input tensors
            labels: Ground truth labels
            
        Returns:
            Dictionary of losses
        """
        # Clear feature storage
        self.teacher_features.clear()
        self.student_features.clear()
        
        # Forward pass through both models
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        student_outputs = self.student(inputs)
        
        losses = {}
        
        # Task loss
        if labels is not None:
            losses['task_loss'] = F.cross_entropy(student_outputs, labels)
        
        # Feature matching loss
        if self.config.feature_layers:
            losses['feature_loss'] = self._compute_feature_loss()
        
        # Attention transfer loss
        if self.config.use_attention_transfer:
            losses['attention_loss'] = self._compute_attention_loss()
        
        # FSP loss
        if self.config.use_fsp:
            losses['fsp_loss'] = self._compute_fsp_loss()
        
        # RKD loss
        if self.config.use_rkd:
            losses['rkd_loss'] = self._compute_rkd_loss()
        
        # Combine losses
        total_loss = torch.zeros(1, device=inputs.device)
        if 'task_loss' in losses:
            total_loss = total_loss + self.config.task_weight * losses['task_loss']
        if 'feature_loss' in losses:
            total_loss = total_loss + self.config.feature_weight * losses['feature_loss']
        if 'attention_loss' in losses:
            total_loss = total_loss + self.config.attention_weight * losses['attention_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_feature_loss(self) -> torch.Tensor:
        """Compute feature matching loss."""
        loss = 0.0
        num_layers = 0
        
        for layer_name in self.config.feature_layers or []:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]
                
                # Apply adaptation if needed
                if self.config.use_adaptation and layer_name in self.adaptation_layers:
                    student_feat = self.adaptation_layers[layer_name](student_feat)
                
                # Compute distance
                if self.config.distance_metric == 'mse':
                    layer_loss = F.mse_loss(student_feat, teacher_feat)
                elif self.config.distance_metric == 'cosine':
                    layer_loss = 1 - F.cosine_similarity(
                        student_feat.view(student_feat.size(0), -1),
                        teacher_feat.view(teacher_feat.size(0), -1)
                    ).mean()
                elif self.config.distance_metric == 'l1':
                    layer_loss = F.l1_loss(student_feat, teacher_feat)
                else:
                    raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
                
                loss = loss + layer_loss
                num_layers += 1
        
        if num_layers > 0:
            loss = loss / num_layers
        
        return loss
    
    def _compute_attention_loss(self) -> torch.Tensor:
        """Compute attention transfer loss."""
        loss = 0.0
        num_pairs = 0
        
        for layer_name in self.config.feature_layers or []:
            if layer_name in self.teacher_features and layer_name in self.student_features:
                teacher_feat = self.teacher_features[layer_name]
                student_feat = self.student_features[layer_name]
                
                # Compute attention maps
                if self.config.attention_type == 'spatial':
                    teacher_att = self._spatial_attention(teacher_feat)
                    student_att = self._spatial_attention(student_feat)
                elif self.config.attention_type == 'channel':
                    teacher_att = self._channel_attention(teacher_feat)
                    student_att = self._channel_attention(student_feat)
                else:
                    continue
                
                # Attention transfer loss
                loss = loss + F.mse_loss(student_att, teacher_att)
                num_pairs += 1
        
        if num_pairs > 0:
            loss = loss / num_pairs
        
        return loss
    
    def _spatial_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention map."""
        # Sum across channels and normalize
        if len(features.shape) == 3:  # [batch, seq_len, hidden]
            attention = features.pow(2).mean(dim=-1)
        else:
            attention = features.pow(2).sum(dim=1)
        
        attention = F.normalize(attention.view(attention.size(0), -1), dim=1)
        return attention
    
    def _channel_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Compute channel attention map."""
        if len(features.shape) == 3:  # [batch, seq_len, hidden]
            attention = features.pow(2).mean(dim=1)
        else:
            attention = features.pow(2).mean(dim=[2, 3]) if len(features.shape) == 4 else features
        
        attention = F.normalize(attention, dim=1)
        return attention
    
    def _compute_fsp_loss(self) -> torch.Tensor:
        """
        Compute Flow of Solution Procedure (FSP) loss.
        
        Reference:
        Yim et al. (2017): "A Gift from Knowledge Distillation"
        """
        loss = 0.0
        
        # FSP matrix computation for consecutive layers
        layer_names = self.config.feature_layers or []
        for i in range(len(layer_names) - 1):
            if layer_names[i] in self.teacher_features and \
               layer_names[i+1] in self.teacher_features:
                # Teacher FSP
                teacher_fsp = self._compute_fsp_matrix(
                    self.teacher_features[layer_names[i]],
                    self.teacher_features[layer_names[i+1]]
                )
                
                # Student FSP
                student_fsp = self._compute_fsp_matrix(
                    self.student_features[layer_names[i]],
                    self.student_features[layer_names[i+1]]
                )
                
                loss = loss + F.mse_loss(student_fsp, teacher_fsp)
        
        return loss
    
    def _compute_fsp_matrix(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> torch.Tensor:
        """Compute FSP matrix between two feature maps."""
        if len(feat1.shape) == 3:  # Transformer features
            feat1 = feat1.mean(dim=1)
            feat2 = feat2.mean(dim=1)
        
        # Flatten features
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        
        # Compute FSP matrix
        fsp = torch.matmul(feat1.unsqueeze(2), feat2.unsqueeze(1))
        return fsp
    
    def _compute_rkd_loss(self) -> torch.Tensor:
        """
        Compute Relational Knowledge Distillation loss.
        
        Reference:
        Park et al. (2019): "Relational Knowledge Distillation"
        """
        # Get final embeddings
        if 'output' in self.teacher_features and 'output' in self.student_features:
            teacher_embed = self.teacher_features['output']
            student_embed = self.student_features['output']
            
            # Distance-wise RKD
            teacher_dist = self._pdist(teacher_embed)
            student_dist = self._pdist(student_embed)
            
            distance_loss = F.smooth_l1_loss(student_dist, teacher_dist)
            
            # Angle-wise RKD
            teacher_angle = self._compute_angles(teacher_embed)
            student_angle = self._compute_angles(student_embed)
            
            angle_loss = F.smooth_l1_loss(student_angle, teacher_angle)
            
            return distance_loss + angle_loss
        
        return torch.tensor(0.0)
    
    def _pdist(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        n = embeddings.size(0)
        dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Get upper triangular part
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        return dist[mask]
    
    def _compute_angles(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise angles."""
        normalized = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(normalized, normalized.T)
        
        n = embeddings.size(0)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        
        angles = torch.acos(torch.clamp(similarity[mask], -1, 1))
        return angles
