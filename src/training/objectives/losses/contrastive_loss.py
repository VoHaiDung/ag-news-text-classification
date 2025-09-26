"""
Contrastive Loss Implementation for AG News Text Classification
================================================================

This module implements various contrastive learning objectives for
representation learning and improved generalization.

Mathematical Foundation:
------------------------
SimCLR Loss (NT-Xent):
L = -log(exp(sim(zi, zj)/τ) / Σ_k exp(sim(zi, zk)/τ))

Supervised Contrastive Loss:
L = -1/|P(i)| Σ_p∈P(i) log(exp(sim(zi, zp)/τ) / Σ_a exp(sim(zi, za)/τ))

References:
- Chen et al. (2020): "A Simple Framework for Contrastive Learning"
- Khosla et al. (2020): "Supervised Contrastive Learning"
- He et al. (2020): "Momentum Contrast for Unsupervised Visual Representation Learning"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Used in SimCLR for self-supervised contrastive learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True,
        contrast_mode: str = 'all'  # all, one
    ):
        """
        Initialize NT-Xent Loss.
        
        Args:
            temperature: Temperature parameter for scaling
            normalize: Whether to normalize embeddings
            contrast_mode: How to select negative samples
        """
        super().__init__()
        
        self.temperature = temperature
        self.normalize = normalize
        self.contrast_mode = contrast_mode
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            features: Embeddings of shape (N, D) or (N, 2, D) for pairs
            labels: Optional labels for supervised contrastive
            mask: Optional mask for valid samples
            
        Returns:
            Computed loss value
        """
        device = features.device
        
        # Handle paired features
        if len(features.shape) == 3:
            # Reshape (N, 2, D) -> (2N, D)
            batch_size = features.shape[0]
            features = features.view(-1, features.shape[-1])
            
            # Create labels for positive pairs
            if labels is None:
                labels = torch.arange(batch_size).repeat(2).to(device)
        else:
            batch_size = features.shape[0] // 2
            if labels is None:
                labels = torch.arange(batch_size).repeat(2).to(device)
        
        # Normalize features
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-similarity
        batch_size_full = features.shape[0]
        mask_positive.fill_diagonal_(0)
        
        # Create mask for negative pairs
        mask_negative = 1 - mask_positive
        
        # For numerical stability
        max_sim = torch.max(similarity_matrix, dim=1, keepdim=True)[0]
        similarity_matrix = similarity_matrix - max_sim.detach()
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        
        if self.contrast_mode == 'all':
            # All negatives
            log_prob = similarity_matrix - torch.log(
                (exp_sim * mask_negative).sum(dim=1, keepdim=True) + 1e-12
            )
        else:
            # One negative (next in batch)
            log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        
        # Mean of positive pairs
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + 1e-12)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for leveraging label information.
    
    Extends contrastive learning to supervised setting by contrasting
    samples from the same class as positives.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        normalize: bool = True,
        reduction: str = 'mean'
    ):
        """
        Initialize Supervised Contrastive Loss.
        
        Args:
            temperature: Temperature for similarity scaling
            base_temperature: Base temperature for normalization
            normalize: Whether to normalize features
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.normalize = normalize
        self.reduction = reduction
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings (N, D) or (N, M, D) for M views
            labels: Ground truth labels (N,)
            mask: Optional mask for valid samples
            
        Returns:
            Computed loss value
        """
        device = features.device
        
        # Handle multiple views
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        num_views = features.shape[1]
        
        # Normalize features
        if self.normalize:
            features = F.normalize(features, p=2, dim=2)
        
        # Flatten views
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        labels = labels.repeat(num_views)
        
        # Compute similarity
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create masks
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        mask_positive.fill_diagonal_(0)
        
        # Apply provided mask if any
        if mask is not None:
            mask = mask.repeat(num_views)
            mask = mask.view(-1, 1) * mask.view(1, -1)
            mask_positive = mask_positive * mask
        
        # Compute log probabilities
        exp_similarity = torch.exp(similarity)
        
        # Mask out self-similarity
        logits_mask = torch.ones_like(exp_similarity)
        logits_mask.fill_diagonal_(0)
        
        # Compute log_prob
        exp_sum = (exp_similarity * logits_mask).sum(dim=1, keepdim=True)
        log_prob = similarity - torch.log(exp_sum + 1e-12)
        
        # Compute mean of log-likelihood over positive samples
        num_positives = mask_positive.sum(dim=1)
        valid_samples = num_positives > 0
        
        if valid_samples.any():
            log_prob_positive = (mask_positive * log_prob)[valid_samples]
            num_positives = num_positives[valid_samples]
            
            loss = -(self.temperature / self.base_temperature) * \
                   (log_prob_positive.sum(dim=1) / num_positives)
            
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss


class TripletContrastiveLoss(nn.Module):
    """
    Triplet-based Contrastive Loss.
    
    Combines triplet loss with contrastive learning for better
    representation learning.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1,
        normalize: bool = True,
        distance: str = 'cosine'  # cosine, euclidean
    ):
        """
        Initialize Triplet Contrastive Loss.
        
        Args:
            margin: Margin for triplet loss
            temperature: Temperature for contrastive scaling
            normalize: Whether to normalize embeddings
            distance: Distance metric to use
        """
        super().__init__()
        
        self.margin = margin
        self.temperature = temperature
        self.normalize = normalize
        self.distance = distance
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet contrastive loss.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
            
        Returns:
            Computed loss value
        """
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        if self.distance == 'cosine':
            pos_similarity = F.cosine_similarity(anchor, positive)
            neg_similarity = F.cosine_similarity(anchor, negative)
            
            # Convert to distance (1 - similarity)
            pos_distance = 1 - pos_similarity
            neg_distance = 1 - neg_similarity
        else:  # euclidean
            pos_distance = F.pairwise_distance(anchor, positive, p=2)
            neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        triplet_loss = F.relu(pos_distance - neg_distance + self.margin)
        
        # Contrastive component
        anchor_norm = anchor / self.temperature
        pos_norm = positive / self.temperature
        neg_norm = negative / self.temperature
        
        pos_logit = torch.sum(anchor_norm * pos_norm, dim=1)
        neg_logit = torch.sum(anchor_norm * neg_norm, dim=1)
        
        contrastive_loss = -torch.log(
            torch.exp(pos_logit) / (torch.exp(pos_logit) + torch.exp(neg_logit) + 1e-12)
        )
        
        # Combine losses
        total_loss = triplet_loss + contrastive_loss
        
        return total_loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive predictive coding.
    
    Maximizes mutual information between different views of data.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        negative_samples: int = 256,
        normalize: bool = True
    ):
        """
        Initialize InfoNCE Loss.
        
        Args:
            temperature: Temperature parameter
            negative_samples: Number of negative samples
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        
        self.temperature = temperature
        self.negative_samples = negative_samples
        self.normalize = normalize
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings (N, D)
            positive_key: Positive key embeddings (N, D)
            negative_keys: Negative key embeddings (N, K, D) or None
            
        Returns:
            Computed loss value
        """
        batch_size = query.shape[0]
        
        # Normalize if required
        if self.normalize:
            query = F.normalize(query, p=2, dim=1)
            positive_key = F.normalize(positive_key, p=2, dim=1)
            if negative_keys is not None:
                negative_keys = F.normalize(negative_keys, p=2, dim=-1)
        
        # Positive logits
        positive_logits = torch.sum(query * positive_key, dim=1, keepdim=True)
        positive_logits = positive_logits / self.temperature
        
        # Negative logits
        if negative_keys is None:
            # Use other samples in batch as negatives
            negative_logits = torch.matmul(query, positive_key.T) / self.temperature
            
            # Mask out self-similarity
            mask = torch.eye(batch_size, dtype=torch.bool, device=query.device)
            negative_logits.masked_fill_(mask, float('-inf'))
        else:
            # Use provided negative keys
            negative_logits = torch.matmul(
                query.unsqueeze(1), negative_keys.transpose(1, 2)
            ).squeeze(1) / self.temperature
        
        # Concatenate positive and negative logits
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        
        # Labels: first position is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
