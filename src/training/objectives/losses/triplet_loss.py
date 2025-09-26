"""
Triplet Loss Implementation for AG News Text Classification
============================================================

This module implements triplet loss and its variants for learning
discriminative embeddings through relative distance constraints.

Mathematical Foundation:
------------------------
Triplet Loss:
L = max(0, d(a, p) - d(a, n) + margin)

where a=anchor, p=positive, n=negative, d=distance function

References:
- Schroff et al. (2015): "FaceNet: A Unified Embedding for Face Recognition"
- Wang et al. (2014): "Learning Fine-grained Image Similarity with Deep Ranking"
- Hermans et al. (2017): "In Defense of the Triplet Loss for Person Re-Identification"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import numpy as np


class TripletLoss(nn.Module):
    """
    Standard Triplet Loss with various distance metrics.
    
    Encourages the distance between anchor and positive to be smaller
    than the distance between anchor and negative by at least a margin.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance: str = 'euclidean',  # euclidean, cosine, manhattan
        normalize: bool = False,
        reduction: str = 'mean',
        swap: bool = False
    ):
        """
        Initialize Triplet Loss.
        
        Args:
            margin: Margin for triplet constraint
            distance: Distance metric to use
            normalize: Whether to normalize embeddings
            reduction: Loss reduction method
            swap: Whether to use swap variant (considers both anchor-negative and positive-negative)
        """
        super().__init__()
        
        self.margin = margin
        self.distance = distance
        self.normalize = normalize
        self.reduction = reduction
        self.swap = swap
        
        # Select distance function
        self.distance_fn = self._get_distance_function(distance)
    
    def _get_distance_function(self, distance: str) -> Callable:
        """Get distance function based on metric name."""
        if distance == 'euclidean':
            return lambda x, y: F.pairwise_distance(x, y, p=2)
        elif distance == 'cosine':
            return lambda x, y: 1 - F.cosine_similarity(x, y)
        elif distance == 'manhattan':
            return lambda x, y: F.pairwise_distance(x, y, p=1)
        else:
            raise ValueError(f"Unknown distance metric: {distance}")
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
            mask: Optional mask for valid triplets
            
        Returns:
            Computed loss value
        """
        # Normalize if required
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        dist_ap = self.distance_fn(anchor, positive)
        dist_an = self.distance_fn(anchor, negative)
        
        # Basic triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin)
        
        # Swap variant: also ensure positive is closer to anchor than to negative
        if self.swap:
            dist_pn = self.distance_fn(positive, negative)
            loss_swap = F.relu(dist_ap - dist_pn + self.margin)
            loss = torch.max(loss, loss_swap)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            
            if self.reduction == 'mean':
                loss = loss.sum() / (mask.sum() + 1e-12)
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        
        return loss


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss with hard/semi-hard mining.
    
    Automatically selects triplets from a batch instead of requiring
    pre-formed triplets.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        mining: str = 'hard',  # hard, semi-hard, all
        distance: str = 'euclidean',
        normalize: bool = True,
        squared: bool = False
    ):
        """
        Initialize Online Triplet Loss.
        
        Args:
            margin: Margin for triplet constraint
            mining: Triplet mining strategy
            distance: Distance metric
            normalize: Whether to normalize embeddings
            squared: Whether to use squared distance
        """
        super().__init__()
        
        self.margin = margin
        self.mining = mining
        self.distance = distance
        self.normalize = normalize
        self.squared = squared
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute online triplet loss.
        
        Args:
            embeddings: Batch embeddings (N, D)
            labels: Batch labels (N,)
            
        Returns:
            Computed loss value
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        dist_matrix = self._compute_distance_matrix(embeddings)
        
        # Mine triplets
        triplets = self._mine_triplets(dist_matrix, labels)
        
        if len(triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute loss for mined triplets
        anchor_idx, positive_idx, negative_idx = triplets
        
        dist_ap = dist_matrix[anchor_idx, positive_idx]
        dist_an = dist_matrix[anchor_idx, negative_idx]
        
        loss = F.relu(dist_ap - dist_an + self.margin)
        
        return loss.mean()
    
    def _compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix."""
        if self.distance == 'euclidean':
            # Efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
            dot_product = torch.matmul(embeddings, embeddings.T)
            square_norm = torch.diag(dot_product)
            
            dist_matrix = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            dist_matrix = torch.clamp(dist_matrix, min=0.0)
            
            if not self.squared:
                dist_matrix = torch.sqrt(dist_matrix + 1e-12)
        elif self.distance == 'cosine':
            dist_matrix = 1 - torch.matmul(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        return dist_matrix
    
    def _mine_triplets(
        self,
        dist_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine triplets based on strategy."""
        batch_size = labels.size(0)
        
        # Create masks for valid positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal
        indices_not_equal = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        labels_equal[indices_not_equal] = False
        
        if self.mining == 'hard':
            # Hard triplet mining
            anchor_idx = []
            positive_idx = []
            negative_idx = []
            
            for i in range(batch_size):
                # Get hardest positive
                positive_mask = labels_equal[i]
                if positive_mask.any():
                    hardest_positive = dist_matrix[i][positive_mask].argmax()
                    positive_indices = torch.where(positive_mask)[0]
                    
                    # Get hardest negative
                    negative_mask = labels_not_equal[i]
                    hardest_negative = dist_matrix[i][negative_mask].argmin()
                    negative_indices = torch.where(negative_mask)[0]
                    
                    anchor_idx.append(i)
                    positive_idx.append(positive_indices[hardest_positive])
                    negative_idx.append(negative_indices[hardest_negative])
            
            if anchor_idx:
                return (
                    torch.tensor(anchor_idx, device=labels.device),
                    torch.tensor(positive_idx, device=labels.device),
                    torch.tensor(negative_idx, device=labels.device)
                )
        
        elif self.mining == 'semi-hard':
            # Semi-hard triplet mining
            anchor_idx = []
            positive_idx = []
            negative_idx = []
            
            for i in range(batch_size):
                positive_mask = labels_equal[i]
                if positive_mask.any():
                    for j in torch.where(positive_mask)[0]:
                        dist_ap = dist_matrix[i, j]
                        
                        # Find semi-hard negatives
                        negative_mask = labels_not_equal[i]
                        dist_an = dist_matrix[i][negative_mask]
                        
                        # Semi-hard: dist_ap < dist_an < dist_ap + margin
                        semi_hard_mask = (dist_an > dist_ap) & (dist_an < dist_ap + self.margin)
                        
                        if semi_hard_mask.any():
                            negative_indices = torch.where(negative_mask)[0]
                            semi_hard_negatives = negative_indices[semi_hard_mask]
                            
                            # Select random semi-hard negative
                            k = torch.randint(len(semi_hard_negatives), (1,)).item()
                            
                            anchor_idx.append(i)
                            positive_idx.append(j.item())
                            negative_idx.append(semi_hard_negatives[k].item())
            
            if anchor_idx:
                return (
                    torch.tensor(anchor_idx, device=labels.device),
                    torch.tensor(positive_idx, device=labels.device),
                    torch.tensor(negative_idx, device=labels.device)
                )
        
        elif self.mining == 'all':
            # All valid triplets
            anchor_idx = []
            positive_idx = []
            negative_idx = []
            
            for i in range(batch_size):
                positive_mask = labels_equal[i]
                negative_mask = labels_not_equal[i]
                
                if positive_mask.any() and negative_mask.any():
                    positive_indices = torch.where(positive_mask)[0]
                    negative_indices = torch.where(negative_mask)[0]
                    
                    # Create all combinations
                    for j in positive_indices:
                        for k in negative_indices:
                            anchor_idx.append(i)
                            positive_idx.append(j.item())
                            negative_idx.append(k.item())
            
            if anchor_idx:
                return (
                    torch.tensor(anchor_idx, device=labels.device),
                    torch.tensor(positive_idx, device=labels.device),
                    torch.tensor(negative_idx, device=labels.device)
                )
        
        return (torch.tensor([]), torch.tensor([]), torch.tensor([]))


class AngularTripletLoss(nn.Module):
    """
    Angular Triplet Loss that operates on angles rather than distances.
    
    Encourages the angle between anchor-positive to be smaller than
    anchor-negative by a margin.
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        scale: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Angular Triplet Loss.
        
        Args:
            margin: Angular margin (in radians)
            scale: Scaling factor for embeddings
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.margin = margin
        self.scale = scale
        self.reduction = reduction
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angular triplet loss.
        
        Args:
            anchor: Anchor embeddings (N, D)
            positive: Positive embeddings (N, D)
            negative: Negative embeddings (N, D)
            
        Returns:
            Computed loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1) * self.scale
        positive = F.normalize(positive, p=2, dim=1) * self.scale
        negative = F.normalize(negative, p=2, dim=1) * self.scale
        
        # Compute cosine similarities (normalized dot products)
        cos_ap = F.cosine_similarity(anchor, positive)
        cos_an = F.cosine_similarity(anchor, negative)
        
        # Convert to angles
        angle_ap = torch.acos(torch.clamp(cos_ap, -1 + 1e-7, 1 - 1e-7))
        angle_an = torch.acos(torch.clamp(cos_an, -1 + 1e-7, 1 - 1e-7))
        
        # Angular triplet loss
        loss = F.relu(angle_ap - angle_an + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
