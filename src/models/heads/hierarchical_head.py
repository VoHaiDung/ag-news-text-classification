"""
Hierarchical Classification Head Implementation
================================================

Implementation of hierarchical classification heads for multi-level categorization,
based on:
- Silla & Freitas (2011): "A survey of hierarchical classification across different application domains"
- Cerri et al. (2014): "Hierarchical multi-label classification using local neural networks"
- Wehrmann et al. (2018): "Hierarchical Multi-Label Classification Networks"

Mathematical Foundation:
Hierarchical classification decomposes P(y|x) into:
P(y|x) = P(y_root|x) ∏ P(y_i|y_parent(i), x)
where y_i is the label at level i and parent(i) is its parent node.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HierarchicalHeadConfig:
    """Configuration for hierarchical classification head."""
    hidden_size: int = 768
    hierarchy: Dict[str, List[str]] = None  # Hierarchy structure
    
    # Architecture
    use_global_classifier: bool = True  # Global classifier for all nodes
    use_local_classifiers: bool = True  # Local classifier per parent
    share_parameters: bool = False  # Share parameters across levels
    
    # Loss weighting
    level_weights: Optional[List[float]] = None  # Weights for each level
    use_hierarchical_softmax: bool = True
    consistency_weight: float = 0.1  # Weight for hierarchy consistency loss
    
    # Inference
    inference_strategy: str = "top_down"  # "top_down", "bottom_up", "global"
    threshold: float = 0.5  # Threshold for multi-label
    
    # Regularization
    dropout: float = 0.1
    use_label_smoothing: bool = True
    label_smoothing_epsilon: float = 0.1


class HierarchyTree:
    """
    Tree structure for label hierarchy.
    
    Manages parent-child relationships and path operations.
    """
    
    def __init__(self, hierarchy: Dict[str, List[str]]):
        """
        Initialize hierarchy tree.
        
        Args:
            hierarchy: Dictionary mapping parent to children
        """
        self.hierarchy = hierarchy
        self.nodes = set()
        self.children = defaultdict(list)
        self.parent = {}
        self.level = {}
        self.node_to_id = {}
        self.id_to_node = {}
        
        self._build_tree()
    
    def _build_tree(self):
        """Build tree structure from hierarchy."""
        # Find all nodes
        for parent, children in self.hierarchy.items():
            self.nodes.add(parent)
            self.nodes.update(children)
            self.children[parent] = children
            
            for child in children:
                self.parent[child] = parent
        
        # Find root
        self.root = None
        for node in self.nodes:
            if node not in self.parent:
                self.root = node
                break
        
        if self.root is None:
            # If no clear root, create virtual root
            self.root = "ROOT"
            for node in self.nodes:
                if node not in self.parent:
                    self.parent[node] = self.root
                    self.children[self.root].append(node)
            self.nodes.add(self.root)
        
        # Assign levels
        self._assign_levels()
        
        # Create node ID mappings
        for i, node in enumerate(sorted(self.nodes)):
            self.node_to_id[node] = i
            self.id_to_node[i] = node
    
    def _assign_levels(self):
        """Assign level to each node using BFS."""
        from collections import deque
        
        queue = deque([(self.root, 0)])
        visited = set()
        
        while queue:
            node, level = queue.popleft()
            if node in visited:
                continue
            
            visited.add(node)
            self.level[node] = level
            
            for child in self.children.get(node, []):
                queue.append((child, level + 1))
    
    def get_path(self, node: str) -> List[str]:
        """Get path from root to node."""
        path = []
        current = node
        
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        
        return list(reversed(path))
    
    def get_ancestors(self, node: str) -> List[str]:
        """Get all ancestors of a node."""
        ancestors = []
        current = self.parent.get(node)
        
        while current is not None:
            ancestors.append(current)
            current = self.parent.get(current)
        
        return ancestors
    
    def get_descendants(self, node: str) -> List[str]:
        """Get all descendants of a node."""
        descendants = []
        stack = [node]
        
        while stack:
            current = stack.pop()
            children = self.children.get(current, [])
            descendants.extend(children)
            stack.extend(children)
        
        return descendants
    
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if one node is ancestor of another."""
        return ancestor in self.get_ancestors(descendant)


class LocalClassifier(nn.Module):
    """
    Local classifier for a specific parent node.
    
    Classifies among children of a parent node.
    """
    
    def __init__(
        self,
        input_size: int,
        num_children: int,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize local classifier.
        
        Args:
            input_size: Input dimension
            num_children: Number of children to classify
            hidden_size: Hidden layer size
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_size:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_children)
            )
        else:
            self.classifier = nn.Linear(input_size, num_children)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through local classifier."""
        return self.classifier(x)


class HierarchicalClassificationHead(nn.Module):
    """
    Hierarchical classification head for multi-level categorization.
    
    Supports various hierarchical classification strategies:
    1. Top-down: Classify from root to leaves
    2. Bottom-up: Aggregate from leaves to root
    3. Global: Classify all nodes simultaneously
    
    Enforces hierarchy consistency through specialized losses.
    """
    
    def __init__(self, config: HierarchicalHeadConfig):
        """
        Initialize hierarchical classification head.
        
        Args:
            config: Configuration
        """
        super().__init__()
        
        self.config = config
        
        # Default AG News hierarchy if not provided
        if config.hierarchy is None:
            config.hierarchy = {
                "News": ["World", "Sports", "Business", "Technology"],
                "World": ["Politics", "Conflicts", "Disasters"],
                "Sports": ["Football", "Basketball", "Tennis"],
                "Business": ["Markets", "Companies", "Economy"],
                "Technology": ["AI", "Internet", "Gadgets"]
            }
        
        # Build hierarchy tree
        self.tree = HierarchyTree(config.hierarchy)
        self.num_nodes = len(self.tree.nodes)
        
        # Create classifiers
        self._create_classifiers()
        
        logger.info(
            f"Initialized HierarchicalClassificationHead with "
            f"{self.num_nodes} nodes in {max(self.tree.level.values()) + 1} levels"
        )
    
    def _create_classifiers(self):
        """Create classification layers based on configuration."""
        # Global classifier for all nodes
        if self.config.use_global_classifier:
            self.global_classifier = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_size, self.num_nodes)
            )
        
        # Local classifiers for each parent
        if self.config.use_local_classifiers:
            self.local_classifiers = nn.ModuleDict()
            
            for parent, children in self.tree.children.items():
                if children:  # Only create if parent has children
                    if self.config.share_parameters:
                        # Share parameters across same level
                        level = self.tree.level[parent]
                        classifier_name = f"level_{level}"
                        
                        if classifier_name not in self.local_classifiers:
                            self.local_classifiers[classifier_name] = LocalClassifier(
                                self.config.hidden_size,
                                max(len(c) for p, c in self.tree.children.items() 
                                    if self.tree.level[p] == level),
                                dropout=self.config.dropout
                            )
                    else:
                        # Separate classifier for each parent
                        self.local_classifiers[parent] = LocalClassifier(
                            self.config.hidden_size,
                            len(children),
                            dropout=self.config.dropout
                        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hierarchical_labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Forward pass through hierarchical classification head.
        
        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            labels: Flat labels (leaf nodes)
            hierarchical_labels: Labels for each level
            
        Returns:
            Tuple of (logits, loss, predictions_dict)
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        predictions = {}
        
        # Global classification
        if self.config.use_global_classifier:
            global_logits = self.global_classifier(hidden_states)
            predictions["global"] = global_logits
        
        # Local classification
        if self.config.use_local_classifiers:
            local_predictions = self._local_classification(hidden_states)
            predictions["local"] = local_predictions
        
        # Combine predictions based on inference strategy
        if self.config.inference_strategy == "top_down":
            final_logits = self._top_down_inference(predictions, hidden_states)
        elif self.config.inference_strategy == "bottom_up":
            final_logits = self._bottom_up_inference(predictions)
        else:  # global
            final_logits = global_logits if self.config.use_global_classifier else None
        
        # Compute loss
        loss = None
        if labels is not None or hierarchical_labels is not None:
            loss = self._compute_hierarchical_loss(
                predictions,
                labels,
                hierarchical_labels
            )
        
        return final_logits, loss, predictions
    
    def _local_classification(
        self,
        hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local classification for each parent node.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Dictionary of local predictions
        """
        local_predictions = {}
        
        for parent, children in self.tree.children.items():
            if not children:
                continue
            
            if self.config.share_parameters:
                level = self.tree.level[parent]
                classifier_name = f"level_{level}"
                classifier = self.local_classifiers[classifier_name]
            else:
                classifier = self.local_classifiers[parent]
            
            # Get predictions for children
            logits = classifier(hidden_states)
            
            # Trim to actual number of children if sharing parameters
            if self.config.share_parameters:
                logits = logits[:, :len(children)]
            
            local_predictions[parent] = logits
        
        return local_predictions
    
    def _top_down_inference(
        self,
        predictions: Dict[str, Any],
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Top-down hierarchical inference.
        
        Traverse from root to leaves, making decisions at each level.
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Initialize leaf probabilities
        leaf_nodes = [node for node in self.tree.nodes 
                     if node not in self.tree.children or not self.tree.children[node]]
        leaf_logits = torch.zeros(batch_size, len(leaf_nodes), device=device)
        
        # Start from root
        if "local" in predictions:
            # Traverse tree top-down
            for leaf_idx, leaf in enumerate(leaf_nodes):
                path = self.tree.get_path(leaf)
                path_prob = 1.0
                
                for i in range(len(path) - 1):
                    parent = path[i]
                    child = path[i + 1]
                    
                    if parent in predictions["local"]:
                        parent_logits = predictions["local"][parent]
                        parent_probs = F.softmax(parent_logits, dim=-1)
                        
                        # Find child index
                        children = self.tree.children[parent]
                        if child in children:
                            child_idx = children.index(child)
                            path_prob = path_prob * parent_probs[:, child_idx]
                
                leaf_logits[:, leaf_idx] = torch.log(path_prob + 1e-10)
        
        return leaf_logits
    
    def _bottom_up_inference(
        self,
        predictions: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Bottom-up hierarchical inference.
        
        Aggregate predictions from leaves to root.
        """
        # Start with global predictions if available
        if "global" in predictions:
            # Extract leaf node predictions from global
            leaf_nodes = [node for node in self.tree.nodes 
                         if node not in self.tree.children or not self.tree.children[node]]
            
            leaf_indices = [self.tree.node_to_id[leaf] for leaf in leaf_nodes]
            leaf_logits = predictions["global"][:, leaf_indices]
            
            return leaf_logits
        
        # Otherwise, aggregate local predictions
        # Implementation depends on specific aggregation strategy
        return None
    
    def _compute_hierarchical_loss(
        self,
        predictions: Dict[str, Any],
        labels: Optional[torch.Tensor],
        hierarchical_labels: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute hierarchical classification loss.
        
        Combines:
        1. Local classification losses at each level
        2. Global classification loss
        3. Hierarchy consistency loss
        """
        total_loss = 0
        loss_components = {}
        
        # Global loss
        if "global" in predictions and labels is not None:
            global_loss = F.cross_entropy(predictions["global"], labels)
            total_loss = total_loss + global_loss
            loss_components["global"] = global_loss
        
        # Local losses
        if "local" in predictions and hierarchical_labels:
            local_loss = 0
            for parent, logits in predictions["local"].items():
                if parent in hierarchical_labels:
                    local_loss += F.cross_entropy(
                        logits,
                        hierarchical_labels[parent]
                    )
            
            if local_loss > 0:
                total_loss = total_loss + local_loss
                loss_components["local"] = local_loss
        
        # Hierarchy consistency loss
        if self.config.consistency_weight > 0:
            consistency_loss = self._compute_consistency_loss(predictions)
            total_loss = total_loss + self.config.consistency_weight * consistency_loss
            loss_components["consistency"] = consistency_loss
        
        return total_loss
    
    def _compute_consistency_loss(
        self,
        predictions: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute hierarchy consistency loss.
        
        Ensures predictions respect hierarchy constraints:
        - If parent is not predicted, children should not be predicted
        - Predictions should form valid paths in hierarchy
        """
        consistency_loss = 0
        
        if "global" in predictions:
            global_probs = F.sigmoid(predictions["global"])
            
            # For each parent-child pair
            for parent, children in self.tree.children.items():
                if not children:
                    continue
                
                parent_id = self.tree.node_to_id[parent]
                parent_prob = global_probs[:, parent_id]
                
                for child in children:
                    child_id = self.tree.node_to_id[child]
                    child_prob = global_probs[:, child_id]
                    
                    # Child probability should not exceed parent
                    violation = F.relu(child_prob - parent_prob)
                    consistency_loss = consistency_loss + violation.mean()
        
        return consistency_loss
    
    def decode_predictions(
        self,
        logits: torch.Tensor,
        strategy: str = "threshold"
    ) -> List[List[str]]:
        """
        Decode predictions to hierarchical labels.
        
        Args:
            logits: Prediction logits
            strategy: Decoding strategy ("threshold", "top_k", "beam")
            
        Returns:
            List of predicted label paths
        """
        batch_size = logits.size(0)
        predictions = []
        
        if strategy == "threshold":
            probs = F.sigmoid(logits)
            
            for i in range(batch_size):
                predicted_nodes = []
                
                for j, prob in enumerate(probs[i]):
                    if prob > self.config.threshold:
                        node = self.tree.id_to_node.get(j)
                        if node:
                            predicted_nodes.append(node)
                
                # Ensure hierarchy consistency
                predicted_nodes = self._ensure_consistency(predicted_nodes)
                predictions.append(predicted_nodes)
        
        return predictions
    
    def _ensure_consistency(self, nodes: List[str]) -> List[str]:
        """
        Ensure predicted nodes form consistent hierarchy.
        
        Add missing ancestors to maintain hierarchy.
        """
        consistent_nodes = set(nodes)
        
        for node in nodes:
            ancestors = self.tree.get_ancestors(node)
            consistent_nodes.update(ancestors)
        
        return list(consistent_nodes)
