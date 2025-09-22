"""
Knowledge Distillation for Model Compression
=============================================

Implementation of knowledge distillation techniques for compressing large models,
based on:
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
- Sun et al. (2019): "Patient Knowledge Distillation for BERT Model Compression"

Mathematical Foundation:
L_KD = α * L_CE(y, p_student) + (1-α) * τ² * L_KL(p_teacher/τ, p_student/τ)
where τ is temperature, α is interpolation weight, and L_KL is KL divergence.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union, List, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss, MSELoss, CosineEmbeddingLoss

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    
    # Distillation type
    distillation_type: str = "soft"  # "soft", "hard", "feature", "attention", "patient"
    
    # Temperature for soft targets
    temperature: float = 4.0
    temperature_scheduler: str = "constant"  # "constant", "linear", "cosine"
    initial_temperature: float = 8.0
    final_temperature: float = 1.0
    
    # Loss weights
    alpha: float = 0.7  # Weight for student loss
    beta: float = 0.3   # Weight for distillation loss
    
    # Feature distillation
    feature_layers: List[int] = None  # Layers to distill features from
    feature_loss: str = "mse"  # "mse", "cosine", "l1"
    feature_weight: float = 0.1
    
    # Attention distillation
    attention_layers: List[int] = None
    attention_loss: str = "mse"
    attention_weight: float = 0.1
    
    # Patient distillation (layer-wise)
    patient_layers: int = 6  # Number of layers for patient distillation
    patient_strategy: str = "skip"  # "skip", "last", "emb"
    
    # Advanced options
    use_hard_labels: bool = False  # Use hard labels from teacher
    label_smoothing: float = 0.0
    clip_grad: float = 1.0
    
    # Optimization
    separate_optimizer: bool = False  # Use separate optimizer for distillation
    distill_lr: float = 1e-4
    
    # Data augmentation for distillation
    augment_distillation: bool = False
    augmentation_factor: int = 2
    
    # Dynamic teacher
    use_teacher_ensemble: bool = False
    ensemble_weights: List[float] = None


class TemperatureScheduler:
    """Schedule temperature during distillation"""
    
    def __init__(self, config: DistillationConfig):
        """
        Initialize temperature scheduler.
        
        Args:
            config: Distillation configuration
        """
        self.config = config
        self.current_step = 0
        self.total_steps = 10000  # Default, should be set
        
    def set_total_steps(self, total_steps: int):
        """Set total training steps"""
        self.total_steps = total_steps
        
    def step(self) -> float:
        """Get current temperature and update step"""
        if self.config.temperature_scheduler == "constant":
            temperature = self.config.temperature
        elif self.config.temperature_scheduler == "linear":
            # Linear decay from initial to final
            progress = min(self.current_step / self.total_steps, 1.0)
            temperature = self.config.initial_temperature - \
                         (self.config.initial_temperature - self.config.final_temperature) * progress
        elif self.config.temperature_scheduler == "cosine":
            # Cosine annealing
            import math
            progress = min(self.current_step / self.total_steps, 1.0)
            temperature = self.config.final_temperature + \
                         0.5 * (self.config.initial_temperature - self.config.final_temperature) * \
                         (1 + math.cos(math.pi * progress))
        else:
            temperature = self.config.temperature
            
        self.current_step += 1
        return temperature


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Implements various distillation losses including:
    - Soft target distillation
    - Feature matching
    - Attention transfer
    - Patient knowledge distillation
    """
    
    def __init__(self, config: DistillationConfig):
        """
        Initialize distillation loss.
        
        Args:
            config: Distillation configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize loss functions
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.mse_loss = MSELoss()
        self.cosine_loss = CosineEmbeddingLoss()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Temperature scheduler
        self.temperature_scheduler = TemperatureScheduler(config)
        
        # Statistics
        self.loss_history = {
            'student_loss': [],
            'distillation_loss': [],
            'feature_loss': [],
            'attention_loss': []
        }
        
        logger.info(f"Initialized DistillationLoss with type: {config.distillation_type}")
        
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs  
            labels: Ground truth labels
            
        Returns:
            Total loss and loss components dictionary
        """
        losses = {}
        total_loss = 0
        
        # Get current temperature
        temperature = self.temperature_scheduler.step()
        
        # 1. Soft target distillation loss
        if 'logits' in student_outputs and 'logits' in teacher_outputs:
            distill_loss = self._soft_target_loss(
                student_outputs['logits'],
                teacher_outputs['logits'],
                temperature
            )
            losses['distillation_loss'] = distill_loss.item()
            total_loss += self.config.beta * distill_loss
            
        # 2. Student prediction loss (if labels available)
        if labels is not None and 'logits' in student_outputs:
            student_loss = self.ce_loss(student_outputs['logits'], labels)
            losses['student_loss'] = student_loss.item()
            total_loss += self.config.alpha * student_loss
            
        # 3. Feature distillation loss
        if self.config.feature_layers and 'hidden_states' in student_outputs:
            feature_loss = self._feature_matching_loss(
                student_outputs.get('hidden_states', []),
                teacher_outputs.get('hidden_states', [])
            )
            if feature_loss is not None:
                losses['feature_loss'] = feature_loss.item()
                total_loss += self.config.feature_weight * feature_loss
                
        # 4. Attention distillation loss
        if self.config.attention_layers and 'attentions' in student_outputs:
            attention_loss = self._attention_transfer_loss(
                student_outputs.get('attentions', []),
                teacher_outputs.get('attentions', [])
            )
            if attention_loss is not None:
                losses['attention_loss'] = attention_loss.item()
                total_loss += self.config.attention_weight * attention_loss
                
        # Update history
        for key, value in losses.items():
            self.loss_history[key].append(value)
            
        return total_loss, losses
    
    def _soft_target_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Calculate soft target distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            temperature: Temperature for softmax
            
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence loss
        # Scale by T^2 as per Hinton et al.
        loss = self.kl_loss(student_soft, teacher_soft) * (temperature ** 2)
        
        return loss
    
    def _feature_matching_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Calculate feature matching loss.
        
        Args:
            student_features: Student hidden states
            teacher_features: Teacher hidden states
            
        Returns:
            Feature matching loss
        """
        if not student_features or not teacher_features:
            return None
            
        if self.config.feature_layers is None:
            return None
            
        total_loss = 0
        count = 0
        
        for layer_idx in self.config.feature_layers:
            if layer_idx < len(student_features) and layer_idx < len(teacher_features):
                s_feat = student_features[layer_idx]
                t_feat = teacher_features[layer_idx]
                
                # Ensure same dimensions
                if s_feat.shape != t_feat.shape:
                    # Project student features to teacher dimension if needed
                    if hasattr(self, f'projection_{layer_idx}'):
                        projection = getattr(self, f'projection_{layer_idx}')
                    else:
                        projection = nn.Linear(s_feat.shape[-1], t_feat.shape[-1])
                        setattr(self, f'projection_{layer_idx}', projection)
                    s_feat = projection(s_feat)
                
                if self.config.feature_loss == "mse":
                    loss = self.mse_loss(s_feat, t_feat.detach())
                elif self.config.feature_loss == "cosine":
                    s_flat = s_feat.view(-1, s_feat.shape[-1])
                    t_flat = t_feat.view(-1, t_feat.shape[-1])
                    target = torch.ones(s_flat.shape[0], device=s_feat.device)
                    loss = self.cosine_loss(s_flat, t_flat.detach(), target)
                elif self.config.feature_loss == "l1":
                    loss = F.l1_loss(s_feat, t_feat.detach())
                else:
                    loss = self.mse_loss(s_feat, t_feat.detach())
                    
                total_loss += loss
                count += 1
                
        return total_loss / count if count > 0 else None
    
    def _attention_transfer_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Calculate attention transfer loss.
        
        Args:
            student_attentions: Student attention maps
            teacher_attentions: Teacher attention maps
            
        Returns:
            Attention transfer loss
        """
        if not student_attentions or not teacher_attentions:
            return None
            
        if self.config.attention_layers is None:
            return None
            
        total_loss = 0
        count = 0
        
        for layer_idx in self.config.attention_layers:
            if layer_idx < len(student_attentions) and layer_idx < len(teacher_attentions):
                s_att = student_attentions[layer_idx]
                t_att = teacher_attentions[layer_idx]
                
                # Average over attention heads if present
                if s_att.dim() == 4:  # [batch, heads, seq, seq]
                    s_att = s_att.mean(dim=1)
                if t_att.dim() == 4:
                    t_att = t_att.mean(dim=1)
                    
                if self.config.attention_loss == "mse":
                    loss = self.mse_loss(s_att, t_att.detach())
                elif self.config.attention_loss == "kl":
                    # Treat attention as probability distribution
                    s_att_flat = s_att.view(-1, s_att.shape[-1])
                    t_att_flat = t_att.view(-1, t_att.shape[-1])
                    loss = self.kl_loss(
                        F.log_softmax(s_att_flat, dim=-1),
                        F.softmax(t_att_flat.detach(), dim=-1)
                    )
                else:
                    loss = self.mse_loss(s_att, t_att.detach())
                    
                total_loss += loss
                count += 1
                
        return total_loss / count if count > 0 else None


class KnowledgeDistiller:
    """
    Main class for knowledge distillation training.
    
    Manages the distillation process including:
    - Teacher-student training loop
    - Progressive distillation
    - Online distillation
    - Multi-teacher distillation
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[DistillationConfig] = None
    ):
        """
        Initialize knowledge distiller.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            config: Distillation configuration
        """
        self.teacher = teacher_model
        self.student = student_model
        self.config = config or DistillationConfig()
        
        # Set teacher to eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Initialize loss function
        self.loss_fn = DistillationLoss(self.config)
        
        # Separate optimizer for distillation if configured
        if self.config.separate_optimizer:
            self.distill_optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.config.distill_lr
            )
        else:
            self.distill_optimizer = None
            
        # Statistics
        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'best_student_acc': 0.0,
            'compression_ratio': self._calculate_compression_ratio()
        }
        
        logger.info(
            f"Initialized KnowledgeDistiller: "
            f"Teacher params: {sum(p.numel() for p in teacher_model.parameters()):,}, "
            f"Student params: {sum(p.numel() for p in student_model.parameters()):,}"
        )
    
    def distill_step(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one distillation step.
        
        Args:
            inputs: Input tensors
            attention_mask: Attention mask
            labels: Ground truth labels
            
        Returns:
            Dictionary of losses
        """
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            
        # Student forward pass
        student_outputs = self.student(
            inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Convert model outputs to dict format
        teacher_dict = self._outputs_to_dict(teacher_outputs)
        student_dict = self._outputs_to_dict(student_outputs)
        
        # Calculate distillation loss
        loss, loss_components = self.loss_fn(
            student_dict,
            teacher_dict,
            labels
        )
        
        # Update step counter
        self.training_stats['step'] += 1
        
        return {
            'total_loss': loss.item(),
            **loss_components
        }
    
    def _outputs_to_dict(self, outputs) -> Dict[str, torch.Tensor]:
        """Convert model outputs to dictionary format"""
        output_dict = {}
        
        if hasattr(outputs, 'logits'):
            output_dict['logits'] = outputs.logits
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            output_dict['hidden_states'] = outputs.hidden_states
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            output_dict['attentions'] = outputs.attentions
            
        return output_dict
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate model compression ratio"""
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        return teacher_params / student_params if student_params > 0 else 0
    
    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        scheduler=None
    ) -> Dict[str, float]:
        """
        Train for one epoch with distillation.
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer for student
            scheduler: Learning rate scheduler
            
        Returns:
            Epoch statistics
        """
        self.student.train()
        epoch_losses = []
        
        for batch in dataloader:
            # Move batch to device
            inputs = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            labels = batch.get('labels')
            
            # Zero gradients
            optimizer.zero_grad()
            if self.distill_optimizer:
                self.distill_optimizer.zero_grad()
                
            # Distillation step
            losses = self.distill_step(inputs, attention_mask, labels)
            
            # Backward pass
            total_loss = losses['total_loss']
            total_loss = torch.tensor(total_loss, requires_grad=True)
            total_loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.config.clip_grad
                )
            
            # Optimizer step
            optimizer.step()
            if self.distill_optimizer:
                self.distill_optimizer.step()
                
            if scheduler:
                scheduler.step()
                
            epoch_losses.append(losses)
            
        # Update epoch counter
        self.training_stats['epoch'] += 1
        
        # Calculate average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = sum(d[key] for d in epoch_losses) / len(epoch_losses)
            
        return avg_losses


class PatientKnowledgeDistillation(KnowledgeDistiller):
    """
    Patient Knowledge Distillation for BERT compression.
    
    Implements patient distillation where student learns from
    intermediate layers of the teacher progressively.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[DistillationConfig] = None
    ):
        """Initialize patient distillation"""
        super().__init__(teacher_model, student_model, config)
        
        # Map student layers to teacher layers
        self.layer_mapping = self._create_layer_mapping()
        
        logger.info(f"Initialized Patient KD with mapping: {self.layer_mapping}")
    
    def _create_layer_mapping(self) -> Dict[int, int]:
        """Create mapping from student layers to teacher layers"""
        teacher_layers = 12  # Assume BERT-base
        student_layers = self.config.patient_layers
        
        if self.config.patient_strategy == "skip":
            # Skip layers uniformly
            step = teacher_layers // student_layers
            mapping = {i: i * step for i in range(student_layers)}
        elif self.config.patient_strategy == "last":
            # Map to last N teacher layers
            mapping = {
                i: teacher_layers - student_layers + i 
                for i in range(student_layers)
            }
        else:
            # Default: uniform mapping
            mapping = {i: i for i in range(min(student_layers, teacher_layers))}
            
        return mapping


# Export classes
__all__ = [
    'DistillationConfig',
    'TemperatureScheduler',
    'DistillationLoss',
    'KnowledgeDistiller',
    'PatientKnowledgeDistillation'
]
