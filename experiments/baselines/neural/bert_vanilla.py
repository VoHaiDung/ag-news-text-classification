"""
BERT Vanilla Baseline for AG News Text Classification
================================================================================
This module implements vanilla BERT models as baseline for text classification,
including fine-tuning strategies and various pooling methods.

BERT provides contextualized embeddings through bidirectional transformers,
achieving strong performance on text classification tasks.

References:
    - Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
    - Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach
    - Sun, C., et al. (2019). How to Fine-Tune BERT for Text Classification?

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json
from datetime import datetime
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class BERTClassifier(nn.Module):
    """
    BERT-based text classifier.
    
    Implements:
    - Multiple pooling strategies
    - Custom classification heads
    - Layer-wise learning rate decay
    - Gradient checkpointing
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        dropout: float = 0.1,
        pooling_strategy: str = "cls",
        hidden_dim: int = 768,
        num_hidden_layers: int = 1,
        use_all_layers: bool = False,
        layer_weights_trainable: bool = False,
        gradient_checkpointing: bool = False,
        freeze_embeddings: bool = False,
        freeze_layers: Optional[List[int]] = None
    ):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: Pretrained model name
            num_classes: Number of output classes
            dropout: Dropout probability
            pooling_strategy: Pooling strategy ("cls", "mean", "max", "cls_mean", "weighted")
            hidden_dim: Hidden dimension for classifier
            num_hidden_layers: Number of hidden layers in classifier
            use_all_layers: Whether to use all BERT layers
            layer_weights_trainable: Whether layer weights are trainable
            gradient_checkpointing: Whether to use gradient checkpointing
            freeze_embeddings: Whether to freeze embedding layer
            freeze_layers: List of layer indices to freeze
        """
        super(BERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        self.use_all_layers = use_all_layers
        
        # Load pretrained model
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        # Enable gradient checkpointing if specified
        if gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
        
        # Freeze embeddings if specified
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze specific layers
        if freeze_layers:
            for layer_idx in freeze_layers:
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
        
        # Get BERT output dimension
        bert_dim = config.hidden_size
        
        # Layer weights for weighted pooling
        if use_all_layers:
            self.num_bert_layers = config.num_hidden_layers
            if layer_weights_trainable:
                self.layer_weights = nn.Parameter(torch.ones(self.num_bert_layers) / self.num_bert_layers)
            else:
                self.register_buffer('layer_weights', torch.ones(self.num_bert_layers) / self.num_bert_layers)
        
        # Pooling layer for "weighted" strategy
        if pooling_strategy == "weighted":
            self.attention_weights = nn.Linear(bert_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Build classification head
        classifier_layers = []
        input_dim = bert_dim
        
        # Handle concatenated pooling strategies
        if pooling_strategy == "cls_mean":
            input_dim = bert_dim * 2
        elif pooling_strategy == "cls_max_mean":
            input_dim = bert_dim * 3
        
        # Add hidden layers
        for i in range(num_hidden_layers):
            if i == 0:
                classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
        
        # Output layer
        if num_hidden_layers > 0:
            classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            classifier_layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.use_all_layers
        )
        
        # Extract representations based on strategy
        if self.use_all_layers:
            # Use all layers
            all_hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer
            
            # Weighted average of all layers
            if hasattr(self, 'layer_weights'):
                weighted_hidden = torch.stack(all_hidden_states, dim=0)
                weights = F.softmax(self.layer_weights, dim=0)
                weighted_hidden = (weighted_hidden * weights.view(-1, 1, 1, 1)).sum(dim=0)
            else:
                weighted_hidden = torch.mean(torch.stack(all_hidden_states, dim=0), dim=0)
            
            sequence_output = weighted_hidden
        else:
            sequence_output = outputs.last_hidden_state
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token representation
            pooled = sequence_output[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling
            masked_output = sequence_output * attention_mask.unsqueeze(-1)
            pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        elif self.pooling_strategy == "max":
            # Max pooling
            masked_output = sequence_output.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            pooled, _ = torch.max(masked_output, dim=1)
        
        elif self.pooling_strategy == "cls_mean":
            # Concatenate CLS and mean pooling
            cls_token = sequence_output[:, 0, :]
            masked_output = sequence_output * attention_mask.unsqueeze(-1)
            mean_pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            pooled = torch.cat([cls_token, mean_pooled], dim=-1)
        
        elif self.pooling_strategy == "cls_max_mean":
            # Concatenate CLS, max, and mean pooling
            cls_token = sequence_output[:, 0, :]
            
            masked_output = sequence_output * attention_mask.unsqueeze(-1)
            mean_pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            masked_output_max = sequence_output.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            max_pooled, _ = torch.max(masked_output_max, dim=1)
            
            pooled = torch.cat([cls_token, max_pooled, mean_pooled], dim=-1)
        
        elif self.pooling_strategy == "weighted":
            # Attention-weighted pooling
            attention_scores = self.attention_weights(sequence_output).squeeze(-1)
            attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled = torch.sum(sequence_output * attention_weights.unsqueeze(-1), dim=1)
        
        else:
            # Default to CLS
            pooled = sequence_output[:, 0, :]
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class BERTDataset(Dataset):
    """Dataset class for BERT model."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of texts
            labels: List of labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTVanillaBaseline:
    """
    Vanilla BERT baseline for text classification.
    
    Provides:
    - Fine-tuning strategies
    - Multiple pooling methods
    - Layer-wise learning rate decay
    - Mixed precision training
    - Comprehensive evaluation
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 4,
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        pooling_strategy: str = "cls",
        num_hidden_layers: int = 1,
        hidden_dim: int = 768,
        use_all_layers: bool = False,
        gradient_checkpointing: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "linear",
        layer_lr_decay: float = 0.95,
        freeze_embeddings: bool = False,
        freeze_layers: Optional[List[int]] = None,
        mixed_precision: bool = False,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        Initialize BERT baseline.
        
        Args:
            model_name: Pretrained model name
            num_classes: Number of classes
            max_length: Maximum sequence length
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            dropout: Dropout rate
            pooling_strategy: Pooling strategy
            num_hidden_layers: Number of hidden layers
            hidden_dim: Hidden dimension
            use_all_layers: Whether to use all BERT layers
            gradient_checkpointing: Whether to use gradient checkpointing
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm
            scheduler_type: Scheduler type
            layer_lr_decay: Layer-wise LR decay
            freeze_embeddings: Whether to freeze embeddings
            freeze_layers: Layers to freeze
            mixed_precision: Whether to use mixed precision
            device: Device to use
            seed: Random seed
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.pooling_strategy = pooling_strategy
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.use_all_layers = use_all_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler_type = scheduler_type
        self.layer_lr_decay = layer_lr_decay
        self.freeze_embeddings = freeze_embeddings
        self.freeze_layers = freeze_layers
        self.mixed_precision = mixed_precision
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        
        set_seed(seed)
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self.results = {
            "train_history": [],
            "val_history": [],
            "test_metrics": {},
            "best_epoch": 0,
            "training_time": 0
        }
        
        self.registry = Registry()
        self.factory = Factory()
        self.metrics_calculator = ClassificationMetrics()
        
        logger.info(f"Initialized BERT baseline with {model_name}")
    
    def _build_model(self):
        """Build BERT model."""
        self.model = BERTClassifier(
            model_name=self.model_name,
            num_classes=self.num_classes,
            dropout=self.dropout,
            pooling_strategy=self.pooling_strategy,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            use_all_layers=self.use_all_layers,
            gradient_checkpointing=self.gradient_checkpointing,
            freeze_embeddings=self.freeze_embeddings,
            freeze_layers=self.freeze_layers
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.max_length
        )
        
        # Setup optimizer with layer-wise learning rate decay
        optimizer_params = self._get_optimizer_params()
        
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        # Setup mixed precision training
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _get_optimizer_params(self):
        """Get optimizer parameters with layer-wise learning rate decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Group parameters by layer
        bert_layers = self.model.bert.encoder.layer
        num_layers = len(bert_layers)
        
        optimizer_parameters = []
        
        # Embedding parameters
        embedding_params = {
            "params": [p for n, p in self.model.bert.embeddings.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": self.weight_decay,
            "lr": self.learning_rate * (self.layer_lr_decay ** (num_layers + 1))
        }
        embedding_params_no_decay = {
            "params": [p for n, p in self.model.bert.embeddings.named_parameters()
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": self.learning_rate * (self.layer_lr_decay ** (num_layers + 1))
        }
        
        optimizer_parameters.extend([embedding_params, embedding_params_no_decay])
        
        # Layer parameters
        for i, layer in enumerate(bert_layers):
            layer_lr = self.learning_rate * (self.layer_lr_decay ** (num_layers - i))
            
            layer_params = {
                "params": [p for n, p in layer.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
                "lr": layer_lr
            }
            layer_params_no_decay = {
                "params": [p for n, p in layer.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": layer_lr
            }
            
            optimizer_parameters.extend([layer_params, layer_params_no_decay])
        
        # Pooler and classifier parameters
        classifier_params = {
            "params": [p for n, p in self.model.classifier.named_parameters()],
            "weight_decay": self.weight_decay,
            "lr": self.learning_rate
        }
        
        optimizer_parameters.append(classifier_params)
        
        # Filter out empty parameter groups
        optimizer_parameters = [p for p in optimizer_parameters if len(p["params"]) > 0]
        
        return optimizer_parameters
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Train BERT model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            
        Returns:
            Training results
        """
        logger.info("Training BERT model")
        
        self._build_model()
        
        # Create datasets
        train_dataset = BERTDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = BERTDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        if self.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        import time
        start_time = time.time()
        
        best_val_acc = 0
        patience_counter = 0
        early_stopping_patience = 3
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Mixed precision training
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids, attention_mask, token_type_ids)
                        loss = criterion(logits, labels)
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    logits = self.model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                
                train_loss += loss.item() * self.gradient_accumulation_steps
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        token_type_ids = batch['token_type_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        if self.mixed_precision:
                            with torch.cuda.amp.autocast():
                                logits = self.model(input_ids, attention_mask, token_type_ids)
                                loss = criterion(logits, labels)
                        else:
                            logits = self.model(input_ids, attention_mask, token_type_ids)
                            loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.results["best_epoch"] = epoch
                    patience_counter = 0
                    self._save_checkpoint(epoch, val_acc)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                self.results["val_history"].append({
                    "epoch": epoch,
                    "loss": avg_val_loss,
                    "accuracy": val_acc
                })
                
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
            
            self.results["train_history"].append({
                "epoch": epoch,
                "loss": avg_train_loss,
                "accuracy": train_acc
            })
        
        self.results["training_time"] = time.time() - start_time
        logger.info(f"Training completed in {self.results['training_time']:.2f} seconds")
        
        # Load best checkpoint if validation was used
        if val_loader is not None and Path("outputs/checkpoints/bert_best.pt").exists():
            self.load_checkpoint("outputs/checkpoints/bert_best.pt")
        
        return self.results
    
    def evaluate(
        self,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate model.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Test metrics
        """
        logger.info("Evaluating BERT model")
        
        self.model.eval()
        
        test_dataset = BERTDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids, attention_mask, token_type_ids)
                else:
                    logits = self.model(input_ids, attention_mask, token_type_ids)
                
                probs = F.softmax(logits, dim=-1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        self.results["test_metrics"] = self.metrics_calculator.calculate_metrics(
            all_labels,
            all_predictions,
            all_probs
        )
        
        logger.info(f"Test Accuracy: {self.results['test_metrics']['accuracy']:.4f}")
        
        return self.results["test_metrics"]
    
    def _save_checkpoint(self, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'config': {
                'model_name': self.model_name,
                'pooling_strategy': self.pooling_strategy,
                'num_hidden_layers': self.num_hidden_layers
            }
        }
        
        checkpoint_path = Path("outputs/checkpoints/bert_best.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": f"BERT_{self.pooling_strategy}",
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pooling_strategy": self.pooling_strategy,
            "num_hidden_layers": self.num_hidden_layers,
            "use_all_layers": self.use_all_layers,
            "training_time": self.results.get("training_time", 0),
            "best_epoch": self.results.get("best_epoch", 0),
            "performance": {
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0),
                "test_f1": self.results.get("test_metrics", {}).get("f1_weighted", 0)
            }
        }


def run_bert_experiments():
    """Run BERT experiments."""
    logger.info("Starting BERT experiments")
    
    # Load data
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    # Test different configurations
    configurations = [
        {"name": "bert_cls", "pooling_strategy": "cls"},
        {"name": "bert_mean", "pooling_strategy": "mean"},
        {"name": "bert_cls_mean", "pooling_strategy": "cls_mean"}
    ]
    
    results = {}
    
    for config in configurations:
        logger.info(f"\nTesting configuration: {config['name']}")
        
        model = BERTVanillaBaseline(
            model_name="bert-base-uncased",
            pooling_strategy=config["pooling_strategy"],
            num_epochs=3,
            batch_size=32,
            max_length=128
        )
        
        model.train(
            train_data["texts"][:1000],  # Use subset for testing
            train_data["labels"][:1000],
            val_data["texts"][:200],
            val_data["labels"][:200]
        )
        
        test_metrics = model.evaluate(
            test_data["texts"][:200],
            test_data["labels"][:200]
        )
        
        results[config["name"]] = {
            "config": config,
            "metrics": test_metrics,
            "summary": model.get_summary()
        }
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda k: results[k]["metrics"]["accuracy"])
    logger.info(f"\nBest configuration: {best_config}")
    logger.info(f"Best accuracy: {results[best_config]['metrics']['accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    run_bert_experiments()
