"""
LSTM Baseline for AG News Text Classification
================================================================================
This module implements LSTM-based neural network baselines for text classification,
including bidirectional variants and attention mechanisms.

LSTMs provide sequential modeling capabilities that capture long-range dependencies
in text through gating mechanisms and recurrent connections.

References:
    - Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
    - Graves, A., & Schmidhuber, J. (2005). Framewise Phoneme Classification with Bidirectional LSTM
    - Zhou, P., et al. (2016). Attention-Based Bidirectional Long Short-Term Memory Networks

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class LSTMTextClassifier(nn.Module):
    """
    LSTM-based text classifier with optional attention mechanism.
    
    This model implements:
    - Bidirectional LSTM layers
    - Multi-layer architecture
    - Dropout regularization
    - Optional attention mechanism
    - Residual connections
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_residual: bool = False,
        pool_type: str = "max",
        freeze_embeddings: bool = False,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize LSTM classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
            pool_type: Pooling type ("max", "mean", "last", "attention")
            freeze_embeddings: Whether to freeze embedding layer
            pretrained_embeddings: Pretrained embedding weights
        """
        super(LSTMTextClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.pool_type = pool_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_output_dim // 2, 1)
            )
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Classification layers
        if pool_type == "attention" and use_attention:
            classifier_input_dim = lstm_output_dim
        elif pool_type in ["max", "mean"]:
            classifier_input_dim = lstm_output_dim
        elif pool_type == "last":
            classifier_input_dim = lstm_output_dim
        else:
            classifier_input_dim = lstm_output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, num_classes)
        )
        
        # Layer normalization for residual connections
        if use_residual:
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of LSTM classifier.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        batch_size = input_ids.size(0)
        
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM encoding
        if attention_mask is not None:
            # Pack sequences for efficient processing
            lengths = attention_mask.sum(dim=1).cpu()
            packed_embedded = pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Apply residual connection if specified
        if self.use_residual and self.embedding_dim == lstm_output.size(-1):
            lstm_output = self.layer_norm(lstm_output + embedded)
        
        # Pooling
        if self.pool_type == "attention" and self.use_attention:
            # Attention-based pooling
            attention_scores = self.attention(lstm_output)
            
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    ~attention_mask.unsqueeze(-1), float('-inf')
                )
            
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled = torch.sum(lstm_output * attention_weights, dim=1)
        
        elif self.pool_type == "max":
            # Max pooling
            if attention_mask is not None:
                lstm_output = lstm_output.masked_fill(
                    ~attention_mask.unsqueeze(-1), float('-inf')
                )
            pooled, _ = torch.max(lstm_output, dim=1)
        
        elif self.pool_type == "mean":
            # Mean pooling
            if attention_mask is not None:
                lstm_output = lstm_output * attention_mask.unsqueeze(-1)
                pooled = lstm_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = torch.mean(lstm_output, dim=1)
        
        elif self.pool_type == "last":
            # Last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
                hidden = hidden[-1]  # Last layer
                pooled = torch.cat([hidden[0], hidden[1]], dim=-1)
            else:
                pooled = hidden[-1]
        
        else:
            # Default to mean pooling
            pooled = torch.mean(lstm_output, dim=1)
        
        # Dropout
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class LSTMDataset(Dataset):
    """Dataset class for LSTM model."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of input texts
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
        
        # Tokenize
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LSTMBaseline:
    """
    LSTM baseline for text classification.
    
    This class provides:
    - Multiple LSTM architectures
    - Training with early stopping
    - Hyperparameter configuration
    - Model checkpointing
    - Comprehensive evaluation
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_residual: bool = False,
        pool_type: str = "attention",
        max_length: int = 256,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        Initialize LSTM baseline.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of classes
            dropout: Dropout rate
            bidirectional: Whether to use BiLSTM
            use_attention: Whether to use attention
            use_residual: Whether to use residual connections
            pool_type: Type of pooling
            max_length: Maximum sequence length
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
            device: Device to use
            seed: Random seed
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.pool_type = pool_type
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        
        # Set seed
        set_seed(seed)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Results storage
        self.results = {
            "train_history": [],
            "val_history": [],
            "test_metrics": {},
            "best_epoch": 0,
            "training_time": 0
        }
        
        # Initialize registry and metrics
        self.registry = Registry()
        self.factory = Factory()
        self.metrics_calculator = ClassificationMetrics()
        
        logger.info(f"Initialized LSTM baseline (BiLSTM={bidirectional}, Attention={use_attention})")
    
    def _build_model(self):
        """Build LSTM model."""
        self.model = LSTMTextClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention,
            use_residual=self.use_residual,
            pool_type=self.pool_type
        ).to(self.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            model_max_length=self.max_length
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            
        Returns:
            Training results
        """
        logger.info("Training LSTM model")
        
        # Build model
        self._build_model()
        
        # Create datasets
        train_dataset = LSTMDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = LSTMDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        import time
        start_time = time.time()
        
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        early_stopping_patience = 5
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
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
                        labels = batch['labels'].to(self.device)
                        
                        logits = self.model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                
                # Learning rate scheduling
                self.scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    self.results["best_epoch"] = epoch
                    patience_counter = 0
                    
                    # Save best model
                    self._save_checkpoint(epoch, avg_val_loss)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Store history
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
            
            # Store history
            self.results["train_history"].append({
                "epoch": epoch,
                "loss": avg_train_loss,
                "accuracy": train_acc
            })
        
        self.results["training_time"] = time.time() - start_time
        logger.info(f"Training completed in {self.results['training_time']:.2f} seconds")
        
        return self.results
    
    def evaluate(
        self,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Test metrics
        """
        logger.info("Evaluating LSTM model")
        
        self.model.eval()
        
        # Create dataset and loader
        test_dataset = LSTMDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
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
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'bidirectional': self.bidirectional,
                'use_attention': self.use_attention
            }
        }
        
        checkpoint_path = Path("outputs/checkpoints/lstm_best.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Build model with saved config
        config = checkpoint['config']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.use_attention = config['use_attention']
        
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": f"{'Bi' if self.bidirectional else ''}LSTM{'_Attention' if self.use_attention else ''}",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "pool_type": self.pool_type,
            "training_time": self.results.get("training_time", 0),
            "best_epoch": self.results.get("best_epoch", 0),
            "performance": {
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0),
                "test_f1": self.results.get("test_metrics", {}).get("f1_weighted", 0)
            }
        }


def run_lstm_experiments():
    """Run LSTM experiments."""
    logger.info("Starting LSTM experiments")
    
    # Load data
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    # Create and train model
    model = LSTMBaseline(
        bidirectional=True,
        use_attention=True,
        num_epochs=10,
        batch_size=32
    )
    
    model.train(
        train_data["texts"],
        train_data["labels"],
        val_data["texts"],
        val_data["labels"]
    )
    
    # Evaluate
    test_metrics = model.evaluate(
        test_data["texts"],
        test_data["labels"]
    )
    
    # Get summary
    summary = model.get_summary()
    
    logger.info(f"Summary: {summary}")
    
    return test_metrics


if __name__ == "__main__":
    run_lstm_experiments()
