"""
CNN Baseline for AG News Text Classification
================================================================================
This module implements Convolutional Neural Network baselines for text classification,
including multi-channel architectures and various pooling strategies.

CNNs capture local patterns and n-gram features through convolutional filters,
providing efficient and effective text classification.

References:
    - Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
    - Zhang, X., et al. (2015). Character-level Convolutional Networks for Text Classification
    - Conneau, A., et al. (2017). Very Deep Convolutional Networks for Text Classification

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class TextCNN(nn.Module):
    """
    CNN for text classification.
    
    Implements Kim's CNN architecture with:
    - Multiple filter sizes
    - Multiple channels (static/dynamic)
    - Various pooling strategies
    - Highway connections
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_classes: int = 4,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.5,
        use_static_embedding: bool = False,
        use_multichannel: bool = False,
        pool_type: str = "max",
        use_batch_norm: bool = True,
        use_highway: bool = False,
        freeze_embeddings: bool = False,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize TextCNN.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_classes: Number of output classes
            filter_sizes: List of filter window sizes
            num_filters: Number of filters per size
            dropout: Dropout probability
            use_static_embedding: Whether to use static embedding channel
            use_multichannel: Whether to use multi-channel architecture
            pool_type: Pooling type ("max", "mean", "k_max")
            use_batch_norm: Whether to use batch normalization
            use_highway: Whether to use highway connections
            freeze_embeddings: Whether to freeze embeddings
            pretrained_embeddings: Pretrained embedding weights
        """
        super(TextCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.use_multichannel = use_multichannel
        self.pool_type = pool_type
        self.use_highway = use_highway
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Static embedding channel
        if use_static_embedding or use_multichannel:
            self.static_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.static_embedding.weight.data.copy_(pretrained_embeddings)
            self.static_embedding.weight.requires_grad = False
            in_channels = 2 if use_multichannel else 1
        else:
            in_channels = 1
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm2d(num_filters)
                for _ in filter_sizes
            ])
        else:
            self.batch_norms = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Calculate output dimension
        total_filters = num_filters * len(filter_sizes)
        
        # Highway layer
        if use_highway:
            self.highway = Highway(total_filters)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        
        # Add channel dimension
        x = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        # Multi-channel
        if hasattr(self, 'static_embedding'):
            static_embedded = self.static_embedding(input_ids)
            static_embedded = static_embedded.unsqueeze(1)
            
            if self.use_multichannel:
                x = torch.cat([x, static_embedded], dim=1)  # [batch_size, 2, seq_len, embedding_dim]
        
        # Convolution + activation + pooling
        conv_outputs = []
        
        for i, conv in enumerate(self.convs):
            conv_out = conv(x)  # [batch_size, num_filters, conv_seq_len, 1]
            
            # Batch normalization
            if self.batch_norms is not None:
                conv_out = self.batch_norms[i](conv_out)
            
            # Activation
            conv_out = F.relu(conv_out)
            
            # Pooling
            if self.pool_type == "max":
                pooled = F.max_pool2d(conv_out, kernel_size=(conv_out.size(2), 1))
            elif self.pool_type == "mean":
                pooled = F.avg_pool2d(conv_out, kernel_size=(conv_out.size(2), 1))
            else:  # k-max pooling
                k = 3
                pooled, _ = conv_out.topk(k, dim=2)
                pooled = pooled.view(conv_out.size(0), -1, 1, 1)
            
            pooled = pooled.squeeze()  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, total_filters]
        
        # Highway connection
        if self.use_highway:
            x = self.highway(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class Highway(nn.Module):
    """Highway network layer."""
    
    def __init__(self, size: int):
        """
        Initialize Highway layer.
        
        Args:
            size: Input/output dimension
        """
        super(Highway, self).__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        transform = F.relu(self.transform(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * transform + (1 - gate) * x


class CNNDataset(Dataset):
    """Dataset class for CNN model."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        """Initialize dataset."""
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CNNBaseline:
    """
    CNN baseline for text classification.
    
    Provides:
    - Multiple CNN architectures
    - Training with early stopping
    - Hyperparameter configuration
    - Model evaluation
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 300,
        num_classes: int = 4,
        filter_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.5,
        use_multichannel: bool = False,
        pool_type: str = "max",
        use_batch_norm: bool = True,
        use_highway: bool = False,
        max_length: int = 256,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        seed: int = 42
    ):
        """Initialize CNN baseline."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.use_multichannel = use_multichannel
        self.pool_type = pool_type
        self.use_batch_norm = use_batch_norm
        self.use_highway = use_highway
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.device = device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        
        set_seed(seed)
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        self.results = {
            "train_history": [],
            "val_history": [],
            "test_metrics": {},
            "training_time": 0
        }
        
        self.registry = Registry()
        self.factory = Factory()
        self.metrics_calculator = ClassificationMetrics()
        
        logger.info(f"Initialized CNN baseline with filters {filter_sizes}")
    
    def _build_model(self):
        """Build CNN model."""
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            filter_sizes=self.filter_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout,
            use_multichannel=self.use_multichannel,
            pool_type=self.pool_type,
            use_batch_norm=self.use_batch_norm,
            use_highway=self.use_highway
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            model_max_length=self.max_length
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.1
        )
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Train CNN model."""
        logger.info("Training CNN model")
        
        self._build_model()
        
        train_dataset = CNNDataset(
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
            val_dataset = CNNDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        criterion = nn.CrossEntropyLoss()
        
        import time
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
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
            
            self.scheduler.step()
        
        self.results["training_time"] = time.time() - start_time
        logger.info(f"Training completed in {self.results['training_time']:.2f} seconds")
        
        return self.results
    
    def evaluate(
        self,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluate model."""
        logger.info("Evaluating CNN model")
        
        self.model.eval()
        
        test_dataset = CNNDataset(
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": f"TextCNN_{self.pool_type}",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "filter_sizes": self.filter_sizes,
            "num_filters": self.num_filters,
            "use_multichannel": self.use_multichannel,
            "training_time": self.results.get("training_time", 0),
            "performance": {
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0),
                "test_f1": self.results.get("test_metrics", {}).get("f1_weighted", 0)
            }
        }


def run_cnn_experiments():
    """Run CNN experiments."""
    logger.info("Starting CNN experiments")
    
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    model = CNNBaseline(
        filter_sizes=[3, 4, 5],
        num_filters=100,
        num_epochs=10
    )
    
    model.train(
        train_data["texts"],
        train_data["labels"],
        val_data["texts"],
        val_data["labels"]
    )
    
    test_metrics = model.evaluate(
        test_data["texts"],
        test_data["labels"]
    )
    
    summary = model.get_summary()
    logger.info(f"Summary: {summary}")
    
    return test_metrics


if __name__ == "__main__":
    run_cnn_experiments()
