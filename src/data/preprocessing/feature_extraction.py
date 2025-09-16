"""
Feature Extraction Module
=========================

Implements various feature extraction techniques following:
- Mikolov et al. (2013): "Distributed Representations of Words and Phrases"
- Pennington et al. (2014): "GloVe: Global Vectors for Word Representation"
- Bojanowski et al. (2017): "Enriching Word Vectors with Subword Information"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import AutoModel, AutoTokenizer
import spacy

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from configs.constants import MAX_SEQUENCE_LENGTH

logger = setup_logging(__name__)

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    
    # Feature types
    use_tfidf: bool = True
    use_bow: bool = False
    use_embeddings: bool = True
    use_statistical: bool = True
    use_linguistic: bool = False
    
    # TF-IDF parameters
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.95
    
    # Embedding parameters
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_pooling: str = "mean"  # mean, max, cls
    
    # Dimension reduction
    reduce_dims: bool = False
    n_components: int = 300
    
    # Caching
    cache_features: bool = True
    cache_dir: Optional[Path] = None

class FeatureExtractor:
    """
    Extract features from text for classical ML and hybrid models.
    
    Implements feature extraction following:
    - Zhang et al. (2015): "Character-level Convolutional Networks"
    - Joulin et al. (2017): "Bag of Tricks for Efficient Text Classification"
    """
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureExtractionConfig()
        
        # Initialize extractors
        self.tfidf_vectorizer = None
        self.bow_vectorizer = None
        self.embedding_model = None
        self.tokenizer = None
        self.svd = None
        
        # Initialize spacy for linguistic features
        if self.config.use_linguistic:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("Spacy model not found. Linguistic features disabled.")
                self.config.use_linguistic = False
        
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize feature extractors based on config."""
        
        # TF-IDF
        if self.config.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=self.config.tfidf_min_df,
                max_df=self.config.tfidf_max_df,
                strip_accents='unicode',
                stop_words='english'
            )
        
        # Bag of Words
        if self.config.use_bow:
            self.bow_vectorizer = CountVectorizer(
                max_features=self.config.tfidf_max_features,
                ngram_range=self.config.tfidf_ngram_range,
                min_df=self.config.tfidf_min_df,
                max_df=self.config.tfidf_max_df
            )
        
        # Embeddings
        if self.config.use_embeddings:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.embedding_model
                )
                self.embedding_model = AutoModel.from_pretrained(
                    self.config.embedding_model
                )
                self.embedding_model.eval()
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.config.use_embeddings = False
        
        # Dimension reduction
        if self.config.reduce_dims:
            self.svd = TruncatedSVD(n_components=self.config.n_components)
    
    def extract_tfidf_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Extract TF-IDF features.
        
        Args:
            texts: List of texts
            fit: Whether to fit the vectorizer
            
        Returns:
            TF-IDF feature matrix
        """
        if not self.config.use_tfidf or self.tfidf_vectorizer is None:
            return np.array([])
        
        if fit:
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        return features.toarray()
    
    def extract_bow_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Extract Bag of Words features."""
        if not self.config.use_bow or self.bow_vectorizer is None:
            return np.array([])
        
        if fit:
            features = self.bow_vectorizer.fit_transform(texts)
        else:
            features = self.bow_vectorizer.transform(texts)
        
        return features.toarray()
    
    def extract_embedding_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract embedding features using pre-trained model.
        
        Following:
        - Reimers & Gurevych (2019): "Sentence-BERT"
        """
        if not self.config.use_embeddings or self.embedding_model is None:
            return np.array([])
        
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                
                # Pool embeddings
                if self.config.embedding_pooling == "mean":
                    embedding = outputs.last_hidden_state.mean(dim=1)
                elif self.config.embedding_pooling == "max":
                    embedding = outputs.last_hidden_state.max(dim=1)[0]
                else:  # cls
                    embedding = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(embedding.squeeze().numpy())
        
        return np.array(embeddings)
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract statistical features from text.
        
        Following:
        - Yogatama et al. (2015): "Learning to Compose Words into Sentences"
        """
        if not self.config.use_statistical:
            return np.array([])
        
        features = []
        
        for text in texts:
            text_features = []
            
            # Length features
            text_features.append(len(text))  # Character count
            text_features.append(len(text.split()))  # Word count
            text_features.append(len(text.split('.')))  # Sentence count
            
            # Average lengths
            words = text.split()
            if words:
                text_features.append(np.mean([len(w) for w in words]))  # Avg word length
            else:
                text_features.append(0)
            
            # Punctuation counts
            text_features.append(text.count('.'))
            text_features.append(text.count(','))
            text_features.append(text.count('!'))
            text_features.append(text.count('?'))
            
            # Capitalization
            text_features.append(sum(1 for c in text if c.isupper()))
            text_features.append(sum(1 for w in words if w[0].isupper() if w else 0))
            
            # Digit count
            text_features.append(sum(1 for c in text if c.isdigit()))
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract linguistic features using spaCy."""
        if not self.config.use_linguistic or not hasattr(self, 'nlp'):
            return np.array([])
        
        features = []
        
        for text in texts:
            doc = self.nlp(text[:1000000])  # Limit text length for spaCy
            
            text_features = []
            
            # POS tag counts
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            
            # Major POS categories
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET']:
                text_features.append(pos_counts.get(pos, 0))
            
            # Named entities
            ent_counts = {}
            for ent in doc.ents:
                ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
            
            # Major entity types
            for ent_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']:
                text_features.append(ent_counts.get(ent_type, 0))
            
            # Dependency features
            text_features.append(len(list(doc.sents)))  # Sentence count
            
            features.append(text_features)
        
        return np.array(features)
    
    def extract_all_features(
        self,
        texts: List[str],
        fit: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract all configured features.
        
        Args:
            texts: List of texts
            fit: Whether to fit extractors
            
        Returns:
            Dictionary of feature matrices
        """
        features = {}
        
        # TF-IDF
        if self.config.use_tfidf:
            features['tfidf'] = self.extract_tfidf_features(texts, fit)
        
        # Bag of Words
        if self.config.use_bow:
            features['bow'] = self.extract_bow_features(texts, fit)
        
        # Embeddings
        if self.config.use_embeddings:
            features['embeddings'] = self.extract_embedding_features(texts)
        
        # Statistical
        if self.config.use_statistical:
            features['statistical'] = self.extract_statistical_features(texts)
        
        # Linguistic
        if self.config.use_linguistic:
            features['linguistic'] = self.extract_linguistic_features(texts)
        
        return features
    
    def combine_features(
        self,
        features: Dict[str, np.ndarray],
        reduce_dims: bool = None
    ) -> np.ndarray:
        """
        Combine multiple feature types.
        
        Args:
            features: Dictionary of feature matrices
            reduce_dims: Whether to reduce dimensions
            
        Returns:
            Combined feature matrix
        """
        # Concatenate all features
        feature_list = []
        for name, feat in features.items():
            if feat.size > 0:
                feature_list.append(feat)
        
        if not feature_list:
            return np.array([])
        
        combined = np.concatenate(feature_list, axis=1)
        
        # Reduce dimensions if configured
        if reduce_dims is None:
            reduce_dims = self.config.reduce_dims
        
        if reduce_dims and self.svd is not None:
            if combined.shape[0] > 1:  # Need at least 2 samples for SVD
                combined = self.svd.fit_transform(combined)
        
        return combined
    
    def save_extractors(self, save_dir: Union[str, Path]):
        """Save fitted extractors."""
        import joblib
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, save_dir / "tfidf_vectorizer.pkl")
        
        if self.bow_vectorizer:
            joblib.dump(self.bow_vectorizer, save_dir / "bow_vectorizer.pkl")
        
        if self.svd:
            joblib.dump(self.svd, save_dir / "svd.pkl")
        
        logger.info(f"Feature extractors saved to {save_dir}")
    
    def load_extractors(self, load_dir: Union[str, Path]):
        """Load fitted extractors."""
        import joblib
        load_dir = Path(load_dir)
        
        tfidf_path = load_dir / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            self.tfidf_vectorizer = joblib.load(tfidf_path)
        
        bow_path = load_dir / "bow_vectorizer.pkl"
        if bow_path.exists():
            self.bow_vectorizer = joblib.load(bow_path)
        
        svd_path = load_dir / "svd.pkl"
        if svd_path.exists():
            self.svd = joblib.load(svd_path)
        
        logger.info(f"Feature extractors loaded from {load_dir}")
