import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from datasets import load_from_disk
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

def train_naive_bayes():
    """Train Naive Bayes baseline"""
    logger.info("Loading data...")
    
    # Load data
    dataset = load_from_disk("data/raw/ag_news")
    train_texts = [item['text'] for item in dataset['train']]
    train_labels = [item['label'] for item in dataset['train']]
    test_texts = [item['text'] for item in dataset['test']]
    test_labels = [item['label'] for item in dataset['test']]
    
    logger.info("Creating pipeline...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    
    # Train
    logger.info("Training Naive Bayes...")
    pipeline.fit(train_texts, train_labels)
    
    # Evaluate
    train_score = pipeline.score(train_texts, train_labels)
    test_score = pipeline.score(test_texts, test_labels)
    
    logger.info(f"Train accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Detailed evaluation
    test_preds = pipeline.predict(test_texts)
    report = classification_report(test_labels, test_preds, 
                                 target_names=['World', 'Sports', 'Business', 'Sci/Tech'])
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save model
    output_dir = Path("outputs/models/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "naive_bayes.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return pipeline, test_score

if __name__ == "__main__":
    train_naive_bayes()
