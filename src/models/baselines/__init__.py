"""Classical baselines for AG News.

Provides:

* :class:`TfidfClassifier` - TF-IDF + Logistic Regression / Linear SVM.
* :class:`FastTextClassifier` - Facebook FastText supervised classifier.
"""

from src.models.baselines.fasttext_baseline import FastTextClassifier
from src.models.baselines.tfidf import TfidfClassifier

__all__ = ["TfidfClassifier", "FastTextClassifier"]
