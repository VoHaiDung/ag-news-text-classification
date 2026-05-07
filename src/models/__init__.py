"""Model implementations.

The package is split into three sub-packages:

* :mod:`src.models.baselines` - classical baselines (TF-IDF + LR/SVM, FastText);
* :mod:`src.models.transformers` - transformer wrappers (DeBERTa-v3, ModernBERT,
  mDeBERTa-v3, XLM-R);
* :mod:`src.models.setfit_model` - SetFit few-shot classifier.

Every model exposes a uniform interface (``fit``, ``predict``,
``predict_proba``, ``save``, ``load``) so that the evaluation pipeline can
treat them interchangeably.
"""
