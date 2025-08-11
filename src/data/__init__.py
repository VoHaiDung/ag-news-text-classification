# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Data loading, exploratory analysis and preparation.

The submodules of :mod:`src.data` cover:

* :mod:`src.data.loader` - dataset acquisition and stratified splitting;
* :mod:`src.data.eda` - class distribution and text length statistics;
* :mod:`src.data.visualization` - word clouds and n-gram bar charts;
* :mod:`src.data.cleanlab_audit` - label-noise detection;
* :mod:`src.data.topic_modeling` - unsupervised topic discovery via BERTopic;
* :mod:`src.data.translation` - English-to-target OPUS-MT translation
  pipeline driving Phase 5A (Vietnamese) and Phase 5B (French);
* :mod:`src.data.back_translation` - target-to-EN-to-target
  back-translation augmentation for Phase 5A and Phase 5B.
"""

from src.data.loader import AGNewsLoader, DatasetSplits

__all__ = ["AGNewsLoader", "DatasetSplits"]
