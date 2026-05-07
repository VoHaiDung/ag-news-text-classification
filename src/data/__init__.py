"""Data loading, exploratory analysis and preparation.

The submodules of :mod:`src.data` cover:

* :mod:`src.data.loader` - dataset acquisition and stratified splitting;
* :mod:`src.data.eda` - class distribution and text length statistics;
* :mod:`src.data.visualization` - word clouds and n-gram bar charts;
* :mod:`src.data.cleanlab_audit` - label-noise detection;
* :mod:`src.data.topic_modeling` - unsupervised topic discovery via BERTopic;
* :mod:`src.data.translation` - English-Vietnamese translation pipeline (Phase 5);
* :mod:`src.data.back_translation` - back-translation augmentation (Phase 5).
"""

from src.data.loader import AGNewsLoader, DatasetSplits

__all__ = ["AGNewsLoader", "DatasetSplits"]
