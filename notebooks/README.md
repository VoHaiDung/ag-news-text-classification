# Notebooks

Exploratory and reporting notebooks live here. They are intentionally thin
wrappers around the library code under ``src/`` so the same logic is
reachable from both the phase scripts and an interactive session.

| File                          | Purpose                                                |
|-------------------------------|--------------------------------------------------------|
| ``01_eda.ipynb``              | Reproduce Phase 2 plots and tables interactively.      |
| ``02_baselines.ipynb``        | Inspect classical-baseline confusion matrices.         |
| ``03_transformers.ipynb``     | Walk through DeBERTa-v3 / ModernBERT training curves.  |
| ``04_setfit_curve.ipynb``     | Render the SetFit data-efficiency learning curve.      |
| ``05_explainability.ipynb``   | Generate one SHAP and one LIME explanation on demand.  |

Notebooks are not committed with their cell outputs. Use ``nbstripout`` or
``jupyter nbconvert --clear-output`` before committing changes.
