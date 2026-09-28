# Phase 0 environment drift

Interpreter: Python 3.14.3. `pip freeze` is preserved in `reports/env_before.txt`; requirements pins are in `requirements.txt`.

| Package | Required | Installed |
|---|---:|---:|
| numpy | 2.3.5 | 2.4.3 |
| scipy | 1.16.3 | 1.17.1 |
| pandas | 2.3.3 | 3.0.3 |
| pyarrow | 21.0.0 | 24.0.0 |
| scikit-learn | 1.6.1 | 1.8.0 |
| xgboost | 3.2.0 | 2.1.4 |
| torch | 2.10.0 | 2.11.0 |
| statsmodels | 0.14.5 | MISSING |
| joblib | 1.5.2 | 1.5.3 |
| shap | 0.51.0 | 0.52.0 |
| notebook | 7.4.5 | MISSING |
| jupyter-core | 5.8.1 | MISSING |
| nbformat | 5.10.4 | MISSING |
| nbclient | 0.10.2 | MISSING |
| matplotlib | 3.10.6 | 3.11.0 |
| seaborn | 0.13.2 | MISSING |
| pytest | 8.4.2 | 9.0.3 |
| tqdm | 4.67.1 | 4.67.3 |

The most relevant drift is NumPy 2.4.3, pandas 3.0.3, scikit-learn 1.8.0, XGBoost 2.1.4, and PyTorch 2.11.0 versus the pins. This environment differs substantially from the repo lock and could explain some version-sensitive behavior; source changes still need bisect evidence.

Baseline `python3 -m pytest tests/ -q --tb=short`: 17 failures, captured verbatim in `reports/pytest_before.txt`.
