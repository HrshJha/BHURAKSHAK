# Research slide evidence mode

**Mode A — locked synthetic results allowed.** `reports/final_eval.md` exists; `reports/test_lock.json` records `evals_run: 1`; `tests/test_build_feature_store.py::test_label_blind_and_permuted_labels_leave_features_identical` passes; the full-suite log `reports/pytest_phase8_pretest.txt` is green. Results on this slide are limited to the single locked synthetic evaluation. The split is the unseen scenario-regime holdout in `data/synthetic/dataset_manifest.json`. No real-mine validation is claimed.
