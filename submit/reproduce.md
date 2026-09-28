# Reproduce the ML submission bundle

Run from the repository root in the pinned project environment.

```sh
python -m pip install -r requirements.txt
python submit/build/make_submit.py --verify
```

The builder reads committed post-fix reports, the simulator configuration/source, and the reliability-curve artifact. It does not open the locked dataset. It regenerates tables and figures; `--verify` builds twice in temporary directories, compares all CSV bytes, checks source gates, and then writes `submit/`.

To independently recheck the feature-leakage gate and test suite:

```sh
python -m pytest tests/test_build_feature_store.py::test_label_blind_and_permuted_labels_leave_features_identical -q
python -m pytest tests/ -q
```

The final test results are read from `reports/final_eval.json` and `reports/final_eval.md`; do not rerun `scripts/final_eval.py` because the one-use evaluation has already been consumed. Dataset construction statistics come from `data/synthetic/dataset_manifest.json`; model reports are committed under `reports/`.
