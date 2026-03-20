# Deprecation And Primary Path Map

## Goal

Track duplicated or ambiguous scripts and define a primary path without deleting files yet.

## Current Decisions

### Test Action Entrypoint

- Primary path: `src/efficientnet_lite_gpu/test/test.py`
- Legacy/secondary path: `src/efficientnet_lite_gpu/test.py`
- Reason:
  - `main.py` dispatch imports `from test.test import run as test_run`.
- Policy:
  - Keep `src/efficientnet_lite_gpu/test.py` for now.
  - Do not introduce new usage of this file in docs or scripts.

### Training Pipeline

- Primary path: `src/efficientnet_lite_gpu/train/train.py`
- Legacy/secondary path: `src/efficientnet_lite_gpu/train/run_train.py`
- Reason:
  - `main.py` currently dispatches `from train.train import run as train_run`.
- Policy:
  - Keep both files temporarily to avoid breaking manual workflows.
  - New docs and examples should reference `main.py --action train`.

### Dataset Fetch Naming

- Current path: `src/efficientnet_lite_gpu/tools/fetch_google_dataset.py`
- Behavior note:
  - Script supports multiple engines (duckduckgo/bing/google), not only Google.
- Migration direction:
  - Future rename can use behavior-based naming with compatibility wrapper.
  - No rename in this phase.

## Review Trigger

Revisit this map when:

- all runtime and CI references are consolidated,
- legacy files have no active references,
- wrapper-based migration is prepared.
