# Structure And Naming Conventions

## Scope

These conventions define responsibilities for the current active module `src/efficientnet_lite_gpu` without changing business behavior.

## Directory Responsibilities

- `src/efficientnet_lite_gpu/main.py`
  - Single orchestration entrypoint for action dispatch (`train`, `eval`, `test`).
- `src/efficientnet_lite_gpu/train/`
  - Training pipeline implementations and training-focused helpers.
- `src/efficientnet_lite_gpu/validation/`
  - Evaluation/validation stage logic.
- `src/efficientnet_lite_gpu/test/`
  - Test action implementation invoked by `main.py`.
- `src/efficientnet_lite_gpu/tools/`
  - Utility scripts (dataset fetch, config generation, environment validation).
- `documentation/`
  - Design docs, operational notes, migration/deprecation records.

## Entrypoint Rules

- Recommended runtime entrypoint for pipeline actions: `python main.py --action <train|eval|test>`.
- Recommended runtime entrypoint for dataset fetch: `python -m tools.fetch_google_dataset`.
- Windows convenience scripts may exist, but CLI entrypoints are canonical.

## Naming Rules

- New scripts should use behavior-oriented names instead of provider-specific names.
  - Example: prefer `fetch_web_dataset.py` over engine-specific names.
- Avoid duplicate top-level and package-level script names for the same purpose.
  - Example risk: both `test.py` and `test/test.py`.
- Keep action names aligned with `main.py` dispatch keys.

## Migration Strategy For Existing Files

- For legacy names that cannot be changed immediately:
  - Keep current file path for compatibility.
  - Add a deprecation note and target replacement in `documentation/DEPRECATION_MAP.md`.
  - If a rename is later required, keep a compatibility wrapper during transition.
