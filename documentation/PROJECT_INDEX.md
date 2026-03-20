# Project Documentation Index

## Purpose

This index centralizes the active project entry points, runtime commands, and supporting documents.

## Active Module

- Main module: `src/efficientnet_lite_gpu`
- Pipeline entrypoint: `src/efficientnet_lite_gpu/main.py`
- Dataset fetch entrypoint: `src/efficientnet_lite_gpu/tools/fetch_google_dataset.py`

## Recommended Run Flow

1. `cd src/efficientnet_lite_gpu`
2. `pip install -r requirements.txt`
3. Run one action:
   - `python main.py --action train`
   - `python main.py --action eval`
   - `python main.py --action test`

## Dependencies

- Main runtime dependencies: `src/efficientnet_lite_gpu/requirements.txt`
- Fetch-only dependencies: `src/efficientnet_lite_gpu/requirements-fetch-only.txt`

## Dataset Fetch

- CLI:
  - `python -m tools.fetch_google_dataset`
  - `python -m tools.fetch_google_dataset --dry-run`
- Windows scripts:
  - `src/efficientnet_lite_gpu/run_fetch_google_dataset.bat`
  - `src/efficientnet_lite_gpu/run_fetch_google_dataset_dry_run.bat`

## Environment Notes

- Windows long path issue fix:
  - `documentation/WINDOWS_LONG_PATHS.md`

## Architecture Reference

- `documentation/1_architecture/1_system_design.md`
- Structure conventions: `documentation/STRUCTURE_CONVENTIONS.md`

## Repository Cleanup Notes

- Legacy or duplicated scripts are tracked in:
  - `documentation/DEPRECATION_MAP.md`
