# EfficientNet Lite GPU Module

This directory contains the active training/validation/test pipeline used by this repository.

## Working Directory

Run all commands from this folder:

```bash
cd src/efficientnet_lite_gpu
```

## Dependencies

Install full runtime dependencies (train/eval/test + tools):

```bash
pip install -r requirements.txt
```

Install fetch-only dependency set (for dataset crawler only):

```bash
pip install -r requirements-fetch-only.txt
```

## Main Entrypoint

`main.py` dispatches to the following actions:

- `train`
- `eval`
- `test`

Examples:

```bash
python main.py --action train
python main.py --action eval
python main.py --action test
```

## Dataset Fetch Tool

Script:

- `tools/fetch_google_dataset.py`

Config file:

- `tools/dataset_download_config.yaml`

Commands:

```bash
python -m tools.fetch_google_dataset --dry-run
python -m tools.fetch_google_dataset
```

Windows helper scripts:

- `run_fetch_google_dataset.bat`
- `run_fetch_google_dataset_dry_run.bat`

Safety note:

- If config uses `balance: true`, the tool can delete extra images to rebalance classes.
- Always run `--dry-run` first and backup datasets before enabling balancing.

## Related Docs

- Root entry: `../../README.md`
- Project index: `../../documentation/PROJECT_INDEX.md`
- Windows long path fix: `../../documentation/WINDOWS_LONG_PATHS.md`
