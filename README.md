# Moderation Service

This repository currently focuses on the EfficientNet GPU training and validation pipeline for image moderation tasks.

## Quick Start

1. Clone the repository and enter the project:

```bash
git clone https://github.com/whispr-messenger/moderation-service.git
cd moderation-service
```

2. Switch to the active module directory:

```bash
cd src/efficientnet_lite_gpu
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run pipeline actions:

```bash
python main.py --action train
python main.py --action eval
python main.py --action test
```

## Dataset Fetch (Optional)

If you only need the dataset fetch tool dependencies:

```bash
pip install -r requirements-fetch-only.txt
```

Dry-run fetch command:

```bash
python -m tools.fetch_google_dataset --dry-run
```

Windows helpers:

- `run_fetch_google_dataset.bat`
- `run_fetch_google_dataset_dry_run.bat`

## Documentation

- Project index: `documentation/PROJECT_INDEX.md`
- Module guide: `src/efficientnet_lite_gpu/README.md`
- Windows long path setup: `documentation/WINDOWS_LONG_PATHS.md`
- Architecture reference: `documentation/1_architecture/1_system_design.md`
