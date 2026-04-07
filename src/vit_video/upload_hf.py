from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError as e:
    raise ImportError(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from e

MODEL_EXTENSIONS = {".pt", ".onnx", ".tflite", ".mlpackage"}
METADATA_EXTENSIONS = {".json"}


def _build_readme(model_card_path: Path) -> str:
    """Generate a HF-style README.md from the model_card.json."""
    with open(model_card_path, encoding="utf-8") as f:
        card = json.load(f)

    classes = card.get("classes", [])
    inp = card.get("input_shape", {})
    norm = card.get("normalization", {})
    evaluation = card.get("evaluation", {})
    training = card.get("training", {})
    exported = card.get("exported_formats", [])

    lines = [
        "---",
        "license: apache-2.0",
        "tags:",
        "  - video-classification",
        "  - pytorch",
        f"  - {card.get('backbone', 'vit')}",
        "---",
        "",
        f"# {card.get('model_name', 'Video Classifier')}",
        "",
        f"Video classification model for **{', '.join(classes)}** detection.",
        "",
        "## Model Details",
        "",
        f"- **Task:** {card.get('task', 'video_classification')}",
        f"- **Backbone:** {card.get('backbone', 'unknown')}",
        f"- **Framework:** {card.get('framework', 'pytorch')}",
        f"- **Classes:** {', '.join(classes)} ({card.get('num_classes', len(classes))} classes)",
        f"- **Input format:** `{card.get('input_format', 'BTCHW')}` — "
        f"batch={inp.get('batch')}, frames={inp.get('frames')}, "
        f"channels={inp.get('channels')}, height={inp.get('height')}, width={inp.get('width')}",
        "",
        "## Normalization",
        "",
        f"- **Mean:** {norm.get('mean', [])}",
        f"- **Std:** {norm.get('std', [])}",
        "",
        f"## Exported Formats",
        "",
    ]
    for fmt in exported:
        lines.append(f"- {fmt}")

    if evaluation:
        lines += [
            "",
            "## Evaluation",
            "",
            f"- **Accuracy:** {evaluation.get('accuracy', 'N/A'):.4f}"
            if isinstance(evaluation.get("accuracy"), (int, float))
            else f"- **Accuracy:** {evaluation.get('accuracy', 'N/A')}",
            f"- **F1 (macro):** {evaluation.get('f1_macro', 'N/A'):.4f}"
            if isinstance(evaluation.get("f1_macro"), (int, float))
            else f"- **F1 (macro):** {evaluation.get('f1_macro', 'N/A')}",
            f"- **Samples:** {evaluation.get('num_samples', 'N/A')}",
        ]
        per_class = evaluation.get("per_class", {})
        if per_class:
            lines += ["", "| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
            for cls, metrics in per_class.items():
                lines.append(
                    f"| {cls} | {metrics['precision']:.4f} | "
                    f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |"
                )

    if training:
        lines += [
            "",
            "## Training",
            "",
            f"- **Backbone:** {training.get('backbone', 'N/A')}",
            f"- **Epochs:** {training.get('epochs_completed', 'N/A')}/{training.get('epochs_requested', 'N/A')}",
            f"- **Learning rate:** {training.get('lr', 'N/A')}",
            f"- **Best val accuracy:** {training.get('best_val_accuracy', 'N/A'):.2f}%"
            if isinstance(training.get("best_val_accuracy"), (int, float))
            else f"- **Best val accuracy:** {training.get('best_val_accuracy', 'N/A')}",
            f"- **Train samples:** {training.get('train_samples', 'N/A')}",
            f"- **Val samples:** {training.get('val_samples', 'N/A')}",
        ]

    lines += [
        "",
        "## Usage",
        "",
        "```python",
        "import torch",
        "",
        "model = torch.jit.load('best_food_classifier.pt')",
        "model.eval()",
        f"# Input: torch.randn({inp.get('batch', 1)}, {inp.get('frames', 8)}, "
        f"{inp.get('channels', 3)}, {inp.get('height', 224)}, {inp.get('width', 224)})",
        "```",
        "",
    ]
    return "\n".join(lines)


def upload(
    repo_id: str,
    export_dir: Path,
    private: bool = False,
    commit_message: str = "Upload exported model",
) -> str:
    """Upload all model files from export_dir to a HF repo. Returns the repo URL."""
    api = HfApi()

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"Repository: https://huggingface.co/{repo_id}")

    # Collect files to upload
    files = sorted(export_dir.iterdir())
    model_files = [f for f in files if f.suffix in MODEL_EXTENSIONS and f.is_file()]
    metadata_files = [f for f in files if f.suffix in METADATA_EXTENSIONS and f.is_file()]

    if not model_files:
        raise FileNotFoundError(f"No model files found in {export_dir}")

    all_files = model_files + metadata_files

    # Generate README from model_card.json if it exists
    model_card_path = export_dir / "model_card.json"
    readme_path = export_dir / "README.md"
    if model_card_path.exists():
        readme_content = _build_readme(model_card_path)
        readme_path.write_text(readme_content, encoding="utf-8")
        all_files.append(readme_path)
        print(f"  Generated README.md from model_card.json")

    # Upload
    print(f"\nUploading {len(all_files)} file(s):")
    for f in all_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.2f} MB)")

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(export_dir),
        path_in_repo=".",
        commit_message=commit_message,
        allow_patterns=[f.name for f in all_files],
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"\nUpload complete: {url}")
    return url


def main(args: argparse.Namespace) -> str:
    print("=" * 60)
    print("Hugging Face Upload")
    print("=" * 60)

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    return upload(
        repo_id=args.repo_id,
        export_dir=export_dir,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload exported models to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HF repo ID, e.g. your-username/food-classifier")
    parser.add_argument("--export-dir", type=str, default="exported_models",
                        help="Directory containing exported model files")
    parser.add_argument("--private", action="store_true",
                        help="Make the repository private")
    parser.add_argument("--commit-message", type=str, default="Upload exported model",
                        help="Commit message for the upload")
    args = parser.parse_args()
    main(args)
