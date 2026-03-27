from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Ensure vit_video is importable from repo root, src/, or src/vit_video/
_script_dir = Path(__file__).resolve().parent
_src = _script_dir.parent
if _src.name == "src" and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
elif _script_dir.name == "vit_video" and str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from vit_video.utils import print_device_info, parse_normalization_values, get_device, load_model_from_checkpoint


def export_torchscript(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    optimize_for_mobile: bool = True,
) -> Path:
    print(f"\n[TorchScript] Exporting to {output_path}...")
    
    model.eval()
    model = model.cpu()
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    traced_model = torch.jit.trace(model, example_input, check_trace=False)
    
    # Optionally optimize for mobile
    if optimize_for_mobile:
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile as _optimize
            traced_model = _optimize(traced_model)
            print("  Applied mobile optimizations")
        except ImportError:
            print("  Warning: torch.utils.mobile_optimizer not available, skipping mobile optimization")
    
    # Save
    traced_model.save(str(output_path))
    print(f"  Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Verify
    loaded = torch.jit.load(str(output_path))
    test_out = loaded(example_input)
    print(f"  Verification: output shape = {test_out.shape}")
    
    return output_path


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    opset_version: int = 17,
    dynamic_axes: Optional[dict] = None,
) -> Path:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output file path (.onnx)
        input_shape: (batch, frames, channels, height, width)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable-size inputs
    
    Returns:
        Path to exported model
    """
    print(f"\n[ONNX] Exporting to {output_path}...")
    
    model.eval()
    model = model.cpu()
    
    example_input = torch.randn(*input_shape)
    
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    torch.onnx.export(
        model,
        example_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    
    print(f"  Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Verify with ONNX Runtime if available
    try:
        import onnxruntime as ort
        import numpy as np
        
        session = ort.InferenceSession(str(output_path))
        test_input = example_input.numpy().astype(np.float32)
        outputs = session.run(None, {"input": test_input})
        print(f"  Verification (ONNX Runtime): output shape = {outputs[0].shape}")
    except ImportError:
        print("  Note: Install onnxruntime to verify ONNX model")
    
    return output_path


def export_coreml(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    class_labels: Optional[List[str]] = None,
) -> Optional[Path]:
    """
    Export model to CoreML format for iOS/macOS.
    
    Requires: coremltools
    
    Args:
        model: PyTorch model
        output_path: Output path (.mlpackage or .mlmodel)
        input_shape: (batch, frames, channels, height, width)
        class_labels: List of class names for classification
    
    Returns:
        Path to exported model or None if coremltools unavailable
    """
    print(f"\n[CoreML] Exporting to {output_path}...")
    
    try:
        import coremltools as ct
    except ImportError:
        print("  Error: coremltools not installed. Install with: pip install coremltools")
        return None
    
    model.eval()
    model = model.cpu()
    
    example_input = torch.randn(*input_shape)
    traced_model = torch.jit.trace(model, example_input, check_trace=False)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=input_shape, name="input")],
        convert_to="mlprogram",  # Use newer ML Program format
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Add class labels if provided
    if class_labels:
        mlmodel.user_defined_metadata["classes"] = json.dumps(class_labels)
    
    # Save
    mlmodel.save(str(output_path))
    print(f"  Saved to {output_path}")
    
    return output_path


def export_tflite(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    quantize: bool = False,
) -> Optional[Path]:
    """
    Export model to TensorFlow Lite format for Android.
    
    This uses one of several methods:
    1. ai_edge_torch (recommended for PyTorch -> TFLite)
    2. ONNX -> TFLite via onnx2tf
    3. Manual TensorFlow conversion
    
    Args:
        model: PyTorch model
        output_path: Output file path (.tflite)
        input_shape: (batch, frames, channels, height, width)
        quantize: Apply int8 quantization (smaller model, faster inference)
    
    Returns:
        Path to exported model or None if conversion failed
    """
    print(f"\n[TFLite] Exporting to {output_path}...")
    
    model.eval()
    model = model.cpu()
    
    # Method 1: Try ai_edge_torch (Google's official PyTorch -> TFLite)
    try:
        import ai_edge_torch
        
        example_input = torch.randn(*input_shape)
        edge_model = ai_edge_torch.convert(model, (example_input,))
        edge_model.export(str(output_path))
        print(f"  Saved via ai_edge_torch to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return output_path
    except ImportError:
        print("  ai_edge_torch not available, trying ONNX -> TFLite...")
    except Exception as e:
        print(f"  ai_edge_torch failed: {e}, trying ONNX -> TFLite...")
    
    # Method 2: ONNX intermediate -> TFLite
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
        
        # First export to ONNX
        onnx_path = output_path.with_suffix(".onnx")
        export_onnx(model, onnx_path, input_shape)
        
        # Load ONNX and convert to TF
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        
        # Export to SavedModel format
        saved_model_dir = output_path.parent / "tf_saved_model"
        tf_rep.export_graph(str(saved_model_dir))
        
        # Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        
        tflite_model = converter.convert()
        
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        
        print(f"  Saved via ONNX->TF->TFLite to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return output_path
        
    except ImportError:
        print("  Error: onnx-tf or tensorflow not installed.")
        print("  Install with: pip install onnx-tf tensorflow")
        print("  Or use: pip install ai-edge-torch (recommended)")
        return None
    except Exception as e:
        print(f"  Error during ONNX->TFLite conversion: {e}")
        return None


def create_model_card(
    output_dir: Path,
    model_name: str,
    num_classes: int,
    classes: List[str],
    input_shape: Tuple[int, ...],
    exported_formats: List[str],
    checkpoint_path: Path,
    normalization_mean: List[float],
    normalization_std: List[float],
    evaluation_metrics: Optional[dict] = None,
    training_metadata: Optional[dict] = None,
) -> Path:
    """Create a model card JSON with metadata for app integration."""
    card = {
        "model_name": model_name,
        "task": "video_classification",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "num_classes": num_classes,
        "classes": classes,
        "input_shape": {
            "batch": input_shape[0],
            "frames": input_shape[1],
            "channels": input_shape[2],
            "height": input_shape[3],
            "width": input_shape[4],
        },
        "input_format": "BTCHW",  # Batch, Time, Channels, Height, Width
        "normalization": {
            "mean": normalization_mean,
            "std": normalization_std,
        },
        "exported_formats": exported_formats,
        "framework": "pytorch",
        "backbone": model_name,
    }

    if evaluation_metrics:
        card["evaluation"] = {
            "num_samples": evaluation_metrics.get("num_samples"),
            "accuracy": evaluation_metrics.get("accuracy"),
            "precision_macro": evaluation_metrics.get("precision_macro"),
            "recall_macro": evaluation_metrics.get("recall_macro"),
            "f1_macro": evaluation_metrics.get("f1_macro"),
            "per_class": evaluation_metrics.get("per_class"),
            "classes": evaluation_metrics.get("classes", classes),
            "source": "test_results.json",
        }

    if training_metadata:
        card["training"] = training_metadata
    
    card_path = output_dir / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    
    print(f"\nModel card saved to {card_path}")
    return card_path


def main(args: argparse.Namespace) -> None:
    """Main export entry point."""
    print("=" * 60)
    print("Mobile Model Export")
    print("=" * 60)
    device = get_device()
    print_device_info()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Parse formats
    formats = args.format
    if "all" in formats:
        formats = ["torchscript", "onnx", "coreml", "tflite"]
    
    # Input shape: (B, T, C, H, W)
    input_shape = (1, args.num_frames, 3, args.img_size, args.img_size)
    
    # Classes
    classes = args.classes.split(",") if args.classes else [f"class_{i}" for i in range(args.num_classes)]

    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)
    
    print(f"\nModel: {model_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Num classes: {args.num_classes}")
    print(f"Classes: {classes}")
    print(f"Input shape: {input_shape}")
    print(f"Formats: {formats}")
    print(f"Output dir: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(
        model_path=model_path,
        num_classes=args.num_classes,
        model_name=args.backbone,
        device=torch.device("cpu"),
    )
    
    exported = []
    
    # Export to each format
    if "torchscript" in formats:
        ts_path = output_dir / f"{model_path.stem}.pt"
        try:
            export_torchscript(model, ts_path, input_shape, optimize_for_mobile=True)
            if ts_path.exists():
                exported.append("torchscript")
        except Exception as e:
            print(f"  [TorchScript] Export failed: {e}")
    
    if "onnx" in formats:
        onnx_path = output_dir / f"{model_path.stem}.onnx"
        try:
            export_onnx(model, onnx_path, input_shape)
            if onnx_path.exists():
                exported.append("onnx")
        except Exception as e:
            print(f"  [ONNX] Export failed: {e}")
    
    if "coreml" in formats:
        coreml_path = output_dir / f"{model_path.stem}.mlpackage"
        try:
            result = export_coreml(model, coreml_path, input_shape, class_labels=classes)
        except Exception as e:
            print(f"  [CoreML] Export failed: {e}")
            result = None
        if result and coreml_path.exists():
            exported.append("coreml")
    
    if "tflite" in formats:
        tflite_path = output_dir / f"{model_path.stem}.tflite"
        try:
            result = export_tflite(model, tflite_path, input_shape, quantize=args.quantize)
        except Exception as e:
            print(f"  [TFLite] Export failed: {e}")
            result = None
        if result and tflite_path.exists():
            exported.append("tflite")

    evaluation_metrics = None
    if args.eval_results:
        eval_path = Path(args.eval_results)
        if eval_path.exists():
            with eval_path.open("r", encoding="utf-8") as f:
                evaluation_metrics = json.load(f)
        else:
            print(f"Warning: eval results file not found: {eval_path}")

    training_metadata = None
    if args.training_metrics:
        train_metrics_path = Path(args.training_metrics)
        if train_metrics_path.exists():
            with train_metrics_path.open("r", encoding="utf-8") as f:
                training_metadata = json.load(f)
        else:
            print(f"Warning: training metrics file not found: {train_metrics_path}")
    
    # Create model card
    create_model_card(
        output_dir=output_dir,
        model_name=args.backbone,
        num_classes=args.num_classes,
        classes=classes,
        input_shape=input_shape,
        exported_formats=exported,
        checkpoint_path=model_path,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        evaluation_metrics=evaluation_metrics,
        training_metadata=training_metadata,
    )
    
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"Successfully exported: {exported}")
    print(f"Output directory: {output_dir}")
    print("\nUsage hints:")
    if "torchscript" in exported:
        print("  - TorchScript: Use with PyTorch Mobile (Android/iOS)")
        print("    mobile_model = torch.jit.load('model.pt')")
    if "onnx" in exported:
        print("  - ONNX: Use with ONNX Runtime Mobile")
        print("    session = ort.InferenceSession('model.onnx')")
    if "coreml" in exported:
        print("  - CoreML: Use with Core ML on iOS/macOS")
        print("    let model = try MLModel(contentsOf: modelURL)")
    if "tflite" in exported:
        print("  - TFLite: Use with TensorFlow Lite on Android")
        print("    interpreter = Interpreter.create(context, modelFile)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Video Classifier to Mobile Formats")
    parser.add_argument(
        "--model", type=str, default="best_food_classifier.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="exported_models",
        help="Directory to save exported models"
    )
    parser.add_argument(
        "--format", type=str, nargs="+", default=["torchscript", "onnx"],
        choices=["torchscript", "onnx", "coreml", "tflite", "all"],
        help="Export format(s)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=2,
        help="Number of output classes"
    )
    parser.add_argument(
        "--classes", type=str, default="healthy,unhealthy",
        help="Comma-separated class names"
    )
    parser.add_argument(
        "--backbone", type=str, default="auto",
        help="Backbone model name (must match training). Use \"auto\" for auto-detection from checkpoint."
    )
    parser.add_argument(
        "--num-frames", type=int, default=8,
        help="Number of frames per video"
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Image size (height and width)"
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply int8 quantization for TFLite"
    )
    parser.add_argument(
        "--eval-results", type=str, default="",
        help="Optional path to test_results.json to include metrics in model_card.json"
    )
    parser.add_argument(
        "--training-metrics", type=str, default="",
        help="Optional path to *_training_metrics.json to include training metadata in model_card.json"
    )
    parser.add_argument(
        "--norm-mean", type=str, default="0.485,0.456,0.406",
        help="Comma-separated normalization mean used in preprocessing"
    )
    parser.add_argument(
        "--norm-std", type=str, default="0.229,0.224,0.225",
        help="Comma-separated normalization std used in preprocessing"
    )
    
    args = parser.parse_args()
    main(args)
