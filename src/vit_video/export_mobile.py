from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    import _bootstrap; _bootstrap.setup()
except ImportError:
    pass  # sys.path already configured (e.g. Colab notebook)

from vit_video.paths import PACKAGE_ROOT
from vit_video.utils import print_device_info, parse_normalization_values, load_model_from_checkpoint


def export_torchscript(
    model: torch.nn.Module, output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    optimize_for_mobile: bool = True,
) -> Path:
    print(f"\n[TorchScript] Exporting to {output_path}...")
    model.eval()
    model = model.cpu()

    example_input = torch.randn(*input_shape)
    traced_model = torch.jit.trace(model, example_input, check_trace=False)

    if optimize_for_mobile:
        try:
            from torch.utils.mobile_optimizer import optimize_for_mobile as _optimize
            traced_model = _optimize(traced_model)
            print("  Applied mobile optimizations")
        except Exception as e:
            print(f"  Mobile optimization skipped: {e}")

    traced_model.save(str(output_path))
    print(f"  Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # PyTorch Mobile Lite Interpreter (.ptl) — optimized for on-device Android/iOS
    ptl_path = output_path.with_suffix(".ptl")
    try:
        traced_model._save_for_lite_interpreter(str(ptl_path))
        print(f"  Saved Lite Interpreter to {ptl_path} "
              f"({ptl_path.stat().st_size / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"  Lite Interpreter save skipped: {e}")

    loaded = torch.jit.load(str(output_path))
    test_out = loaded(example_input)
    print(f"  Verification: output shape = {test_out.shape}")
    return output_path


def export_onnx(
    model: torch.nn.Module, output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    opset_version: int = 17,
) -> Path:
    print(f"\n[ONNX] Exporting to {output_path}...")
    model.eval()
    model = model.cpu()

    example_input = torch.randn(*input_shape)
    dynamic_axes = {
        "input": {0: "batch_size", 1: "num_frames"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model, example_input, str(output_path),
        opset_version=opset_version,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes, do_constant_folding=True,
    )
    print(f"  Saved to {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")

    try:
        import onnxruntime as ort
        import numpy as np
        session = ort.InferenceSession(str(output_path))
        outputs = session.run(None, {"input": example_input.numpy().astype(np.float32)})
        print(f"  Verification (ONNX Runtime): output shape = {outputs[0].shape}")
    except ImportError:
        print("  Note: Install onnxruntime to verify ONNX model")
    return output_path


def export_coreml(
    model: torch.nn.Module, output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    class_labels: Optional[List[str]] = None,
) -> Optional[Path]:
    print(f"\n[CoreML] Exporting to {output_path}...")
    try:
        import coremltools as ct
    except ImportError:
        print("  Error: coremltools not installed.")
        return None

    model.eval()
    model = model.cpu()
    traced = torch.jit.trace(model, torch.randn(*input_shape), check_trace=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=input_shape, name="input")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
    )
    if class_labels:
        mlmodel.user_defined_metadata["classes"] = json.dumps(class_labels)

    mlmodel.save(str(output_path))
    print(f"  Saved to {output_path}")
    return output_path


def export_tflite(
    model: torch.nn.Module, output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    quantize: bool = False,
) -> Optional[Path]:
    print(f"\n[TFLite] Exporting to {output_path}...")
    model.eval()
    model = model.cpu()

    try:
        import ai_edge_torch
        edge_model = ai_edge_torch.convert(model, (torch.randn(*input_shape),))
        edge_model.export(str(output_path))
        print(f"  Saved via ai_edge_torch ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return output_path
    except ImportError:
        print("  ai_edge_torch not available, trying ONNX -> TFLite...")
    except Exception as e:
        print(f"  ai_edge_torch failed: {e}, trying ONNX -> TFLite...")

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_path = output_path.with_suffix(".onnx")
        export_onnx(model, onnx_path, input_shape)
        tf_rep = prepare(onnx.load(str(onnx_path)))
        saved_model_dir = output_path.parent / "tf_saved_model"
        tf_rep.export_graph(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize:
            # Float16 weight quantization: ~2x smaller, no accuracy loss, no rep dataset needed.
            # Full int8 requires a representative dataset, which isn't wired here.
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        with open(output_path, "wb") as f:
            f.write(converter.convert())
        print(f"  Saved via ONNX->TFLite ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return output_path
    except ImportError:
        print("  Error: onnx-tf or tensorflow not installed.")
        return None
    except Exception as e:
        print(f"  Error during ONNX->TFLite conversion: {e}")
        return None


def export_tfjs(
    model: torch.nn.Module, output_path: Path,
    input_shape: Tuple[int, ...] = (1, 8, 3, 224, 224),
    quantize: bool = True,
) -> Optional[Path]:
    """PyTorch -> ONNX -> TF SavedModel -> TFJS (browser).

    ⚠️ EXPERIMENTAL for ViT/MobileViT. The `onnx-tf` bridge does not cleanly
    support every transformer op (multi-head attention, dynamic reshapes,
    LayerNorm/GELU combos), and `tensorflowjs_converter` drops or rewrites
    some of the resulting ops. Expect partial failures, op-not-supported
    errors in the browser, or numerically wrong outputs.

    For a browser-deployable food moderation model, use MobileNetV3 (Keras
    native path, see src/mobilenet_v3_small). That's a pure CNN and converts
    cleanly with `tensorflowjs.converters.save_keras_model`.
    """
    print(f"\n[TFJS] Exporting to {output_path} (experimental for transformer models)...")
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflowjs as tfjs
    except ImportError as e:
        print(f"  Skipping TFJS: missing dependency ({e}). "
              f"Install: pip install tensorflowjs onnx onnx-tf tensorflow")
        return None

    model.eval().cpu()
    onnx_path = output_path.parent / (output_path.name + ".onnx")
    try:
        export_onnx(model, onnx_path, input_shape)
    except Exception as e:
        print(f"  TFJS: ONNX stage failed ({e}). ViT attention ops may need a newer opset.")
        return None

    saved_model_dir = output_path.parent / (output_path.name + "_tf_saved_model")
    try:
        prepare(onnx.load(str(onnx_path))).export_graph(str(saved_model_dir))
    except Exception as e:
        print(f"  TFJS: ONNX->TF stage failed ({e}). "
              f"onnx-tf commonly chokes on MultiHeadAttention / dynamic reshape. "
              f"Consider training a CNN if you need browser deployment.")
        return None

    output_path.mkdir(parents=True, exist_ok=True)
    try:
        quant_map = {"uint8": "*"} if quantize else None
        tfjs.converters.convert_tf_saved_model(
            str(saved_model_dir), str(output_path),
            quantization_dtype_map=quant_map,
            skip_op_check=False,
        )
    except Exception as e:
        print(f"  TFJS conversion failed: {e}")
        print(f"  This is expected for transformer architectures. Use MobileNetV3 for browser deployment.")
        return None

    total = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"  Saved to {output_path}/ ({total / 1024 / 1024:.2f} MB, "
          f"{'uint8-quantized' if quantize else 'float32'})")
    print(f"  ⚠️ Validate numerical parity against the PyTorch model before shipping.")
    return output_path


def create_model_card(
    output_dir: Path, model_name: str, num_classes: int, classes: List[str],
    input_shape: Tuple[int, ...], exported_formats: List[str], checkpoint_path: Path,
    normalization_mean: List[float], normalization_std: List[float],
    evaluation_metrics: Optional[dict] = None, training_metadata: Optional[dict] = None,
) -> Path:
    card = {
        "model_name": model_name,
        "task": "video_classification",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "num_classes": num_classes,
        "classes": classes,
        "input_shape": {
            "batch": input_shape[0], "frames": input_shape[1],
            "channels": input_shape[2], "height": input_shape[3], "width": input_shape[4],
        },
        "input_format": "BTCHW",
        "normalization": {"mean": normalization_mean, "std": normalization_std},
        "exported_formats": exported_formats,
        "framework": "pytorch",
        "backbone": model_name,
    }
    if evaluation_metrics:
        card["evaluation"] = {
            k: evaluation_metrics.get(k)
            for k in ("num_samples", "accuracy", "precision_macro", "recall_macro", "f1_macro", "per_class", "classes")
        }
    if training_metadata:
        card["training"] = training_metadata

    card_path = output_dir / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"\nModel card saved to {card_path}")
    return card_path


def main(args) -> None:
    print("=" * 60)
    print("Mobile Model Export")
    print("=" * 60)
    print_device_info()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    formats = args.format
    if "all" in formats:
        # 'tfjs' intentionally excluded from 'all' -- transformer ops don't
        # convert cleanly through onnx-tf. Pass --format tfjs explicitly to try it.
        formats = ["torchscript", "onnx", "coreml", "tflite"]

    input_shape = (1, args.num_frames, 3, args.img_size, args.img_size)
    classes = args.classes.split(",") if args.classes else [f"class_{i}" for i in range(args.num_classes)]
    norm_mean, norm_std = parse_normalization_values(args.norm_mean, args.norm_std)

    print(f"\nModel: {model_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Classes: {classes}")
    print(f"Input shape: {input_shape}")
    print(f"Formats: {formats}")

    model = load_model_from_checkpoint(
        model_path=model_path, num_classes=args.num_classes,
        model_name=args.backbone, device=torch.device("cpu"),
    )
    resolved_backbone = getattr(model, "model_name", args.backbone)

    if "tfjs" in formats and "mobilevit" not in str(resolved_backbone).lower():
        print(f"\n[WARN] TFJS export requested with backbone={resolved_backbone!r}. "
              f"Browser bundle will be large (~330 MB for vit_b_16). "
              f"Train on mobilevit_xxs (~1.3M params) for a web-deployable model.")

    exported = []
    exporters = {
        "torchscript": lambda: export_torchscript(model, output_dir / f"{model_path.stem}.pt", input_shape),
        "onnx": lambda: export_onnx(model, output_dir / f"{model_path.stem}.onnx", input_shape),
        "coreml": lambda: export_coreml(model, output_dir / f"{model_path.stem}.mlpackage", input_shape, classes),
        "tflite": lambda: export_tflite(model, output_dir / f"{model_path.stem}.tflite", input_shape, args.quantize),
        "tfjs": lambda: export_tfjs(model, output_dir / f"{model_path.stem}_tfjs", input_shape, args.quantize),
    }
    for fmt in formats:
        try:
            result = exporters[fmt]()
            if result is not None and Path(result).exists():
                exported.append(fmt)
            elif result is None:
                print(f"  [{fmt}] Export skipped (missing dependency).")
        except Exception as e:
            print(f"  [{fmt}] Export failed: {e}")

    evaluation_metrics = None
    if args.eval_results and Path(args.eval_results).exists():
        try:
            with open(args.eval_results, encoding="utf-8") as f:
                evaluation_metrics = json.load(f)
            if not isinstance(evaluation_metrics, dict):
                print(f"  [WARN] eval_results is not a JSON object — ignoring.")
                evaluation_metrics = None
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Could not read eval results: {e}")

    training_metadata = None
    if args.training_metrics and Path(args.training_metrics).exists():
        try:
            with open(args.training_metrics, encoding="utf-8") as f:
                training_metadata = json.load(f)
            if not isinstance(training_metadata, dict):
                print(f"  [WARN] training_metrics is not a JSON object — ignoring.")
                training_metadata = None
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Could not read training metrics: {e}")

    create_model_card(
        output_dir=output_dir, model_name=resolved_backbone,
        num_classes=args.num_classes, classes=classes,
        input_shape=input_shape, exported_formats=exported,
        checkpoint_path=model_path,
        normalization_mean=norm_mean, normalization_std=norm_std,
        evaluation_metrics=evaluation_metrics, training_metadata=training_metadata,
    )

    print(f"\n{'='*60}")
    print(f"Export Summary: {exported}")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Video Classifier to Mobile Formats")
    parser.add_argument(
        "--model", type=str,
        default=str(PACKAGE_ROOT / "models" / "best_food_classifier.pth"),
    )
    parser.add_argument("--output-dir", type=str, default="exported_models")
    parser.add_argument("--format", type=str, nargs="+", default=["torchscript", "onnx"],
                        choices=["torchscript", "onnx", "coreml", "tflite", "tfjs", "all"])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--classes", type=str, default="healthy,other,unhealthy")
    parser.add_argument("--backbone", type=str, default="auto")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--eval-results", type=str, default="")
    parser.add_argument("--training-metrics", type=str, default="")
    parser.add_argument("--norm-mean", type=str, default="0.485,0.456,0.406")
    parser.add_argument("--norm-std", type=str, default="0.229,0.224,0.225")
    args = parser.parse_args()
    main(args)
