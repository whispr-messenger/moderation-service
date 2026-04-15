"""Convert a trained Keras model to TFLite format.

Usage:
    python convert_to_tflite.py                           # uses config.yaml defaults
    python convert_to_tflite.py --model path/to/model.h5  # explicit model path
    python convert_to_tflite.py --quantize                 # enable default optimizations
"""

import argparse
from pathlib import Path

import tensorflow as tf


def convert(model_path: Path, output_path: Path, quantize: bool = False):
    """Load a Keras model, strip data-augmentation, and export to TFLite."""
    base_model = tf.keras.models.load_model(str(model_path))
    base_model.summary()

    # Build an inference model that skips the data_augmentation layer.
    inputs = base_model.input
    x = inputs
    skip_augmentation = True
    for layer in base_model.layers[1:]:
        if skip_augmentation and layer.name == "data_augmentation":
            skip_augmentation = False
            continue
        x = layer(x)

    inference_model = tf.keras.Model(inputs=inputs, outputs=x, name="effnetlite_inference")

    saved_model_dir = output_path.parent / "saved_model_effnetlite_inference"
    inference_model.export(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"TFLite model saved to {output_path} ({len(tflite_model) / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument(
        "--model", type=Path, default=Path("BestModelEfficientNetLite.h5"),
        help="Path to the trained Keras model",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output TFLite path (default: <model_stem>_inference.tflite)",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply default TFLite optimizations (dynamic range quantization)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        raise SystemExit(1)

    output = args.output or args.model.with_name(args.model.stem + "_inference.tflite")
    convert(args.model, output, quantize=args.quantize)


if __name__ == "__main__":
    main()
