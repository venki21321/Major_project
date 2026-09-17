"""Quantize and optimize deepfake models for faster inference and smaller RAM footprint.

Usage:
    .\\venv\\Scripts\\python.exe optimize_models.py --vit --tflite
"""
import argparse
from pathlib import Path


def optimize_vit_model(output_dir=None):
    """Apply dynamic INT8 quantization to the PyTorch ViT model."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    src_dir = Path("models/huggingface_deepfake")
    if not (src_dir / "config.json").exists():
        print(f"[!] ViT model not found at {src_dir}")
        return

    dst_dir = Path(output_dir) if output_dir else Path("models/huggingface_deepfake_quantized")
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading ViT model from {src_dir}...")
    processor = AutoImageProcessor.from_pretrained(src_dir, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(src_dir, local_files_only=True)
    model.eval()

    print("[*] Applying dynamic INT8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    processor.save_pretrained(dst_dir)
    quantized_model.save_pretrained(dst_dir)
    print(f"[✓] Quantized ViT model saved to {dst_dir}")

    # Print file sizes comparison
    orig_size = sum(f.stat().st_size for f in src_dir.glob("*") if f.is_file()) / (1024 * 1024)
    quant_size = sum(f.stat().st_size for f in dst_dir.glob("*") if f.is_file()) / (1024 * 1024)
    print(f"    Original model size:  {orig_size:.2f} MB")
    print(f"    Quantized model size: {quant_size:.2f} MB")
    print(f"    Size reduction:       {(1 - quant_size/orig_size)*100:.1f}%")


def convert_keras_to_tflite(model_path):
    """Convert a .keras model to TFLite format with float16 quantization."""
    import tensorflow as tf

    keras_path = Path(model_path)
    if not keras_path.exists():
        print(f"[!] File not found: {keras_path}")
        return

    tflite_path = keras_path.with_suffix(".tflite")
    print(f"[*] Converting {keras_path.name} to TFLite (Float16)...")

    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    orig_size = keras_path.stat().st_size / (1024 * 1024)
    tflite_size = tflite_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Saved {tflite_path.name}")
    print(f"    Original .keras size:  {orig_size:.2f} MB")
    print(f"    TFLite Float16 size:   {tflite_size:.2f} MB")
    print(f"    Size reduction:        {(1 - tflite_size/orig_size)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Deepfake Model Optimization Utility")
    parser.add_argument("--vit", action="store_true", help="Quantize the Hugging Face ViT model")
    parser.add_argument("--tflite", action="store_true", help="Convert Keras models to TFLite format")
    args = parser.parse_args()

    if not args.vit and not args.tflite:
        args.vit = True
        args.tflite = True

    if args.vit:
        optimize_vit_model()

    if args.tflite:
        models_dir = Path("models")
        keras_files = list(models_dir.glob("*.keras"))
        for keras_file in keras_files:
            try:
                convert_keras_to_tflite(keras_file)
            except Exception as err:
                print(f"[!] Error converting {keras_file.name}: {err}")


if __name__ == "__main__":
    main()
