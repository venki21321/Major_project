"""Download the public models referenced by Ai_Fake_Content_Detector.ipynb."""
import argparse
from pathlib import Path

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


IMAGE_MODEL = "dima806/deepfake_vs_real_image_detection"
NEWS_MODEL = "jy46604790/Fake-News-Bert-Detect"


def download_image():
    destination = Path("models/huggingface_deepfake")
    destination.mkdir(parents=True, exist_ok=True)
    AutoImageProcessor.from_pretrained(IMAGE_MODEL).save_pretrained(destination)
    AutoModelForImageClassification.from_pretrained(IMAGE_MODEL).save_pretrained(
        destination, safe_serialization=True)
    print(f"Image model saved to {destination}")


def download_news():
    destination = Path("models/huggingface_fake_news")
    destination.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(NEWS_MODEL).save_pretrained(destination)
    AutoModelForSequenceClassification.from_pretrained(NEWS_MODEL).save_pretrained(
        destination, safe_serialization=True)
    print(f"News model saved to {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--news", action="store_true")
    arguments = parser.parse_args()
    if not arguments.image and not arguments.news:
        parser.error("Choose --image, --news, or both.")
    if arguments.image:
        download_image()
    if arguments.news:
        download_news()


if __name__ == "__main__":
    main()
