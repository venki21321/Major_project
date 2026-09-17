"""Stream balanced subsets of official deepfake/AI-image datasets.

Sources:
  - saberzl/SID_Set: 0 real, 1 fully synthetic, 2 tampered
  - OwensLab/CommunityForensics-Small: 0 real, 1 generated

DF40 requires accepting its official terms and must be downloaded separately.
"""
import argparse
import base64
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def save_image(value, destination):
    if isinstance(value, Image.Image):
        image = value
    else:
        if isinstance(value, dict):
            value = value.get("bytes") or value.get("path")
        if isinstance(value, str):
            path = Path(value)
            value = path.read_bytes() if path.exists() else base64.b64decode(value)
        image = Image.open(io.BytesIO(value))
    image.convert("RGB").save(destination, "JPEG", quality=92)


def stream_sid(output, real_limit, synthetic_limit, tampered_limit):
    limits = {0: real_limit, 1: synthetic_limit, 2: tampered_limit}
    counts = {0: 0, 1: 0, 2: 0}
    data = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    data = data.shuffle(seed=42, buffer_size=10_000)
    for row in data:
        label = int(row["label"])
        if label not in limits or counts[label] >= limits[label]:
            continue
        folder = "real" if label == 0 else "fake"
        name = output / folder / f"sid_{label}_{counts[label]:06d}.jpg"
        save_image(row["image"], name)
        counts[label] += 1
        if all(counts[key] >= limits[key] for key in limits):
            break
    print("SID-Set:", counts)


def stream_community(output, real_limit, fake_limit):
    limits = {0: real_limit, 1: fake_limit}
    counts = {0: 0, 1: 0}
    data = load_dataset(
        "OwensLab/CommunityForensics-Small",
        split="train", streaming=True)
    data = data.shuffle(seed=43, buffer_size=10_000)
    for row in data:
        label = int(row["label"])
        if label not in limits or counts[label] >= limits[label]:
            continue
        folder = "real" if label == 0 else "fake"
        name = output / folder / f"community_{label}_{counts[label]:06d}.jpg"
        save_image(row["image_data"], name)
        counts[label] += 1
        if all(counts[key] >= limits[key] for key in limits):
            break
    print("Community Forensics:", counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="combined_dataset")
    parser.add_argument("--sid-real", type=int, default=10_000)
    parser.add_argument("--sid-synthetic", type=int, default=5_000)
    parser.add_argument("--sid-tampered", type=int, default=5_000)
    parser.add_argument("--community-real", type=int, default=5_000)
    parser.add_argument("--community-fake", type=int, default=5_000)
    args = parser.parse_args()

    output = Path(args.output)
    (output / "real").mkdir(parents=True, exist_ok=True)
    (output / "fake").mkdir(parents=True, exist_ok=True)
    stream_sid(
        output, args.sid_real, args.sid_synthetic, args.sid_tampered)
    stream_community(
        output, args.community_real, args.community_fake)
    print("Finished. Dataset:", output.resolve())


if __name__ == "__main__":
    main()
