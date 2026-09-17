"""Stream a balanced SID-Set subset without downloading the full 140 GB.

Labels in SID-Set:
    0 = real, 1 = fully synthetic, 2 = tampered
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sid_dataset")
    parser.add_argument("--real", type=int, default=30_000)
    parser.add_argument("--synthetic", type=int, default=15_000)
    parser.add_argument("--tampered", type=int, default=15_000)
    args = parser.parse_args()

    output = Path(args.output)
    real_dir = output / "real"
    fake_dir = output / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    limits = {0: args.real, 1: args.synthetic, 2: args.tampered}
    counts = {0: 0, 1: 0, 2: 0}
    data = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    data = data.shuffle(seed=42, buffer_size=10_000)

    for row in data:
        label = int(row["label"])
        if label not in limits or counts[label] >= limits[label]:
            continue
        image = row["image"].convert("RGB")
        destination = real_dir if label == 0 else fake_dir
        name = f"sid_{label}_{counts[label]:06d}.jpg"
        image.save(destination / name, "JPEG", quality=92)
        counts[label] += 1

        total = sum(counts.values())
        if total % 500 == 0:
            print(
                f"Downloaded {total}: real={counts[0]}, "
                f"synthetic={counts[1]}, tampered={counts[2]}"
            )
        if all(counts[label] >= limits[label] for label in limits):
            break

    print("Finished:", counts)
    if any(counts[label] < limits[label] for label in limits):
        raise RuntimeError("The stream ended before all requested images were saved.")


if __name__ == "__main__":
    main()
