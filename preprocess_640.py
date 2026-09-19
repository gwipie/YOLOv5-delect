import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RGB/NCHW float32 calibration tensors for hb_mapper.")
    parser.add_argument("--src", type=Path, required=True, help="Directory containing representative images.")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory for raw float32 tensors.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of images to convert.")
    parser.add_argument("--size", type=int, default=640, help="Square model input size.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit <= 0 or args.size <= 0:
        raise ValueError("--limit and --size must be positive")

    image_paths = sorted(
        path
        for path in args.src.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    converted = 0
    for src_path in image_paths:
        image = cv2.imread(str(src_path))
        if image is None:
            print(f"skip unreadable image: {src_path}")
            continue

        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(image, (2, 0, 1)).astype(np.float32)

        output_path = args.dst / f"{src_path.stem}.rgb"
        tensor.tofile(output_path)
        converted += 1
        print(f"write: {output_path}")

    if converted == 0:
        raise RuntimeError("All calibration images failed to decode")
    print(f"Done. Converted {converted} images.")


if __name__ == "__main__":
    main()
