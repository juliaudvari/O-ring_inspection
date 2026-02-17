from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Initial O-ring project setup")
    parser.add_argument("--images-dir", default="images", help="Folder with jpg files")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")))

    print(f"found_images={len(image_paths)}")
    for image_path in image_paths:
        print(image_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
