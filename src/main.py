from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# add root import path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import process_image  # noqa: E402


def _collect_images(images_dir: Path) -> list[Path]:
    # read jpg list
    files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
    unique = sorted(set(files))
    return unique


def build_parser() -> argparse.ArgumentParser:
    # cli args
    parser = argparse.ArgumentParser(description="NumPy-only O-ring inspection pipeline.")
    parser.add_argument("--images-dir", default="images", help="Directory containing JPG images.")
    parser.add_argument("--output-dir", default="output", help="Directory for output images.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug masks.")
    parser.add_argument("--ksize", type=int, default=5, help="Morphology kernel size (odd).")
    parser.add_argument(
        "--morph-iterations",
        type=int,
        default=2,
        help="Number of iterations for dilation/erosion in closing.",
    )
    parser.add_argument(
        "--angles",
        type=int,
        default=360,
        help="Number of radial samples for thickness analysis.",
    )
    return parser


def main() -> int:
    # parse cli
    parser = build_parser()
    args = parser.parse_args()

    # set input path
    images_dir = Path(args.images_dir)
    if not images_dir.is_absolute():
        images_dir = PROJECT_ROOT / images_dir
    images_dir = images_dir.resolve()

    # set output path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    # get image files
    image_paths = _collect_images(images_dir)
    if not image_paths:
        print(f"No .jpg files found in {images_dir}")
        return 1

    # print csv header
    print("filename,threshold,PASS/FAIL,time_ms,continuity,thickness_median,thickness_mad,max_gap_deg,reason")
    for image_path in image_paths:
        try:
            # process one image
            result = process_image(
                image_path=image_path,
                output_dir=output_dir,
                debug=args.debug,
                ksize=args.ksize,
                morph_iterations=args.morph_iterations,
                n_angles=args.angles,
            )
            safe_reason = result.reason.replace(",", ";")
            # print csv row
            print(
                f"{result.filename},{result.threshold},{result.status},"
                f"{result.time_ms:.2f},{result.continuity:.4f},"
                f"{result.thickness_median:.2f},{result.thickness_mad:.2f},"
                f"{result.max_angular_gap:.1f},{safe_reason}"
            )
        except Exception as exc:
            print(f"{image_path.name},-1,FAIL,0.00,0.0000,0.00,0.00,0.0,error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
