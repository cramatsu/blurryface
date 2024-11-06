import argparse
import sys
from pathlib import Path

import onnxruntime as ort
from colorama import Fore, Style
from PIL import Image

from app.face_processing import blur_faces
from app.yolo import YOLOModel

ort.set_default_logger_severity(3)


def main(image_path: Path, conf: float, iou: float, rad: int):
    print(
        f"{Fore.BLUE}conf={conf}{Style.RESET_ALL}, "
        f"{Fore.GREEN}iou={iou}{Style.RESET_ALL}, "
        f"{Fore.RED}rad={rad}{Style.RESET_ALL}"
    )
    print(f"{Fore.YELLOW}Loading file, please wait...{Style.RESET_ALL}")

    if not image_path.is_file():
        print(f"{Fore.RED}Error: The image file '{image_path}' does not exist.")
        sys.exit(1)

    image = Image.open(image_path)

    print(f"{Fore.YELLOW}Loading YOLO model, please wait...{Style.RESET_ALL}")
    yolo_model = YOLOModel()

    faces = yolo_model.detect_faces(image_path, conf, iou)

    if faces is None:
        print(f"{Fore.RED}Error: No faces were detected in the image.{Style.RESET_ALL}")
        return

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / image_path.name

    print(f"{Fore.YELLOW}Processing the image, please wait...{Style.RESET_ALL}")
    processed_image = blur_faces(faces, image, rad)

    print(f"{Fore.GREEN}Image saved to '{output_path}'{Style.RESET_ALL}")
    processed_image.save(output_path)


def start():
    parser = argparse.ArgumentParser(description="Process an image with YOLO model and optional thresholds.")
    parser.add_argument("image_path", type=str, help="Path to the image to process.")
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Confidence threshold for face detection (default: 0.25)."
    )
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for face detection (default: 0.45).")
    parser.add_argument(
        "--rad", type=int, default=20, help="Radius for Gaussian blur applied to detected faces (default: 20)."
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)

    main(image_path, args.conf, args.iou, args.rad)


if __name__ == "__main__":
    start()
