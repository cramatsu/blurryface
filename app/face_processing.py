from typing import List

from PIL import Image, ImageFilter


def blur_faces(face_boxes: List[tuple[int, ...]], image: Image, radius: int = 20) -> Image:
    for face_box in face_boxes:
        region = image.crop(face_box).filter(ImageFilter.GaussianBlur(radius))
        image.paste(region, face_box)
    return image
