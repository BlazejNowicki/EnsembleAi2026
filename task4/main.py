import os
import shutil

from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path

from src.crop_line import crop_line_simple
from src.detect_rectangles import detect_rectangles
from src.extract_ink import extract_ink
from src.load_annotations import load_annotations, Annotation

DATA_PATH = Path('data1')
IMAGES_PATH = DATA_PATH / 'images'


def save_step(step_name: str, filename: str, image: np.ndarray):
    step_path = DATA_PATH / 'steps' / step_name
    step_path.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(step_path / filename), image)


def convert_annotation_to_abs(annotation: Annotation, image_shape: tuple[int, int]) -> Annotation:
    height, width = image_shape[:2]
    return tuple((float(x * width), float(y * height)) for x, y in annotation)


def digitize(filename: str, annotation: Annotation):
    path = IMAGES_PATH / filename

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image at {path}. Check file format.")

    image = extract_ink(image)
    save_step('1_ink', filename, image * 255)

    annotation = convert_annotation_to_abs(annotation, image.shape[:2])

    line1 = crop_line_simple(image, annotation[0][0], annotation[0][1], annotation[3][0])
    line2 = crop_line_simple(image, annotation[1][0], annotation[1][1], annotation[4][0])
    line3 = crop_line_simple(image, annotation[2][0], annotation[2][1], annotation[5][0])

    save_step('2_cropped_1', filename, line1 * 255)
    save_step('2_cropped_2', filename, line2 * 255)
    save_step('2_cropped_3', filename, line3 * 255)


def main():
    if (DATA_PATH / 'steps').exists():
        shutil.rmtree(DATA_PATH / 'steps')

    png_files = [p.name for p in IMAGES_PATH.glob('*.png')]
    png_files = [png_files[0]]

    annotations = load_annotations()

    annotations = {k: v for k, v in annotations.items() if v is not None}
    annotation = annotations['ecg_test_0001.png']

    print(annotation)

    annotations = ((0.053963636363636366, 0.4165294117647059), (0.05402121212121212, 0.5833450980392156),
                   (0.05412121212121212, 0.7512666666666666), (0.9484060606060606, 0.4041686274509804),
                   (0.9482363636363635, 0.5759254901960784), (0.9481999999999999, 0.7446196078431373))

    # (0.054, 0.948)

    for file in tqdm(png_files):
        digitize(file, annotation)


if __name__ == "__main__":
    main()
