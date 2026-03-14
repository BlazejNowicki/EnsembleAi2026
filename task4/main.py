import os
import shutil

from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
from typing import List

from src.detect_rectangles import detect_rectangles
from src.extract_ink import extract_ink
from src.load_annotations import load_annotations

DATA_PATH = Path('data1')
IMAGES_PATH = DATA_PATH / 'images'


def save_step(step_name: str, filename: str, image: np.ndarray):
    step_path = DATA_PATH / 'steps' / step_name
    step_path.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(step_path / filename), image)


def digitize(filename: str):
    path = IMAGES_PATH / filename

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image at {path}. Check file format.")

    image = extract_ink(image)
    save_step('1_ink', filename, image * 255)




def main():
    if (DATA_PATH / 'steps').exists():
        shutil.rmtree(DATA_PATH / 'steps')

    png_files = [p.name for p in IMAGES_PATH.glob('*.png')]

    annotations = load_annotations()

    annotations = {k: v for k, v in annotations.items() if v is not None}
    annotation = annotations['ecg_test_0001.png']

    print(annotation)

    for file in tqdm(png_files):
        digitize(file)


if __name__ == "__main__":
    main()
