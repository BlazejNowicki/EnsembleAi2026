from pathlib import Path

import cv2
import numpy as np

DATA_PATH = Path('data1')
IMAGES_PATH = DATA_PATH / 'images'


def save_step(step_name: str, filename: str, image: np.ndarray):
    step_path = DATA_PATH / 'steps' / step_name
    step_path.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(step_path / filename), image)
