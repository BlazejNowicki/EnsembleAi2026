from pathlib import Path

import cv2
import numpy as np

SUBMIT = True
# SUBMIT = False

if SUBMIT:
    DATA_PATH = Path('ecg')
    IMAGES_PATH = DATA_PATH / 'test'
else:
    DATA_PATH = Path('data1')
    IMAGES_PATH = DATA_PATH / 'images'

NPZ_FILE = "data/out/ecg_example_submission.npz"

def save_step(step_name: str, filename: str, image: np.ndarray):
    if SUBMIT:
        return
    step_path = DATA_PATH / 'steps' / step_name
    step_path.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(step_path / filename), image)
