import os
import shutil
from typing import Any

import numpy as np
from tqdm import tqdm
import cv2

from src.crop_line import crop_line_simple
from src.extract_ink import extract_ink, adaptive_extract_ink
from src.load_annotations import load_annotations, Annotation
from src.read_signal import read_signal
from src.utils import IMAGES_PATH, save_step, DATA_PATH, NPZ_FILE, SUBMIT

DEFAULT_ANNOTATION = ((0.053963636363636366, 0.4165294117647059), (0.05402121212121212, 0.5833450980392156),
                      (0.05412121212121212, 0.7512666666666666), (0.9484060606060606, 0.4041686274509804),
                      (0.9482363636363635, 0.5759254901960784), (0.9481999999999999, 0.7446196078431373))


def convert_annotation_to_abs(annotation: Any, image_shape: tuple[int, int]) -> Annotation:
    height, width = image_shape[:2]
    return tuple((float(x * width), float(y * height)) for x, y in annotation)


def digitize(filename: str, default: bool, annotation: Annotation | None):
    path = IMAGES_PATH / filename

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image at {path}. Check file format.")

    # image = extract_ink(image, sensitivity=100)
    image = adaptive_extract_ink(image)

    save_step('1_ink', filename, image * 255)

    if default:
        annotation = DEFAULT_ANNOTATION
        annotation = convert_annotation_to_abs(annotation, image.shape[:2])
        line1 = crop_line_simple(image, annotation[0][0], annotation[0][1], annotation[3][0])
        line2 = crop_line_simple(image, annotation[1][0], annotation[1][1], annotation[4][0])
        line3 = crop_line_simple(image, annotation[2][0], annotation[2][1], annotation[5][0])
    else:
        # todo: change to real rotation
        annotation = DEFAULT_ANNOTATION
        annotation = convert_annotation_to_abs(annotation, image.shape[:2])
        line1 = crop_line_simple(image, annotation[0][0], annotation[0][1], annotation[3][0])
        line2 = crop_line_simple(image, annotation[1][0], annotation[1][1], annotation[4][0])
        line3 = crop_line_simple(image, annotation[2][0], annotation[2][1], annotation[5][0])

    save_step('2_cropped_1', filename, line1 * 255)
    save_step('2_cropped_2', filename, line2 * 255)
    save_step('2_cropped_3', filename, line3 * 255)

    sub_dict = read_signal(line1, line2, line3, filename)
    return sub_dict


def main():
    if (DATA_PATH / 'steps').exists():
        shutil.rmtree(DATA_PATH / 'steps')

    png_files = [p.name for p in IMAGES_PATH.glob('*.png')]

    # if not SUBMIT:
    #   png_files = [list(sorted(png_files))[0]]

    annotations = load_annotations()

    annotations = {k: v for k, v in annotations.items() if v is not None}

    # print(annotations)
    # annotation = annotations['ecg_test_0003.png']

    # annotations = ((0.053963636363636366, 0.4165294117647059), (0.05402121212121212, 0.5833450980392156),
    #                (0.05412121212121212, 0.7512666666666666), (0.9484060606060606, 0.4041686274509804),
    #                (0.9482363636363635, 0.5759254901960784), (0.9481999999999999, 0.7446196078431373))

    # (0.054, 0.948)

    final_dict = {}
    for file in tqdm(png_files):
        sub_dict = digitize(file, default=True, annotation=None)
        # todo: use this code
        # if file not in annotations:
        #     sub_dict = digitize(file, default=True, annotation=None)
        # else:
        #     sub_dict = digitize(file, default=False, annotation=annotations[file])

        final_dict |= sub_dict

    if SUBMIT:
        os.makedirs(os.path.dirname(NPZ_FILE), exist_ok=True)
        np.savez_compressed(NPZ_FILE, **final_dict, )

    print(len(final_dict))
    # print(final_dict.keys())
    # print(type(final_dict["ecg_test_0001_I"]))
    # print(final_dict["ecg_test_0001_I"].shape)
    # print(final_dict["ecg_test_0001_I"].dtype)


if __name__ == "__main__":
    main()
