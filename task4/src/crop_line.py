import cv2
import numpy as np
import math


def crop_rectangle(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> np.ndarray:
    """
    Crops a straight rectangle from an image using standard NumPy slicing.

    Args:
        image: The original image (NumPy array).
        x1, y1: Top-left corner coordinates.
        x2, y2: Bottom-right corner coordinates.

    Returns:
        The cropped image array.
    """
    # 1. Convert coordinates to integers (required for array slicing)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 2. Get the maximum image dimensions
    max_h, max_w = image.shape[:2]

    # 3. Clip coordinates to image boundaries to prevent out-of-bounds errors
    x1 = max(0, min(x1, max_w))
    x2 = max(0, min(x2, max_w))
    y1 = max(0, min(y1, max_h))
    y2 = max(0, min(y2, max_h))

    # 4. Validate that a valid rectangle can be formed
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid coordinates: Top-left must be above and to the left of Bottom-right.")

    # 5. Crop using NumPy slicing [start_y:end_y, start_x:end_x]

    cropped = image[y1:y2, x1:x2]
    return cropped


def crop_line_simple(image, x1, y, x2):
    # x1 = int(x1 * image.shape[1])
    # x2 = int(x2 * image.shape[1])
    # y = int(y * image.shape[0])
    width_ratio = 0.05
    w = int((x2 - x1) * width_ratio)
    # print(x1, x2, y, w)

    # return image
    return crop_rectangle(image, x1, y - w, x2, y + w)
