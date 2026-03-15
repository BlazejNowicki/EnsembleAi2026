import cv2
import numpy as np


def extract_ink(image: np.ndarray, sensitivity: int) -> np.ndarray:
    """
        Extracts the dark ink from an ECG image using HSV color space.

        Args:
            image (np.ndarray): The loaded BGR image.

        Returns:
            np.ndarray: A binary mask where ink is 1 and the background/grid is 0.
        """
    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Define range for black/dark ink
    # Hue (0-180) and Saturation (0-255) cover all colors.
    # Value (0-100) specifically isolates dark pixels.
    # Note: You can tweak the '100' up or down if some ink is being missed
    # or if dark grid lines are bleeding through.
    lower_black = np.array([0, 0, 0])

    # hue, saturation, value
    # sensitivity = 100
    upper_black = np.array([180, 255, sensitivity])

    # 3. Create a binary mask (255 for ink, 0 for background)
    mask_255 = cv2.inRange(hsv, lower_black, upper_black)

    # 4. Apply a slight morphological closing to bridge tiny gaps in the ink lines
    # This is highly recommended for ECGs to keep the pulse lines continuous
    kernel = np.ones((3, 3), np.uint8)
    mask_255 = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)

    # 5. Convert the 255 mask to a 0 and 1 mask as requested
    mask_1 = (mask_255 / 255).astype(np.uint8)

    return mask_1


def adaptive_extract_ink(image: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]
    x_start, x_end = int(0.45 * W), int(0.55 * W)
    y_start, y_end = int(0.10 * H), int(0.20 * H)

    # 1. Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Extract the Value (brightness) channel for the heuristic rectangle
    V_roi = hsv[y_start:y_end, x_start:x_end, 2]

    # 3. The highest sensitivity that keeps the region clear is min(V) - 1
    # We cast to int so that if min(V) is 0, we safely get -1 instead of uint8 underflow (255)
    best_sensitivity = int(np.min(V_roi)) - 1

    # 4. Generate and return the full image mask
    best_sensitivity = max(60, best_sensitivity)
    return extract_ink(image, best_sensitivity)
