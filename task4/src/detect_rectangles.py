from typing import List

import cv2
import numpy as np


def detect_rectangles(mask_1: np.ndarray) -> List[np.ndarray]:
    """
    Detects open-bottom rectangular calibration pulses from a binary ink mask.

    Args:
        mask_1 (np.ndarray): Binary mask where ink is 1 and background is 0.

    Returns:
        List[np.ndarray]: A list of 4-point coordinates for each detected pulse.
    """
    # 1. Convert 0/1 mask back to 0/255 for OpenCV contour detection
    mask_255 = (mask_1 * 255).astype(np.uint8)

    # 2. Find contours (the outlines of the ink shapes)
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_pulses = []

    for cnt in contours:
        # Ignore tiny specks of noise
        if cv2.contourArea(cnt) < 50:
            continue

        # 3. Filter by Aspect Ratio (handles rotation gracefully)
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (width, height), angle = rect

        if width == 0 or height == 0:
            continue

        # Standardize long and short sides
        long_side = max(width, height)
        short_side = min(width, height)
        aspect_ratio = long_side / short_side

        # A standard mark is 10mm tall and 5mm wide (ratio ~ 2.0)
        # We use a wide margin (1.4 to 3.0) to account for scanning distortions
        if 1.4 < aspect_ratio < 3.0:

            # 4. Verify the "Open Bottom" topology
            # Get an upright bounding box to easily slice the region
            x, y, w, h = cv2.boundingRect(cnt)

            # Define the region of interest (ROI): bottom 20% of height, middle 60% of width
            # We use max/min to prevent slicing outside the image boundaries
            roi_y1 = min(y + int(h * 0.8), mask_1.shape[0])
            roi_y2 = min(y + h, mask_1.shape[0])
            roi_x1 = min(x + int(w * 0.2), mask_1.shape[1])
            roi_x2 = min(x + int(w * 0.8), mask_1.shape[1])

            bottom_center_roi = mask_1[roi_y1:roi_y2, roi_x1:roi_x2]

            # Calculate how much ink is in this bottom gap
            total_pixels = bottom_center_roi.size
            if total_pixels > 0:
                ink_pixels = cv2.countNonZero(bottom_center_roi)
                density = ink_pixels / total_pixels

                # If the gap is mostly empty (less than 15% ink), it's a calibration pulse!
                if density < 0.15:
                    # Get the 4 corners of the rotated bounding box for drawing/returning
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    detected_pulses.append(box)

    return detected_pulses
