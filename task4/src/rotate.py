import numpy as np
import cv2
from src.load_annotations import Annotation

def rotate(image: np.ndarray, annotation: Annotation) -> tuple[np.ndarray, Annotation]:
    p1, p2, p3 = annotation

    # Compute angle of the line p1 → p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_line = np.arctan2(dy, dx)  # radians

    # Compute perpendicular angle (the line we want to align with image edge)
    angle_perp = angle_line + np.pi/2  # 90° in radians

    # Convert to degrees
    angle_deg = np.degrees(angle_perp)

    # Use the same rotation approach as original function
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, -1.0)

    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    # Rotate annotation points
    points = np.array([[p1, p2, p3]], dtype=np.float32)
    points_rot = cv2.transform(points, M)[0]
    p1_rot, p2_rot, p3_rot = points_rot

    annotation_rot = Annotation((tuple(p1_rot), tuple(p2_rot), tuple(p3_rot)))
    return rotated, annotation_rot