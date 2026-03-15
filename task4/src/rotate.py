import numpy as np
from src.load_annotations import Annotation
import cv2

def rotate(image: np.ndarray, annotation: Annotation) -> tuple[np.ndarray, Annotation]:
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    p1, p2 = annotation

    angles = []

    for line in lines:
        rho, theta = line[0]
        angle = theta - np.pi/2
        angles.append(angle)

    angle = np.median(angles)
    angle_deg = np.degrees(angle)
    (h, w) = image.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    rotated = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

    points = np.array([[p1, p2]], dtype=np.float32)
    points_rot = cv2.transform(points, M)[0]
    p1_rot, p2_rot = points_rot
    annotation = Annotation((p1_rot, p2_rot))

    return rotated, annotation