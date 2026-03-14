import numpy as np
from matplotlib import pyplot as plt

from src.utils import save_step


def read_slice_signal(slice: np.ndarray, hmm: float) -> float | None:
    slice_size = slice.shape[0]

    one_indices = np.where(slice == 1)

    if len(one_indices[0]) == 0:
        return None

    median = np.median(one_indices)

    value = 1.0 - (median / slice_size) - 0.5
    value *= hmm
    return value


def read_lead_signal(image: np.ndarray) -> np.ndarray:
    values = []
    hmm = image.shape[1] / image.shape[0] * 65 / 2.0
    for x in range(image.shape[1]):
        slice = image[:, x]
        value = read_slice_signal(slice, hmm=hmm)
        if value is not None:
            values.append(value)
        else:
            values.append(np.nan)

    values = np.array(values)

    # interpolate to 1250 values
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, 1250)
    values = np.interp(new_x, old_x, values)

    return values


def read_line_signal(image: np.ndarray):
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Calculate the width of each of the 4 segments
    # Using integer division (//) to get whole numbers
    w_step = width // 4

    # Slice the arrays: array[height_start:height_end, width_start:width_end]
    image1 = image[:, :w_step]
    image2 = image[:, w_step:w_step * 2]
    image3 = image[:, w_step * 2:w_step * 3]
    image4 = image[:, w_step * 3:]  # Grabs the remainder up to the end

    save_step("3_lead_1", "line1.png", image1 * 255)
    save_step("3_lead_2", "line1.png", image2 * 255)
    save_step("3_lead_3", "line1.png", image3 * 255)
    save_step("3_lead_4", "line1.png", image4 * 255)

    s1 = read_lead_signal(image1)
    s2 = read_lead_signal(image2)
    s3 = read_lead_signal(image3)
    s4 = read_lead_signal(image4)

    plt.plot(s1)
    plt.savefig("data1/s1.png", dpi=300, bbox_inches="tight")
    plt.close()

    return s1, s2, s3, s4


def read_signal(line1: np.ndarray, line2: np.ndarray, line3: np.ndarray, filename: str) -> dict[str, np.ndarray]:
    # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    I, AVR, V1, V4 = read_line_signal(line1)
    II, AVL, V2, V5 = read_line_signal(line2)
    III, AVF, V3, V6 = read_line_signal(line3)

    sub_dict = {
        'I': I,
        'II': II,
        'III': III,
        'AVR': AVR,
        'AVL': AVL,
        'AVF': AVF,
        'V1': V1,
        'V2': V2,
        'V3': V3,
        'V4': V4,
        'V5': V5,
        'V6': V6,
    }

    record_name = filename.split('.')[0]
    return {f"{record_name}_{lead}": signal.astype(np.float16) for lead, signal in sub_dict.items()}
