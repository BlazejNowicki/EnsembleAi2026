import numpy as np
from matplotlib import pyplot as plt

from src.utils import save_step, SUBMIT

WIDTH_MM = 62.5


def read_slice_signal(slice: np.ndarray, hmm: float) -> float | None:
    slice_size = slice.shape[0]

    one_indices = np.where(slice == 1)

    if len(one_indices[0]) == 0:
        return None

    median = np.median(one_indices)

    value = 1.0 - (median / slice_size) - 0.5
    value *= hmm / 10
    return value


def read_lead_signal(image: np.ndarray) -> np.ndarray:
    # cut beginning and ending
    # beginning_cut = int(0.008 * image.shape[1])
    # end_cut = int(0.008 * image.shape[1])
    # print(beginning_cut, end_cut)
    # image = image[:, beginning_cut:image.shape[1] - end_cut]

    values = []

    hmm = image.shape[0] / image.shape[1] * WIDTH_MM
    for x in range(image.shape[1]):
        slice = image[:, x]
        value = read_slice_signal(slice, hmm=hmm)
        if value is not None:
            values.append(value)
        else:
            values.append(np.nan)

    values = np.array(values)

    # remove nans and interpolate to 1250 values
    valid_mask = ~np.isnan(values)

    # 2. Handle leading NaNs
    if valid_mask.any():
        # np.argmax returns the index of the FIRST True value
        first_valid_idx = np.argmax(valid_mask)
        # Fill the leading NaNs with 0
        values[:first_valid_idx] = 0
        # Update the mask so the new zeros are included in the interpolation
        valid_mask[:first_valid_idx] = True
    else:
        # Edge case: The array is 100% NaNs
        values[:] = 0
        valid_mask[:] = True

    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, 1250)

    # if all values were nan
    values = np.interp(new_x, old_x[valid_mask], values[valid_mask])

    return values


def remove_text(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    vertical_gap_mm = 3
    vertical_gap = int(vertical_gap_mm * w / WIDTH_MM)

    horizontal_gap = int(0.125 * w)

    image[int(0.5 * h) + vertical_gap:, :horizontal_gap] = 0

    return image


def read_line_signal(image: np.ndarray, filename: str):
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

    image1 = remove_text(image1)
    image2 = remove_text(image2)
    image3 = remove_text(image3)
    image4 = remove_text(image4)

    save_step("3_lead_1", filename, image1 * 255)
    save_step("3_lead_2", filename, image2 * 255)
    save_step("3_lead_3", filename, image3 * 255)
    save_step("3_lead_4", filename, image4 * 255)

    s1 = read_lead_signal(image1)
    s2 = read_lead_signal(image2)
    s3 = read_lead_signal(image3)
    s4 = read_lead_signal(image4)

    if not SUBMIT:
        plt.plot(s4)
        plt.savefig("data1/s4.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.plot(s1)
        plt.savefig("data1/s1.png", dpi=300, bbox_inches="tight")
        plt.close()

    return s1, s2, s3, s4


def read_signal(line1: np.ndarray, line2: np.ndarray, line3: np.ndarray, filename: str) -> dict[str, np.ndarray]:
    # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    I, AVR, V1, V4 = read_line_signal(line1, filename)
    II, AVL, V2, V5 = read_line_signal(line2, filename)
    III, AVF, V3, V6 = read_line_signal(line3, filename)

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
