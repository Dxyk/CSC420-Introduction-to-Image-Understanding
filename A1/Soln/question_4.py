from typing import Tuple

import numpy as np
from PIL import Image


# ================== Helpers ==================
def load_image(src_path: str) -> np.ndarray:
    """
    Loads a grayscale image from the src path into a np array.

    :param src_path: the path for the source image
    :return: the converted np array for the image
    """
    img = Image.open(src_path)
    return np.asarray(img, dtype="float32")


def save_image(np_data: np.ndarray, target_path: str) -> None:
    """
    Saves the np img to a file to the target path as a grayscale image.

    :param np_data: the np img
    :param target_path: the target file path
    :return: None
    """
    if np_data.ndim == 2:
        img = Image.fromarray(
            np.asarray(np.clip(np_data, 0, 255), dtype="uint8"), "L")
    elif np_data.ndim == 3:
        img = Image.fromarray(np_data.astype(np.uint8))
    else:
        raise Exception("np_data must be either 2 or 3 dimensional")
    img.save(target_path)


# ==================== Question 4 ====================
def zero_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Zero pads an image with specified width and height

    :param img: the image to zero pad
    :param height: the height for the padding
    :param width: the width for the padding
    :return: the zero-padded image
    """
    if img.ndim == 3:
        img_h, img_w, num_channel = img.shape
        out_img = np.zeros((img_h + 2 * height, img_w + 2 * width, 3))
        out_img[height:img_h + height, width:img_w + width, :] = img
    else:
        img_h, img_w = img.shape
        out_img = np.zeros((img_h + 2 * height, img_w + 2 * width))
        out_img[height:img_h + height, width:img_w + width] = img
    return out_img


def my_correlation(img: np.ndarray, h: np.ndarray, mode: str) -> np.ndarray:
    """
    My implementation of the Correlation operator.

    Assumptions:
    filter h has odd number of rows and columns

    :param img: an np array that represents the input grayscale image
    :param h: an np array that represents a filter
    :param mode: the mode. Possible values are 'valid', 'same' or 'full'
    :return: the output image produced according to the mode
    """
    if mode not in ["valid", "same", "full"]:
        raise Exception("mode should have value of 'valid', 'same' or 'full'.")
    if img.ndim < 2 or img.ndim > 3:
        raise Exception("image dimension must be 2 or 3")

    h_h, h_w = h.shape
    half_h_h, half_h_w = h_h // 2, h_w // 2

    # Figure out the zero padding sizes according to mode
    if mode == "valid":
        padded_image = zero_pad(img, 0, 0)
        row_range = img.shape[0] - half_h_w * 2
        col_range = img.shape[1] - half_h_h * 2
    elif mode == "same":
        padded_image = zero_pad(img, half_h_h, half_h_w)
        row_range = img.shape[0]
        col_range = img.shape[1]
    else:  # full
        padded_image = zero_pad(img, h_h, h_w)
        row_range = padded_image.shape[0] - half_h_w * 2
        col_range = padded_image.shape[1] - half_h_h * 2

    if img.ndim == 2:
        out_img = np.zeros((row_range, col_range))
    else:  # 3 dimension
        out_img = np.zeros((row_range, col_range, 3))

    if img.ndim == 2:
        for i in range(0, row_range):
            for j in range(0, col_range):
                res = np.sum(padded_image[i: i + h_h,
                             j: j + h_w] * h)
                out_img[i][j] = res
    elif img.ndim == 3:
        for i in range(0, row_range):
            for j in range(0, col_range):
                for k in range(img.shape[2]):
                    res = np.sum(padded_image[i: i + h_h,
                                 j: j + h_w, k] * h)
                    out_img[i][j][k] = res

    return out_img


def my_convolution(img: np.ndarray, h: np.ndarray, mode: str) -> np.ndarray:
    """
    My implementation of the Convolution operator.

    Assumptions:
    filter h has odd number of rows and columns

    :param img: an np array that represents the input grayscale image
    :param h: an np array that represents a filter
    :param mode: the mode. Possible values are 'valid', 'same' or 'full'
    :return: the output image produced according to the mode
    """
    h = np.flip(h)
    return my_correlation(img, h, mode)


def my_portrait_mode(img: np.ndarray,
                     top_left: Tuple[int, int],
                     bottom_right: Tuple[int, int]) -> np.ndarray:
    """
    My implementation of the Portrait mode.

    Assumptions:
    filter h has odd number of rows and columns

    :param img: an np array that represents the input grayscale image
    :param top_left: top left coordinates of the focus rectangle
    :param bottom_right: bottom right coordinates of the focus rectangle
    :return: the output image produced by the portrait mode
    """
    orig_crop = img[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
    # a 3 * 3 filter was too weak. The effect was not obvious
    # blur_filter = 1 / 9 * np.ones((3, 3))
    blur_filter = 1 / 25 * np.ones((5, 5))
    out_img = my_convolution(img, blur_filter, "same")
    out_img[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]] = \
        orig_crop
    return out_img


if __name__ == '__main__':
    # setup: create output directories
    from pathlib import Path

    Path('out/question_4/gray/').mkdir(parents=True, exist_ok=True)
    Path('out/question_4/color/').mkdir(parents=True, exist_ok=True)

    gray_img = load_image("./gray.jpg")
    color_img = load_image("./color.jpg")
    q4c_img = load_image("./4_c_orig.jpg")

    print("{0} Question 4 {0}".format("=" * 20))

    # part a
    print("{0} a {0}".format("=" * 10))
    h = np.asarray([[0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0]]) \
        - 1 / 9 * np.ones((3, 3))
    for mode in ["valid", "same", "full"]:
        print("{0} {1} {0}".format("=" * 5, mode))
        print("processing gray")
        gray_out = my_correlation(gray_img, h, mode)
        save_image(gray_out, "out/question_4/gray/a_{}.jpg".format(mode))
        print("processing color")
        color_out = my_correlation(color_img, h, mode)
        save_image(color_out, "out/question_4/color/a_{}.jpg".format(mode))

    # part b
    print("{0} b {0}".format("=" * 10))
    for mode in ["valid", "same", "full"]:
        print("{0} {1} {0}".format("=" * 5, mode))
        print("processing gray")
        gray_out = my_convolution(gray_img, h, mode)
        save_image(gray_out, "out/question_4/gray/b_{}.jpg".format(mode))
        print("processing color")
        color_out = my_convolution(color_img, h, mode)
        save_image(color_out, "out/question_4/color/b_{}.jpg".format(mode))

    # part c
    print("{0} c {0}".format("=" * 10))
    q4c_out = my_portrait_mode(q4c_img, (125, 125), (300, 270))
    save_image(q4c_out, "out/question_4/color/c_out.jpg")
