from typing import Tuple

import numpy as np


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


# ==================== Question 4 ====================
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
        # TODO: this doesn't seem right yet
        # could do my_correlation for each k and concatenate
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
    pass
