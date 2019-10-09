import cv2
import numpy as np

OUT_PATH = "./out/question_1/"
REPORT_PATH = "./Report/images/question_1"


# ==================== Helper Methods Start ====================
def directory_setup() -> None:
    """
    Sets up the required output directory

    :return: None
    """
    from pathlib import Path

    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).mkdir(parents=True, exist_ok=True)


def load_image(src_path: str, gray=False) -> np.ndarray:
    """
    Loads an image from the src path into a np array.

    :param src_path: the path for the source image
    :param gray: true if return grayscale image, false otherwise
    :return: the converted np array for the image
    """
    img = cv2.imread(src_path)
    code = cv2.COLOR_BGR2GRAY if gray else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(img, code)


def save_image(np_data: np.ndarray, target_path: str) -> None:
    """
    Saves the np img to a file to the target path as a grayscale image.

    :param np_data: the np img
    :param target_path: the target file path
    :return: None
    """
    cv2.imwrite(target_path, np_data)


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


def my_correlation(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    My implementation of the Correlation operator.

    Assumptions:
    filter h has odd number of rows and columns

    :param img: an np array that represents the input grayscale image
    :param h: an np array that represents a filter
    :return: the output image produced according to the mode
    """
    if img.ndim < 2 or img.ndim > 3:
        raise Exception("image dimension must be 2 or 3")

    h_h, h_w = h.shape
    half_h_h, half_h_w = h_h // 2, h_w // 2

    padded_image = zero_pad(img, half_h_h, half_h_w)
    row_range = img.shape[0]
    col_range = img.shape[1]

    if img.ndim == 2:
        out_img = np.zeros((row_range, col_range))
    else:
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


def my_convolution(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    My implementation of the Convolution operator.

    Assumptions:
    filter h has odd number of rows and columns

    :param img: an np array that represents the input grayscale image
    :param h: an np array that represents a filter
    :return: the output image produced according to the mode
    """
    h = np.flip(h)
    return my_correlation(img, h)


# ==================== Helper Methods End ====================


def _get_1d_linear_interpolation_filter(d: int) -> np.ndarray:
    """
    Gets the 1D reconstruction filter.

    The filter should be [1/d, 2/d, ..., (d-1)/d, 1, (d-1)/d, ..., 2/d, 1/d]

    :param d: the image will be up-sampled by factor of d
    :return: the 1D reconstruction filter corresponding to the scale
    """
    h = np.zeros(2 * (d - 1) + 1)
    for i in range(len(h)):
        if i < d - 1:
            h[i] = (i + 1) / d
        else:
            h[i] = (2 * d - i - 1) / d
    return h


def my_linear_interpolation(img: np.ndarray, d: int,
                            dim: int) -> np.ndarray:
    """
    Apply 1D Linear Interpolation on an image on the given dimension to
    up-sample the image by scale.

    :param img: the original image
    :param d: the scale of the up-sampling
    :param dim: the target dimension.
        can be only 0 (vertical) or 1 (horizontal)
    :return: the up-sampled result
    """
    if dim not in (0, 1):
        raise ValueError("The target dimension must be either 0 or 1.")
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("image dimension must be 2 or 3")

    # calculate the 1d reconstruction filter
    h = _get_1d_linear_interpolation_filter(d)
    # adjust the shape for the reconstruction filter
    h = h[np.newaxis]
    if dim == 0:
        h = h.T

    # initialize up-sampled image and fill in the initial values at
    # where i/d is an integer
    shape = img.shape
    if dim == 0:
        shape = (d * shape[0], shape[1], *shape[2:])
    else:
        shape = (shape[0], d * shape[1], *shape[2:])
    new_img = np.zeros(shape)
    for i in range(new_img.shape[dim]):
        if i % d == 0:
            if dim == 0:
                new_img[i, :, :] = img[i // d, :, :]
            else:
                new_img[:, i, :] = img[:, i // d, :]

    return my_convolution(new_img, h)


def generalized_linear_interpolation(img: np.ndarray, d: int) -> np.ndarray:
    """

    :param img:
    :param d:
    :return:
    """
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError("image dimension must be 2 or 3")

    # calculate the 1d reconstruction filter
    h_1d = _get_1d_linear_interpolation_filter(d)
    # adjust the shape for the reconstruction filter
    h = np.outer(h_1d, h_1d)

    # initialize up-sampled image and fill in the initial values at
    # where i/d is an integer
    shape = img.shape
    shape = (d * shape[0], d * shape[1], *shape[2:])
    new_img = np.zeros(shape)
    for i in range(new_img.shape[0]):
        if i % d == 0:
            for j in range(new_img.shape[1]):
                if j % d == 0:
                    new_img[i, j, :] = img[i // d, j // d, :]

    return my_convolution(new_img, h)


def part1() -> None:
    # ========== part a ==========
    print("{0} a {0}".format("=" * 15))
    enlarged_image = my_linear_interpolation(bee_img, 4, 0)
    enlarged_image = my_linear_interpolation(enlarged_image, 4, 1)
    save_image(enlarged_image, "./out/question_1/1_1.jpg")


def part2() -> None:
    # ========== part b ==========
    print("{0} b {0}".format("=" * 15))
    enlarged_image = generalized_linear_interpolation(bee_img, 4)
    save_image(enlarged_image, "./out/question_1/1_2.jpg")


def main() -> None:
    part1()

    part2()


if __name__ == '__main__':
    directory_setup()
    bee_img = load_image("./resource/bee.jpg")

    print("{0} Question 1 {0}".format("=" * 20))
    main()
    print("{0} Done {0}".format("=" * 20))
