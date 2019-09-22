import numpy as np


def zero_pad(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Zero pads an image with specified width and height

    :param img: the image to zero pad
    :param height: the height for the padding
    :param width: the width for the padding
    :return: the zero-padded image
    """
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
    h_h, h_w = h.shape
    half_h_h, half_h_w = h_h // 2, h_w // 2

    # Figure out the zero padding sizes according to mode
    if mode == "valid":
        pad_image = zero_pad(img, 0, 0)
        out_img = np.zeros(pad_image.shape)
    elif mode == "same":
        pad_image = zero_pad(img, half_h_h, half_h_w)
        out_img = np.zeros(pad_image.shape)
    elif mode == "full":
        pad_image = zero_pad(img, h_h, h_w)
        out_img = np.zeros(pad_image.shape)
    else:
        raise Exception("mode should have value of 'valid', 'same' or 'full'.")

    for i in range(half_h_h, pad_image.shape[0] - half_h_h):
        for j in range(half_h_w, pad_image.shape[1] - half_h_h):
            # print(i, j,
            #       pad_image[i - half_h_h: i + half_h_h + 1,
            #                 j - half_h_w: j + half_h_w + 1].shape)
            tmp = np.sum(
                pad_image[i - half_h_h: i + half_h_h + 1,
                j - half_h_w: j + half_h_w + 1]
                * h)
            out_img[i - half_h_h][j - half_h_w] = tmp

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


def my_portrait_mode(img: np.ndarray) -> np.ndarray:
    """
    My implementation of the Portrait mode.

    Assumptions:
    filter h has odd number of rows and columns

    :param img: an np array that represents the input grayscale image
    :return: the output image produced by the portrait mode
    """
    pass


if __name__ == '__main__':
    pass
