from typing import List

import cv2
import numpy as np


# ==================== Question 6 ====================
def add_rand_noise(img: np.ndarray, m: List[float]) -> np.ndarray:
    """
    Add random noise of range m to the image.

    :param img: an np array that represents the input grayscale image
    :param m: the magnitude of the noise
    :return: the output image with random uniform noise
    """
    noise = np.random.uniform(m[0], m[1], img.shape)
    return img + noise


def add_salt_and_pepper_noise(img: np.ndarray, d: float) -> np.ndarray:
    """
    Add salt and pepper noise of density d to the image

    :param img: the input grayscale image
    :param d: the density of the noise
    :return: the output image with salt and pepper noise
    """
    out = np.copy(img)

    if img.ndim == 2:
        prob_matrix = np.random.rand(out.shape[0], out.shape[1])
        out = np.where(prob_matrix > d / 2, out, 1)
        out = np.where(prob_matrix < (1 - d / 2), out, 0)
    elif img.ndim == 3:
        # print(img.shape)
        r, g, b = img[:, :, 0], img[:, :, 0], img[:, :, 0]
        # print(r.shape)
        # print(g.shape)
        # print(b.shape)
        # prob_matrix = np.random.rand(out.shape[0], out.shape[1])
        # r = np.where(prob_matrix > d / 2, r, 1)
        # r = np.where(prob_matrix < (1 - d / 2), r, 0)
        # prob_matrix = np.random.rand(out.shape[0], out.shape[1])
        # g = np.where(prob_matrix > d / 2, g, 1)
        # g = np.where(prob_matrix < (1 - d / 2), g, 0)
        # prob_matrix = np.random.rand(out.shape[0], out.shape[1])
        # b = np.where(prob_matrix > d / 2, b, 1)
        # b = np.where(prob_matrix < (1 - d / 2), b, 0)
        #
        out = np.stack([r, g, b], axis=2)
        print(out)
        print(out.shape)
        for i in range(img.shape[2]):
            prob_matrix = np.random.rand(out.shape[0], out.shape[1])
            out[:, :, i] = np.where(prob_matrix > d / 6, out[:, :, i],
                                    np.ones((out.shape[0], out.shape[1])))
            out[:, :, i] = np.where(prob_matrix < (1 - d / 6), out[:, :, i],
                                    np.ones((out.shape[0], out.shape[1])))
    return out


def median_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply median filter to the image to reduce salt and pepper noise

    :param img: the noisy image
    :return: the smoothed image
    """
    return cv2.medianBlur(np.float32(img), 3)


def denoise_colored(img: np.ndarray) -> np.ndarray:
    """
    Denoise an image with salt and pepper noise

    :param img: the noisy image
    :return: the clean image
    """
    # converted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.fastNlMeansDenoisingColored(img)


if __name__ == '__main__':
    pass
