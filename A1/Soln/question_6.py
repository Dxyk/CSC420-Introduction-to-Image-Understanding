from typing import List

import cv2
from question_4 import *


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
        # add s&p noise to each channel but with d /= 3 (there are 3 channels)
        for k in range(img.shape[2]):
            out[:, :, k] = add_salt_and_pepper_noise(img[:, :, k], d / 3)
    return out


def median_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply median filter to the image to reduce salt and pepper noise

    :param img: the noisy image
    :return: the smoothed image
    """
    return cv2.medianBlur(np.float32(img), 3)


if __name__ == '__main__':
    gray_img = load_image("./gray.jpg")
    color_img = load_image("./color.jpg")

    print("{0} Question 6 {0}".format("=" * 20))

    # part a
    print("{0} a {0}".format("=" * 10))
    print("{0} gray {0}".format("=" * 5))
    scaled_gray = gray_img / 255.0
    noisy_gray = add_rand_noise(scaled_gray, [-0.05, 0.05]) * 255.0
    save_image(noisy_gray, "./out/question_6/gray/a.jpg")
    print("{0} color {0}".format("=" * 5))
    scaled_color = color_img / 255.0
    noisy_color = add_rand_noise(scaled_color, [-0.05, 0.05]) * 255.0
    save_image(noisy_color, "./out/question_6/color/a.jpg")

    # part b
    print("{0} b {0}".format("=" * 10))
    # Use mean filter because each pixel gets set to the average of the pixels
    # in its neighborhood, local variations caused by grain are reduced.
    mean_h = np.ones((3, 3)) / 9
    print("{0} gray {0}".format("=" * 5))
    gray_out = my_convolution(noisy_gray, mean_h, "same")
    save_image(gray_out, "./out/question_6/gray/b.jpg")
    print("{0} color {0}".format("=" * 5))
    color_out = my_convolution(noisy_color, mean_h, "same")
    save_image(color_out, "./out/question_6/color/b.jpg")

    # part c
    print("{0} c {0}".format("=" * 10))
    print("{0} gray {0}".format("=" * 5))
    noisy_gray = add_salt_and_pepper_noise(gray_img, 0.05)
    save_image(noisy_gray, "./out/question_6/gray/c.jpg")
    print("{0} color {0}".format("=" * 5))
    noisy_color = add_salt_and_pepper_noise(color_img, 0.05)
    save_image(noisy_color, "./out/question_6/color/c.jpg")

    # part d
    print("{0} d {0}".format("=" * 10))
    print("{0} gray {0}".format("=" * 5))
    mean_h = np.ones((3, 3)) / 9
    gray_out = my_convolution(noisy_gray, mean_h, "same")
    save_image(gray_out, "./out/question_6/gray/d_mean.jpg")
    gray_out = median_filter(noisy_gray)
    save_image(gray_out, "./out/question_6/gray/d_median.jpg")
    print("{0} color {0}".format("=" * 5))
    color_out = my_convolution(noisy_color, mean_h, "same")
    save_image(color_out, "./out/question_6/color/d_mean.jpg")
    color_out = median_filter(noisy_color)
    save_image(color_out, "./out/question_6/color/d_median.jpg")

    # part e
    print("{0} e {0}".format("=" * 10))
    noisy_color = add_salt_and_pepper_noise(color_img, 0.05)
    save_image(noisy_color, "./out/question_6/color/e_noisy.jpg")

    color_out = median_filter(noisy_color)
    save_image(color_out, "./out/question_6/color/e_median.jpg")
