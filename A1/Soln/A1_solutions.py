from PIL import Image

from question_4 import *
from question_5 import *
from question_6 import *


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


# ================== QUESTIONS ==================
def question_4() -> None:
    print("{0} Question 4 {0}".format("=" * 20))
    sharpen_filter = np.asarray([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - \
                     1 / 9 * np.ones((3, 3))
    print(sharpen_filter)
    for mode in ["valid", "same", "full"]:
        print("{0} {1} {0}".format("=" * 10, mode))

        print("processing correlation")
        correlation_out_data = my_correlation(gray_img, sharpen_filter, mode)
        save_image(correlation_out_data,
                   "out/4_correlation_{}.jpg".format(mode))

        print("processing convolution")
        convolution_out_data = my_convolution(gray_img, sharpen_filter, mode)
        save_image(convolution_out_data,
                   "out/4_convolution_{}.jpg".format(mode))


def question_5() -> None:
    # Question 5
    print("{0} Question 5 {0}".format("=" * 20))
    separable_filter = np.asarray([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]) / 16
    inseparable_filter = np.asarray([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
    print(is_separable_filter(separable_filter))
    print(is_separable_filter(inseparable_filter))


def question_6() -> None:
    print("{0} Question 6 {0}".format("=" * 20))
    # part a
    # scaled_img = gray_img / 255.0
    # noise_out = add_rand_noise(scaled_img, [-0.05, 0.05]) * 255.0
    # save_image(noise_out, "./out/6_a_rand_noise.jpg")
    #
    # # part b
    # # Use mean filter because each pixel gets set to the average of the pixels
    # # in its neighborhood, local variations caused by grain are reduced.
    # mean_f = np.ones((3, 3)) / 9
    # out_img = my_convolution(noise_out, mean_f, "valid")
    # save_image(out_img, "./out/6_b_mean.jpg")
    #
    # # part c
    # noise_out = add_salt_and_pepper_noise(gray_img, 0.05)
    # save_image(noise_out, "./out/6_c_sp_noise.jpg")
    #
    # # part d
    # mean_f = np.ones((3, 3)) / 9
    # out_img = my_convolution(noise_out, mean_f, "valid")
    # save_image(out_img, "./out/6_d_mean.jpg")
    # out_img = median_filter(noise_out)
    # save_image(out_img, "./out/6_d_median.jpg")

    # part e
    noise_out = add_salt_and_pepper_noise(color_img, 0.05)
    save_image(noise_out, "./out/6_e_noise.jpg")

    out_img = denoise_colored(noise_out)
    save_image(out_img, "./out/6_e_clean.jpg")


if __name__ == '__main__':
    gray_img = load_image("./gray.jpg")
    color_img = load_image("./color.jpg")

    # question_4()

    # question_5()

    question_6()
