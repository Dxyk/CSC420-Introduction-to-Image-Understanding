from typing import Tuple

import cv2
import numpy as np

OUT_PATH = "./out/question_2/"
TUNING_PATH = OUT_PATH + "tuning/"
REPORT_PATH = "./Report/images/question_2/"


# ==================== Helper Methods Start ====================
def directory_setup() -> None:
    """
    Sets up the required output directory

    :return: None
    """
    from pathlib import Path

    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    Path(TUNING_PATH).mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).mkdir(parents=True, exist_ok=True)


def load_image(src_path: str, gray=False) -> np.ndarray:
    """
    Loads an image from the src path into a np array.

    :param src_path: the path for the source image
    :param gray: true if return grayscale image, false otherwise
    :return: the converted np array for the image
    """
    code = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(src_path, code)
    return img


def save_image(np_data: np.ndarray, target_path: str) -> None:
    """
    Saves the np img to a file to the target path as a grayscale image.

    :param np_data: the np img
    :param target_path: the target file path
    :return: None
    """
    cv2.imwrite(OUT_PATH + target_path, np_data)
    if REPORT:
        cv2.imwrite(REPORT_PATH + target_path, np_data)


# ==================== Helper Methods End ====================


# ========== part a ==========
def _generate_det_and_trace(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates the determinant and trace matrices according to the image

    :param img: the image
    :return: the determinant and the trace matrices
    """
    blur = cv2.GaussianBlur(img, (5, 5), 7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    Ix2_blur = cv2.GaussianBlur(Ix2, (7, 7), 10)
    Iy2_blur = cv2.GaussianBlur(Iy2, (7, 7), 10)
    IxIy_blur = cv2.GaussianBlur(IxIy, (7, 7), 10)
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur, IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    return det, trace


def _non_maximum_suppression(m: np.ndarray, neighbour_size: int) -> np.ndarray:
    """
    Performs Non-Maximum Suppression on the given matrix.

    :param m: the given matrix
    :param neighbour_size: the size of the neighbourhood in order to compare
    :return: the matrix after NMS
    """
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            x_min_idx = max(x - neighbour_size, 0)
            x_max_idx = min(x + neighbour_size, m.shape[0])
            y_min_idx = max(y - neighbour_size, 0)
            y_max_idx = min(y + neighbour_size, m.shape[1])
            if m[x, y] != np.amax(m[x_min_idx: x_max_idx,
                                  y_min_idx: y_max_idx]):
                m[x, y] = 0
    return m


def corner_detection_harris(img: np.ndarray, alpha: float = 0.06,
                            threshold: float = 1e10,
                            neighbour_size: int = 2) -> np.ndarray:
    """
    Return the harris corner detection matrix.

    :param img: the image
    :param alpha: the alpha value that controls the trace
    :param threshold: the threshold for finding large R
    :param neighbour_size: the size of the neighbourhood for NMS
    :return: the generated corner matrix
    """
    print("{0} Harris with alpha={1} and "
          "threshold={2} {0}".format("=" * 5, alpha, threshold))
    det, trace = _generate_det_and_trace(img)
    R = det - alpha * np.multiply(trace, trace)
    R[R < threshold] = 0
    R = _non_maximum_suppression(R, neighbour_size)
    return R


def corner_detection_brown(img: np.ndarray,
                           threshold: float = 2e5,
                           neighbour_size: int = 2) -> np.ndarray:
    """
    Return the Brown corner detection matrix.

    :param img: the image
    :param threshold: the threshold for finding large B
    :param neighbour_size: the size of the neighbourhood for NMS
    :return: the generated corner matrix
    """
    print("{0} Brown with threshold={1} {0}".format("=" * 5, threshold))
    det, trace = _generate_det_and_trace(img)
    B = np.divide(det, trace)
    B[B < threshold] = 0
    B = _non_maximum_suppression(B, neighbour_size)
    return B


def tune_params_a() -> None:
    """
    Tunes the parameters by cycling through possible options and comparing

    :return: None
    """
    # tune for alpha
    # by comparing the results, alpha = 0.06 gives a better result because its
    # image contains less noise
    alphas = [0.04, 0.045, 0.05, 0.055, 0.06]
    for alpha in alphas:
        R = corner_detection_harris(img, alpha)
        save_image(R, "tuning_Harris_{}.jpg".format(alpha))

    thresholds = [1000, 10000, 50000, 100000, 2e5, 1e6, 1e8, 1e10]
    for threshold in thresholds:
        # tune for R threshold
        # by comparing the results, the best threshold is 1e10 because this
        # yields an appropriate amount of corners
        R = corner_detection_harris(img, threshold=threshold)
        save_image(R, "tuning_Harris_{}.jpg".format(threshold))

        # tune for B threshold
        # by comparing the results, the best threshold is 2e5
        B = corner_detection_brown(img, threshold=threshold)
        save_image(B, "tuning_Brown_{}.jpg".format(threshold))


# ========== part b ==========
def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image with angle (degrees) counter-clockwise

    :param img: the image
    :param angle: the angle in degrees
    :return: the rotated image
    """
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rotated_img = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotated_img, img.shape[1::-1],
                                 flags=cv2.INTER_LINEAR)
    return rotated_img


# ========== part c ==========
def find_interest_points(img: np.ndarray, threshold: float = 4000) -> np.ndarray:
    interest_points = []
    scale_map = []
    sigma_list = [0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3, 2.5, 2.9, 3.0, 3.1,
                  3.5, 4.0, 4.5, 4.8, 5.0, 5.3, 5.8, 6.4]
    for sigma in sigma_list:
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        Ix2 = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        Iy2 = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=5)
        blur_Ix2 = cv2.GaussianBlur(Ix2, (15, 15), sigma)
        blur_Iy2 = cv2.GaussianBlur(Iy2, (15, 15), sigma)
        laplacian = np.sqrt(np.square(blur_Ix2) + np.square(blur_Iy2))
        scale_map.append(laplacian)

    def neighbour_mode(scale_map, l, x, y, mode):
        neighbour = []
        if mode == "plus":
            neighbour.append(scale_map[l].item(x, y))
        if x < scale_map[0].shape[0] - 1:
            neighbour.append(scale_map[l].item(x + 1, y))
        if y < scale_map[0].shape[1] - 1:
            neighbour.append(scale_map[l].item(x, y + 1))
        if x < scale_map[0].shape[0] - 1 and y < scale_map[0].shape[1] - 1:
            neighbour.append(scale_map[l].item(x + 1, y + 1))
        if x >= 1 and y < scale_map[0].shape[1] - 1:
            neighbour.append(scale_map[l].item(x - 1, y))
            neighbour.append(scale_map[l].item(x - 1, y + 1))
        if y >= 1 and x < scale_map[0].shape[0] - 1:
            neighbour.append(scale_map[l].item(x, y - 1))
            neighbour.append(scale_map[l].item(x + 1, y - 1))
        if x >= 1 and y >= 1:
            neighbour.append(scale_map[l].item(x - 1, y - 1))
        return neighbour

    def find_neighbour(scale_map, l, x, y):
        neighbour = neighbour_mode(scale_map, l, x, y, "no")
        if l >= 1:
            neighbour.extend(neighbour_mode(scale_map, l - 1, x, y, "plus"))
        if l < len(sigma_list) - 1:
            neighbour.extend(neighbour_mode(scale_map, l + 1, x, y, "plus"))
        return neighbour

    for i in range(len(sigma_list) - 1):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                pixel = scale_map[i][x, y]
                if pixel > threshold:
                    neighbour = find_neighbour(scale_map, i, x, y)
                    Min, Max = True, True
                    for n in neighbour:
                        if n >= pixel:
                            Max = False
                        if n <= pixel:
                            Min = False
                    if Max or Min:
                        interest_points.append([x, y, sigma_list[i]])
    print(interest_points)
    return interest_points


# ========== mains ==========
def part_a() -> None:
    print("{0} a {0}".format("=" * 15))
    # tune parameters
    # tune_params_a()

    R = corner_detection_harris(img)
    save_image(R, "2_1_Harris.jpg")
    B = corner_detection_brown(img)
    save_image(B, "2_1_Brown.jpg")


def part_b() -> None:
    print("{0} b {0}".format("=" * 15))
    save_image(img, "2_2_orig_img.jpg")
    R = corner_detection_harris(img)
    save_image(R, "2_2_orig_R.jpg")

    rotated_img = rotate_img(img, 60)
    save_image(rotated_img, "2_2_rotated_img.jpg")
    rotated_R = corner_detection_harris(rotated_img)
    save_image(rotated_R, "2_2_rotated_R.jpg")

    rerotated_r = rotate_img(rotated_R, -60)
    save_image(rerotated_r, "2_2_rerotated_R.jpg")

    num_hi = 0
    num_match = 0
    for i in range(rerotated_r.shape[0]):
        for j in range(rerotated_r.shape[1]):
            if rerotated_r[i, j] != 0:
                num_hi += 1
                if np.count_nonzero(img[i, j]) != 0:
                    num_match += 1
    print("# total: {}; # match: {}".format(num_hi, num_match))
    print("match percentage: {}%".format(num_match / num_hi * 100))


def part_c() -> None:
    print(find_interest_points(img))


def main() -> None:
    # part_a()

    # part_b()

    part_c()


if __name__ == '__main__':
    REPORT = False

    directory_setup()
    img = load_image("./resource/building.jpg", gray=True)

    print("{0} Question 2 {0}".format("=" * 20))
    main()
    print("{0} Done {0}".format("=" * 20))
