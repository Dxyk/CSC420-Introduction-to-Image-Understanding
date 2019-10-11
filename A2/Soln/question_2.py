import pickle
from typing import Tuple, List, Any

import cv2
import numpy as np

OUT_PATH = "./out/question_2/"
TUNING_PATH = OUT_PATH + "tuning/"
REPORT_PATH = "./Report/images/question_2/"
PICKLE_PATH = OUT_PATH + "pickle/"


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
    Path(PICKLE_PATH).mkdir(parents=True, exist_ok=True)


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


def save_data(data: Any, target_path: str) -> None:
    """
    Saves serialized data to target path
    :param data: the data to save
    :param target_path: the target path
    :return: None
    """
    with open(PICKLE_PATH + target_path, 'wb') as f:
        pickle.dump(data, f)


def load_data(src_path: str) -> Any:
    """
    Loads the serialized data from the src path

    :param src_path: the path to the stored serialized data
    :return: the loaded data
    """
    with open(PICKLE_PATH + src_path, 'rb') as f:
        return pickle.load(f)


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
    m_cpy = np.copy(m)
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            x_min_idx = max(x - neighbour_size, 0)
            x_max_idx = min(x + neighbour_size, m.shape[0])
            y_min_idx = max(y - neighbour_size, 0)
            y_max_idx = min(y + neighbour_size, m.shape[1])
            if m[x, y] != np.amax(m[x_min_idx: x_max_idx,
                                  y_min_idx: y_max_idx]):
                m_cpy[x, y] = 0

    return m_cpy


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
        R = corner_detection_harris(img_gray, alpha)
        save_image(R, "tuning_Harris_{}.jpg".format(alpha))

    thresholds = [1000, 10000, 50000, 100000, 2e5, 1e6, 1e8, 1e10]
    for threshold in thresholds:
        # tune for R threshold
        # by comparing the results, the best threshold is 1e10 because this
        # yields an appropriate amount of corners
        R = corner_detection_harris(img_gray, threshold=threshold)
        save_image(R, "tuning_Harris_{}.jpg".format(threshold))

        # tune for B threshold
        # by comparing the results, the best threshold is 2e5
        B = corner_detection_brown(img_gray, threshold=threshold)
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
def find_interest_points(img: np.ndarray,
                         threshold: float = 100000) -> \
        List[Tuple[int, int, float]]:
    """
    Gets the interest points given the image using laplacian of gaussian at
    different scales.

    :param img: the target image
    :param threshold: the threshold for checking if the laplacian is the extrema
    :return: a list of interest points and their coordinates
    """
    import matplotlib.pyplot as plt
    interest_points = []
    scale_map: List[np.ndarray] = []
    # sigma_list = np.arange(0.5, 5.5, 0.5)
    sigma_list = [x / 10 for x in range(5, 55, 5)]

    print("Using {0} sigmas: {1}".format(len(sigma_list), sigma_list))

    for sigma in sigma_list:
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        Ix2 = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        Iy2 = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=5)
        blur_Ix2 = cv2.GaussianBlur(Ix2, (15, 15), sigma)
        blur_Iy2 = cv2.GaussianBlur(Iy2, (15, 15), sigma)
        laplacian = np.sqrt(np.square(blur_Ix2) + np.square(blur_Iy2))
        plt.imshow(laplacian, cmap='gray'), plt.axis('off')
        plt.show()
        scale_map.append(laplacian)

    def get_neighbours(scale, x_coord, y_coord):
        neighbours = []

        min_scale = max(0, scale - 1)
        max_scale = min(len(scale_map) - 1, scale + 1)
        min_x = max(0, x_coord - 1)
        max_x = min(h - 1, x_coord + 1)
        min_y = max(0, y_coord - 1)
        max_y = min(w - 1, y_coord + 1)
        for s in range(min_scale, max_scale + 1):
            for cur_x in range(min_x, max_x + 1):
                for cur_y in range(min_y, max_y + 1):
                    if s != scale or cur_x != x_coord or cur_y != y_coord:
                        neighbours.append(scale_map[s][cur_x, cur_y])
        return neighbours

    h, w = img.shape[0], img.shape[1]
    for i in range(len(sigma_list)):
        print("Processing {0} / {1} of the sigmas: {2}".format(i, len(sigma_list), sigma_list[i]))
        curr_sigma = sigma_list[i]
        for x in range(h):
            for y in range(w):
                pixel = scale_map[i][x, y]
                if pixel > threshold:
                    neighbours2 = get_neighbours(i, x, y)
                    if all([pixel > n for n in neighbours2]) or all([pixel < n for n in neighbours2]):
                        interest_points.append((x, y, curr_sigma))
    print("Found {0} interest points".format(len(interest_points)))
    return interest_points


def c_tuning() -> None:
    thresholds = [50000, 100000, 150000]
    for threshold in thresholds:
        print("Threshold: {}".format(threshold))
        img_cpy = np.copy(img_gray)
        interest_points = find_interest_points(img_cpy, threshold=threshold)
        save_data(interest_points, "tuning_2_3_{}.pkl".format(threshold))
        interest_points = load_data("tuning_2_3_{}.pkl".format(threshold))
        tmp = np.zeros(img_cpy.shape)
        for x, y, sigma in interest_points:
            tmp[x, y] = 255
            cv2.circle(img_cpy, (y, x), int(sigma * 3), (0, 0, 0), thickness=1)
        save_image(tmp, "tuning_points_2_3_{}.jpg".format(threshold))
        save_image(img_cpy, "tuning_2_3_{}.jpg".format(threshold))


# ========== mains ==========
def part_a() -> None:
    print("{0} a {0}".format("=" * 15))
    # tune parameters
    # tune_params_a()

    img_gray_cpy = np.copy(img_gray)
    R_cpy = np.copy(img_gray)
    B_cpy = np.copy(img_gray)

    R = corner_detection_harris(img_gray_cpy)
    for coord in np.transpose(np.nonzero(R)):
        x, y = coord[1], coord[0]
        cv2.circle(R_cpy, (x, y), 3, 0, 2)
    save_image(R, "2_1_Harris.jpg")
    save_image(R_cpy, "2_1_Harris_plotted.jpg")
    B = corner_detection_brown(img_gray)
    for coord in np.transpose(np.nonzero(B)):
        x, y = coord[1], coord[0]
        cv2.circle(B_cpy, (x, y), 3, 0, 2)
    save_image(B, "2_1_Brown.jpg")
    save_image(R_cpy, "2_1_Brown_plotted.jpg")


def part_b() -> None:
    print("{0} b {0}".format("=" * 15))
    img_gray_cpy = np.copy(img_gray)
    save_image(img_gray_cpy, "2_2_orig_img.jpg")

    R = corner_detection_harris(img_gray_cpy)
    save_image(R, "2_2_orig_R.jpg")

    rotated_img = rotate_img(img_gray, 60)
    save_image(rotated_img, "2_2_rotated_img.jpg")
    rotated_R = corner_detection_harris(np.copy(rotated_img))
    save_image(rotated_R, "2_2_rotated_R.jpg")

    rerotated_R = rotate_img(rotated_R, -60)
    save_image(rerotated_R, "2_2_rerotated_R.jpg")

    num_hi = 0
    num_match = 0
    for i in range(rerotated_R.shape[0]):
        for j in range(rerotated_R.shape[1]):
            if rerotated_R[i, j] != 0:
                num_hi += 1
                if np.count_nonzero(img_gray[i, j]) != 0:
                    num_match += 1
    print("# total: {}; # match: {}".format(num_hi, num_match))
    print("match percentage: {}%".format(num_match / num_hi * 100))

    for coord in np.transpose(np.nonzero(rotated_R)):
        x, y = coord[1], coord[0]
        cv2.circle(rotated_img, (x, y), 3, 0, 2)
    save_image(rotated_img, "2_2_rotated_plotted.jpg")

    for coord in np.transpose(np.nonzero(rerotated_R)):
        x, y = coord[1], coord[0]
        cv2.circle(img_gray_cpy, (x, y), 3, 0, 2)
    save_image(img_gray_cpy, "2_2_rerotated_plotted.jpg")


def part_c() -> None:
    print("{0} c {0}".format("=" * 15))
    # c_tuning()
    img_cpy = np.copy(img_gray)
    interest_points = find_interest_points(img_cpy)
    save_data(interest_points, "2_3_interest_points.pkl")
    interest_points = load_data("2_3_interest_points.pkl")
    tmp = np.zeros(img_cpy.shape)
    for x, y, sigma in interest_points:
        tmp[x, y] = 255
        img_cpy = cv2.circle(img_cpy, (y, x), 4, 0, thickness=2)
    save_image(tmp, "2_3_interest_points.jpg")
    save_image(img_cpy, "2_3_result.jpg")


def main() -> None:
    # part_a()

    # part_b_c()

    part_c()


if __name__ == '__main__':
    REPORT = False

    directory_setup()
    img_rgb = load_image("./resource/building.jpg", gray=False)
    img_gray = load_image("./resource/building.jpg", gray=True)

    print("{0} Question 2 {0}".format("=" * 20))
    main()
    print("{0} Done {0}".format("=" * 20))
