import pickle
from typing import Any, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "./out/question_4/"
TUNING_PATH = OUT_PATH + "tuning/"
REPORT_PATH = "./Report/images/question_4/"
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
    from pathlib import Path

    if Path(PICKLE_PATH + src_path).exists():
        with open(PICKLE_PATH + src_path, 'rb') as f:
            return pickle.load(f)
    return None


# ==================== Helper Methods End ====================


# ==================== Part 1 ====================
def get_sift_kp_des(img: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Calculate the sift key points and features

    :param img: the input image
    :return: the key points and the features
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    kp, des = sift.compute(img, kp)
    return kp, des


# ==================== Part b ====================
def part_b(img1: np.ndarray, img2: np.ndarray, question: int = 2) -> None:
    # sorted_matches = load_data(
    #     "4_{}_matches_L{}.pkl".format(question, norm_type))
    # if load and sorted_matches:
    #     return sorted_matches

    img1_kp, img1_des = get_sift_kp_des(img1)
    img2_kp, img2_des = get_sift_kp_des(img2)

    ratios = [r / 10 for r in range(1, 10)]

    l1_matches = {ratio: [] for ratio in ratios}
    l2_matches = {ratio: [] for ratio in ratios}
    l3_matches = {ratio: [] for ratio in ratios}

    ratio_match_count = {ratio: 0 for ratio in ratios}

    # all_matches = []

    # brute force loop through each pixel in sample1
    for i, (kp_1, des_1) in enumerate(zip(img1_kp, img1_des)):
        if i % 100 == 0:
            print("Progress: {0}%\t{1} / {2}".format(i / len(img1_kp), i,
                                                     len(img1_kp)))

        # calculate the diffs between sample2 pixels and the curr sample1 pixel
        diff = np.subtract(img2_des, des_1)

        # we calculate dists for different order (L1 - L3)
        for order in [1, 2, 3]:
            dist = np.linalg.norm(diff, axis=1, ord=order)
            # dist = np.zeros(diff.shape)
            # for j in range(diff.shape[0]):
            #     dist[j] = np.linalg.norm(diff[j], ord=order)

            # get the mins after calculation
            first_min = np.amin(dist)
            min_idx = np.where(dist == first_min)[0][0]
            kp_2 = img2_kp[min_idx]
            second_min = np.amin(dist[dist != first_min])
            curr_ratio = first_min / second_min

            # figure out which dictionary to use
            if order == 1:
                curr_matches = l1_matches
            elif order == 2:
                curr_matches = l2_matches
            else:
                curr_matches = l3_matches

            # add match to dictionary and update counter if the threshold is met
            for ratio in ratios:
                if curr_ratio <= ratio:
                    curr_matches[ratio].append((curr_ratio, kp_1.pt, kp_2.pt))
                    if order == 2:
                        ratio_match_count[ratio] += 1
                    # all_matches.append((curr_ratio, kp_1.pt, img2_kp[min_idx].pt))
                else:
                    break
    # sorted_matches = sorted(all_matches, key=lambda x: x[0])
    #
    # if save:
    #     save_data(sorted_matches,
    #               "4_{}_matches_L{}.pkl".format(question, norm_type))

    l1_10_best = sorted(l1_matches[0.8], key=lambda x: x[0])[:10]
    l2_10_best = sorted(l2_matches[0.8], key=lambda x: x[0])[:10]
    l3_10_best = sorted(l3_matches[0.8], key=lambda x: x[0])[:10]

    # best_10 = sorted_matches[:10]
    # print(len(best_10))

    # mark the top keypoints on the images
    for i in [1, 2, 3]:
        if i == 1:
            best_10 = l1_10_best
        elif i == 2:
            best_10 = l2_10_best
        else:
            best_10 = l3_10_best

        marked_img1 = np.copy(img1)
        marked_img2 = np.copy(img2)
        for match in best_10:
            print(match)
            point_1 = (int(match[1][0]), int(match[1][1]))
            point_2 = (int(match[2][0]), int(match[2][1]))
            marked_img1 = cv2.circle(marked_img1, point_1, 7, 255, -1)
            marked_img2 = cv2.circle(marked_img2, point_2, 7, 255, -1)

        save_image(marked_img1,
                   "4_{}_marked_sample_1_L{}.jpg".format(question, i))
        save_image(marked_img2,
                   "4_{}_marked_sample_2_L{}.jpg".format(question, i))

    # plot the ratio matches count
    fig = plt.figure()
    matches = []
    for ratio in ratios:
        matches.append(ratio_match_count[ratio])
    plt.plot(ratios, matches)
    fig.suptitle('Match count against Threshold', fontsize=20)
    plt.xlabel('Threshold', fontsize=18)
    plt.ylabel('Number of matches', fontsize=16)
    # plt.show()
    plt.savefig(
        OUT_PATH + "4_{}_match_vs_threshold.png".format(question))

    # return sorted_matches


def plot_matches_against_threshold(use_existing_data: bool = False,
                                   norm: int = 2, question: int = 2) -> None:
    """
    Plots the match counts against the thresholds
    :param use_existing_data: true if load from pickle
    :param norm: the norm
    :param question: the question number
    :return: the question number
    """
    ratios = np.arange(0.1, 1.0, 0.1)
    matches = load_data(
        "4_{}_match_against_threshold_L{}.pkl".format(question, norm))
    if not use_existing_data or not matches:
        sample1_des = load_data("sample1_des.pkl")
        sample2_des = load_data("sample2_des.pkl")
        matches = [0] * len(ratios)
        for i, des_1 in enumerate(sample1_des):
            if i % 100 == 0:
                print("Progress: {0}%\t{1} / {2}".format(i / len(sample1_des),
                                                         i, len(sample1_des)))
            diff = np.subtract(sample2_des, des_1)
            dist = np.zeros(diff.shape)
            for i in range(diff.shape[0]):
                dist[i] = np.linalg.norm(diff[i])
            first_min = np.amin(dist)
            second_min = np.amin(dist[dist != first_min])
            curr_ratio = first_min / second_min

            for i in range(len(ratios)):
                if curr_ratio <= ratios[i]:
                    matches[i] += 1

        save_data(matches,
                  "4_{}_match_against_threshold_L{}.pkl".format(question, norm))

    fig = plt.figure()
    plt.plot(ratios.tolist(), matches)
    fig.suptitle('Match count against Threshold', fontsize=20)
    plt.xlabel('Threshold', fontsize=18)
    plt.ylabel('Number of matches', fontsize=16)
    # plt.show()
    plt.savefig(
        OUT_PATH + "4_{}_match_vs_threshold_L{}.png".format(question, norm))
    return matches


# ==================== Part e ====================
def get_dominant_color_channel(img):
    red_sum = np.sum(img[:, :, 2])
    green_sum = np.sum(img[:, :, 1])
    blue_sum = np.sum(img[:, :, 0])
    max_sum = max([red_sum, green_sum, blue_sum])
    if red_sum == max_sum:
        return 2
    elif green_sum == max_sum:
        return 1
    elif blue_sum == max_sum:
        return 0


def get_sorted_match_e(colour_template, colour_search, ratio=0.8):
    sorted_matches = load_data("4_5_matches.pkl")
    if sorted_matches:
        return sorted_matches

    ct_main = colour_template[:, :, get_dominant_color_channel(colour_template)]
    ct_kp, ct_des = get_sift_kp_des(ct_main)

    cs_main = colour_search[:, :, get_dominant_color_channel(colour_search)]
    cs_kp, cs_des = get_sift_kp_des(cs_main)

    all_matches = []

    for i, (kp_1, des_1) in enumerate(zip(ct_kp, ct_des)):
        if i % 100 == 0:
            print("Progress: {0}%\t{1} / {2}".format(i / len(ct_kp), i,
                                                     len(ct_kp)))
        diff = np.subtract(cs_des, des_1)
        dist = np.linalg.norm(diff, axis=1)
        print(dist.shape)
        # dist = np.zeros(diff.shape)
        # for j in range(diff.shape[0]):
        #     dist[j] = np.linalg.norm(diff[j])
        first_min = np.amin(dist)
        min_idx = np.where(dist == first_min)[0][0]
        second_min = np.amin(dist[dist != first_min])
        curr_ratio = first_min / second_min
        if curr_ratio <= ratio:
            all_matches.append((curr_ratio, kp_1.pt, cs_kp[min_idx].pt))
    sorted_matches = sorted(all_matches, key=lambda x: x[0])

    save_data(sorted_matches, "4_5_matches.pkl")

    return sorted_matches


# ==================== main ====================
def part_a(img1: np.ndarray, img2: np.ndarray, question: int = 1) -> None:
    draw_flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # sample 1
    sample1_copy = np.copy(img1)
    sample1_kp, sample1_des = get_sift_kp_des(sample1_copy)
    save_data(sample1_des, "4_{}_sample1_des.pkl".format(question))
    sample1_copy = cv2.drawKeypoints(sample1_copy, sample1_kp, None,
                                     flags=draw_flag)
    save_image(sample1_copy, "4_{}_sample1_img.jpg".format(question))

    # sample 2
    sample2_copy = np.copy(img2)
    sample2_kp, sample2_des = get_sift_kp_des(sample2_copy)
    save_data(sample2_des, "4_{}_sample2_des.pkl".format(question))
    sample2_copy = cv2.drawKeypoints(sample2_copy, sample2_kp, None,
                                     flags=draw_flag)
    save_image(sample2_copy, "4_{}_sample2_img.jpg".format(question))


# def part_b(img1: np.ndarray, img2: np.ndarray, norm: int = 2,
#            question: int = 2) -> None:
#     # matching
#     sorted_matches = get_sorted_matches(norm_type=norm, question=question,
#                                         load=False)
#     best_10 = sorted_matches[:10]
#     print(len(best_10))
#     marked_sample1 = np.copy(img1)
#     marked_sample2 = np.copy(img2)
#     for match in best_10:
#         print(match)
#         point_1 = (int(match[1][0]), int(match[1][1]))
#         point_2 = (int(match[2][0]), int(match[2][1]))
#         marked_sample1 = cv2.circle(marked_sample1, point_1, 5, 255, -1)
#         marked_sample2 = cv2.circle(marked_sample2, point_2, 5, 255, -1)
#
#     save_image(marked_sample1,
#                "4_{}_marked_sample_1_L{}.jpg".format(question, norm))
#     save_image(marked_sample2,
#                "4_{}_marked_sample_2_L{}.jpg".format(question, norm))
#
#     # threshold tuning
#     plot_matches_against_threshold(use_existing_data=False, norm=norm,
#                                    question=question)


# def part_c():
#     for norm in [1, 3]:
#         part_b(sample1_gray, sample2_gray, norm=norm, question=3)


def part_d():
    normalized_sample1 = np.copy(sample1_gray) / 255.0
    normalized_sample2 = np.copy(sample2_gray) / 255.0

    noise1 = np.random.normal(0, 0.08, normalized_sample1.shape)
    noise2 = np.random.normal(0, 0.08, normalized_sample2.shape)

    noisy_sample1 = normalized_sample1 + noise1
    np.clip(noisy_sample1, 0, 1)

    noisy_sample2 = normalized_sample2 + noise2
    np.clip(noisy_sample2, 0, 1)

    noisy_sample1 = np.uint8(noisy_sample1 * 255)
    noisy_sample2 = np.uint8(noisy_sample2 * 255)

    part_a(noisy_sample1, noisy_sample2, question=4)
    part_b(noisy_sample1, noisy_sample2, question=4)


def part_e(img1, img2):
    sorted_matches = get_sorted_match_e(img1, img2)

    best_10 = sorted_matches[:10]
    marked_sample1 = np.copy(img1)
    marked_sample2 = np.copy(img2)
    for match in best_10:
        print(match)
        point_1 = (int(match[1][0]), int(match[1][1]))
        point_2 = (int(match[2][0]), int(match[2][1]))
        marked_sample1 = cv2.circle(marked_sample1, point_1, 5, 255, -1)
        marked_sample2 = cv2.circle(marked_sample2, point_2, 5, 255, -1)

    save_image(marked_sample1, "4_5_marked_ct.png")
    save_image(marked_sample2, "4_5_marked_cs.png")

    # threshold tuning
    # plot_matches_against_threshold(use_existing_data=True, norm=2,
    #                                question=5)


def main() -> None:
    # print("{0} a {0}".format("=" * 15))
    # part_a(sample1_gray, sample2_gray)

    print("{0} b {0}".format("=" * 15))
    part_b(sample1_gray, sample2_gray)

    # print("{0} c {0}".format("=" * 15))
    # part_c()

    # print("{0} d {0}".format("=" * 15))
    # part_d()
    #
    # print("{0} e {0}".format("=" * 15))
    # part_e(colour_template, colour_search)


if __name__ == '__main__':
    REPORT = False

    directory_setup()
    sample1_gray = load_image("./resource/sample1.jpg", gray=True)
    sample2_gray = load_image("./resource/sample2.jpg", gray=True)

    colour_template = load_image("./resource/colourTemplate.png")
    colour_search = load_image("./resource/colourSearch.png")

    print("{0} Question 4 {0}".format("=" * 20))
    main()
    print("{0} Done {0}".format("=" * 20))
