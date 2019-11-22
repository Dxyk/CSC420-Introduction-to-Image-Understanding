import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==================== Constants ====================
OPENCV_ENABLE_NONFREE = True
IMAGE_DIR = "./q3_images/"
IMAGE_1_NAME = "img_1.jpg"
IMAGE_2_NAME = "img_2.jpg"
IMAGE_3_NAME = "img_3.jpg"
OUT_DIR = "./out/q3/"

# ==================== Images ====================
img_1 = cv2.imread(IMAGE_DIR + IMAGE_1_NAME, cv2.IMREAD_COLOR)
img_2 = cv2.imread(IMAGE_DIR + IMAGE_2_NAME, cv2.IMREAD_COLOR)
img_3 = cv2.imread(IMAGE_DIR + IMAGE_3_NAME, cv2.IMREAD_COLOR)
gray_img_1 = cv2.imread(IMAGE_DIR + IMAGE_1_NAME, cv2.IMREAD_GRAYSCALE)
gray_img_2 = cv2.imread(IMAGE_DIR + IMAGE_2_NAME, cv2.IMREAD_GRAYSCALE)
gray_img_3 = cv2.imread(IMAGE_DIR + IMAGE_3_NAME, cv2.IMREAD_GRAYSCALE)

# ==================== KP Matches ====================
# These results are equivalent to results generated from part a
kp_1_12 = np.array([[1574, 1654], [2710, 2123], [2345, 1956], [1165, 2805],
                    [1539, 2166], [2513, 2588], [1523, 2205], [1400, 2256]])
kp_2_12 = np.array([[836, 1678], [1901, 2132], [1583, 1978], [401, 2893],
                    [802, 2198], [1732, 2570], [784, 2237], [653, 2294]])
kp_1_13 = np.array([[1603, 2310], [1429, 2084], [1566, 2208], [1405, 1805],
                    [1577, 1729], [1442, 2250], [1613, 2964], [1156, 1748]])
kp_3_13 = np.array([[1206, 2334], [1039, 2113], [1170, 2234], [1016, 1841],
                    [1181, 1769], [1051, 2273], [698, 2948], [783, 1786]])


# ==================== Part a ====================
def part_a(save_image=False):
    """
    Use SIFT matching (or any other point matching technique) to find a number
    of point correspondences in the (I1, I2)​ image pair and in the (I1, I3)​
    image pair.

    Report:
        - Visualize the results.
        - If there are any outliers, either manually remove them or increase
          the matching threshold so no outliers remain.
        - Pick 8 point correspondences from the remaining set for each image
          pair, i.e. (I1, I2) and (​I1, I2).
        - Visualize those 8 point matches.
    """
    print("{0} Part A Start {0}".format("=" * 20))
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    kp_1, desc_1 = sift.detectAndCompute(gray_img_1, None)
    kp_2, desc_2 = sift.detectAndCompute(gray_img_2, None)
    kp_3, desc_3 = sift.detectAndCompute(gray_img_3, None)

    # Match img1 and img2.
    matches_12 = bf.match(desc_1, desc_2)
    matches_12 = sorted(matches_12, key=lambda x: x.distance)
    hard_code_indices_12 = [1, 2, 3, 4, 5, 6, 9, 10]
    hard_code_matches_12 = [matches_12[i] for i in hard_code_indices_12]
    if save_image:
        match_12 = cv2.drawMatches(gray_img_1, kp_1, gray_img_2, kp_2,
                                   matches_12[:20], None, flags=2)
        cv2.imwrite(OUT_DIR + "a_match_12.jpg", match_12)
        match_12 = cv2.drawMatches(gray_img_1, kp_1, gray_img_2, kp_2,
                                   hard_code_matches_12, None, flags=2)
        cv2.imwrite(OUT_DIR + "a_match_12_8_points.jpg", match_12)

    # Match img1 and img3.
    matches_13 = bf.match(desc_1, desc_3)
    matches_13 = sorted(matches_13, key=lambda x: x.distance)
    hard_code_indices_13 = [0, 2, 3, 5, 6, 9, 15, 20]
    hard_code_matches_13 = [matches_13[i] for i in hard_code_indices_13]
    if save_image:
        match_13 = cv2.drawMatches(gray_img_1, kp_1, gray_img_3, kp_3,
                                   matches_13[:20], None, flags=2)
        cv2.imwrite(OUT_DIR + "a_match_13.jpg", match_13)
        match_13 = cv2.drawMatches(gray_img_1, kp_1, gray_img_3, kp_3,
                                   hard_code_matches_13, None, flags=2)
        cv2.imwrite(OUT_DIR + "a_match_13_8_points.jpg", match_13)

    kp_1_12_coords = []
    kp_2_12_coords = []
    kp_1_13_coords = []
    kp_3_13_coords = []

    for i, m in enumerate(hard_code_matches_12):
        kp_1_12_coords.append(kp_1[m.queryIdx].pt)
        kp_2_12_coords.append(kp_2[m.trainIdx].pt)

    for i, m in enumerate(hard_code_matches_13):
        kp_1_13_coords.append(kp_1[m.queryIdx].pt)
        kp_3_13_coords.append(kp_3[m.trainIdx].pt)

    kp_1_12_coords = np.int32(kp_1_12_coords)
    kp_2_12_coords = np.int32(kp_2_12_coords)
    kp_1_13_coords = np.int32(kp_1_13_coords)
    kp_3_13_coords = np.int32(kp_3_13_coords)

    for coords in [kp_1_12_coords, kp_2_12_coords, kp_1_13_coords,
                   kp_3_13_coords]:
        print(coords)

    print("{0} Part A End {0}".format("=" * 20))
    return kp_1_12_coords, kp_2_12_coords, kp_1_13_coords, kp_3_13_coords


# ==================== Part b ====================
def part_b():
    """
    Using standard 8-point algorithm, calculate:
    - the fundamental matrix F12 for image pair (I1, I2).
    - the fundamental matrix F13 for image pair (I1, I3).
    """
    print("{0} Part B Start {0}".format("=" * 20))
    A_12 = np.ones((8, 9))
    for i in range(len(kp_1_12)):
        x_l, y_l = kp_1_12[i]
        x_r, y_r = kp_2_12[i]
        A_12[i] = np.array([
            x_r * x_l, x_r * y_l, x_r,
            y_r * x_l, y_r * y_l, y_r,
            x_l, y_l, 1
        ])
    u, s, vh = np.linalg.svd(A_12)
    f_12 = vh[-1, :]
    F_12 = f_12.reshape((3, 3))

    u, s, vh = np.linalg.svd(F_12, full_matrices=False)
    s[-1] = 0
    F_12 = np.matmul(np.matmul(u, np.diag(s)), vh)

    A_13 = np.ones((8, 9))
    for i in range(len(kp_1_13)):
        x_l, y_l = kp_1_13[i]
        x_r, y_r = kp_3_13[i]
        A_13[i] = np.array([
            x_r * x_l, x_r * y_l, x_r,
            y_r * x_l, y_r * y_l, y_r,
            x_l, y_l, 1
        ])
    u, s, vh = np.linalg.svd(A_13)
    f_13 = vh[-1, :]
    F_13 = f_13.reshape((3, 3))

    u, s, vh = np.linalg.svd(F_13)
    F_13 = np.matmul(np.matmul(u, np.diag(s)), vh)

    if DEBUG:
        print("F_12:\n", F_12)
        print("F_13:\n", F_13)

    print("{0} Part B End {0}".format("=" * 20))
    return F_12, F_13


# ==================== Part c ====================
def draw_lines(img, lines, pts1, pts2):
    """
    img1 - image on which we draw the epipolar lines for the points in img2
    lines - corresponding epipolar lines
    """
    r, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv2.line(img, (x0, y0), (x1, y1), color=color, thickness=2)
        img = cv2.circle(img, tuple(pt1), 5, color, -1)
    return img


def part_c(F_12: np.ndarray, save_image: bool = False):
    """
    Using F12, calculate the epipolar lines in the right image for each of the 8
    points in the left image and plot them on the right image.
    """
    print("{0} Part C Start {0}".format("=" * 20))

    lines = cv2.computeCorrespondEpilines(kp_2_12, 2, F_12)
    lines = lines.reshape(-1, 3)
    img_2_with_lines = draw_lines(gray_img_2, lines, kp_1_12, kp_2_12)

    if PLOT:
        plt.imshow(img_2_with_lines)
        plt.show()
        plt.clf()

    if save_image:
        cv2.imwrite(OUT_DIR + "c_right_line.jpg", img_2_with_lines)

    print("{0} Part C End {0}".format("=" * 20))


# ==================== Part d ====================
def part_d(F_12: np.ndarray, F_13: np.ndarray, save_image: bool = True):
    """
    - Using F12, rectify I2 with I1 and visualize the resulting image
      side by side with I1.
    - Do the same for I3 using F13.
    """
    print("{0} Part D Start {0}".format("=" * 20))

    # warp I_2 with I_1
    ret, H_l_12, H_r_12 = cv2.stereoRectifyUncalibrated(
        kp_1_12, kp_2_12, F_12, gray_img_2.shape)
    if DEBUG:
        print("ret\n", ret)
        print("H_l_12\n", H_l_12)
        print("H_r_12\n", H_r_12)
    warped_img_2 = cv2.warpPerspective(gray_img_2, H_r_12, gray_img_2.shape)

    # warp I_3 with I_1
    ret, H_l_13, H_r_13 = cv2.stereoRectifyUncalibrated(
        kp_1_13, kp_3_13, F_13, gray_img_3.shape)
    if DEBUG:
        print("ret\n", ret)
        print("H_l_13\n", H_l_13)
        print("H_r_13\n", H_r_13)
    warped_img_3 = cv2.warpPerspective(gray_img_3, H_r_13, gray_img_3.shape)

    if PLOT:
        plt.imshow(warped_img_2, cmap='gray')
        plt.show()
        plt.clf()
        plt.imshow(warped_img_3, cmap='gray')
        plt.show()
        plt.clf()

    if save_image:
        cv2.imwrite(OUT_DIR + "d_img_2_warped.jpg", warped_img_2)
        cv2.imwrite(OUT_DIR + "d_img_3_warped.jpg", warped_img_3)

    print("{0} Part D End {0}".format("=" * 20))


# ==================== Part e ====================
def part_e():
    """
    ​Using OpenCV, compute F’12 and F’13 and compare with your results.
    """
    print("{0} Part E Start {0}".format("=" * 20))
    F_12_cv, mask = cv2.findFundamentalMat(kp_1_12, kp_2_12, cv2.FM_7POINT)
    F_13_cv, mask = cv2.findFundamentalMat(kp_1_13, kp_3_13, cv2.FM_7POINT)

    if DEBUG:
        print("F_12_CV:\n", F_12_cv)
        print("F_13_CV:\n", F_13_cv)

    print("{0} Part E End {0}".format("=" * 20))
    return F_12_cv, F_13_cv


# ==================== Part f ====================
def part_f(F_12_cv, F_13_cv, save_image=True):
    """
    Using OpenCV, rectify the images using F’12 and F’13 and compare with your
    rectifications (part d).
    """
    print("{0} Part F Start {0}".format("=" * 20))
    ret, H_l_12, H_r_12 = cv2.stereoRectifyUncalibrated(
        kp_1_12, kp_2_12, F_12_cv, gray_img_2.shape)
    if DEBUG:
        print("ret\n", ret)
        print("H_l_12\n", H_l_12)
        print("H_r_12\n", H_r_12)
    warped_img_2 = cv2.warpPerspective(gray_img_2, H_r_12, gray_img_2.shape)

    # warp I_3 with I_1
    ret, H_l_13, H_r_13 = cv2.stereoRectifyUncalibrated(
        kp_1_13, kp_3_13, F_13_cv, gray_img_3.shape)
    if DEBUG:
        print("ret\n", ret)
        print("H_l_13\n", H_l_13)
        print("H_r_13\n", H_r_13)
    warped_img_3 = cv2.warpPerspective(gray_img_3, H_r_13, gray_img_3.shape)

    if PLOT:
        plt.imshow(warped_img_2, cmap='gray')
        plt.show()
        plt.clf()
        plt.imshow(warped_img_3, cmap='gray')
        plt.show()
        plt.clf()

    if save_image:
        cv2.imwrite(OUT_DIR + "f_img_2_warped.jpg", warped_img_2)
        cv2.imwrite(OUT_DIR + "f_img_3_warped.jpg", warped_img_3)

    print("{0} Part F End {0}".format("=" * 20))


if __name__ == '__main__':
    PLOT = False
    DEBUG = True
    if PLOT:
        fig, axes = plt.subplots(nrows=2, ncols=3)
        axes[0, 0].imshow(img_1)
        axes[0, 1].imshow(img_2)
        axes[0, 2].imshow(img_3)
        axes[1, 0].imshow(gray_img_1, cmap='gray')
        axes[1, 1].imshow(gray_img_2, cmap='gray')
        axes[1, 2].imshow(gray_img_3, cmap='gray')
        plt.show()
        plt.clf()

    part_a(save_image=True)

    # F_12, F_13 = part_b()
    # pt2 = np.hstack([kp_2_12[0], 1])
    # pt1 = np.hstack([kp_1_12[0], 1])
    # print(np.matmul(np.matmul(pt2.T, F_12), pt1))

    # part_c(F_12, save_image=True)
    #
    # part_d(F_12, F_13, save_image=True)
    #
    # F_12_cv, F_13_cv = part_e()
    # print(np.matmul(np.matmul(pt2.T, F_12_cv), pt1))

    # part_f(F_12_cv, F_13_cv, save_image=True)
    #
    # if DEBUG:
    #     print(f"F_12 vs F_12_cv:\n {F_12}\n {F_12_cv}")
    #     print(f"F_13 vs F_13_cv:\n {F_13}\n {F_13_cv}")
