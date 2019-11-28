import cv2
import numpy as np

# ==================== Constants ====================
# IMAGE_DIR = "./q3_images/"
IMAGE_DIR = "./q3_sample_images/"
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
# kp_1_12 = np.array([[1574, 1654], [2710, 2123], [2345, 1956], [1165, 2805],
#                     [1539, 2166], [2513, 2588], [1523, 2205], [1400, 2256]])
# kp_2_12 = np.array([[836, 1678], [1901, 2132], [1583, 1978], [401, 2893],
#                     [802, 2198], [1732, 2570], [784, 2237], [653, 2294]])
# kp_1_13 = np.array([[1603, 2310], [1429, 2084], [1566, 2208], [1405, 1805],
#                     [1577, 1729], [1442, 2250], [1613, 2964], [1156, 1748]])
# kp_3_13 = np.array([[1206, 2334], [1039, 2113], [1170, 2234], [1016, 1841],
#                     [1181, 1769], [1051, 2273], [698, 2948], [783, 1786]])

hard_code_indices_12 = [0, 1, 2, 4, 6, 9, 11, 13]
kp_1_12 = np.array([(2881.928955078125, 1621.4644775390625),
                    (2223.441650390625, 861.7791137695312),
                    (1774.7781982421875, 1792.964111328125),
                    (1937.4525146484375, 748.00927734375),
                    (3897.219482421875, 1309.3978271484375),
                    (1600.680419921875, 1814.0863037109375),
                    (1338.01611328125, 836.5589599609375),
                    (1392.722900390625, 2548.712646484375)])
kp_2_12 = np.array([(2848.749755859375, 1200.1876220703125),
                    (1920.167236328125, 829.8592529296875),
                    (1979.800537109375, 1884.9825439453125),
                    (1619.34228515625, 870.6213989257812),
                    (3589.759521484375, 436.66790771484375),
                    (1842.66015625, 1985.69384765625),
                    (1144.5994873046875, 1207.8626708984375),
                    (2014.410400390625, 2732.1103515625)])

# hard_code_indices_13 = [6, 19, 29, 54, 55, 56, 57, 66, 71, 78, 93, 97]
# possibles = [1, 6, 11, 19, 21, 28, 29, 40, 54, 55, 56, 57, 66, 71, 78, 79,
# 81, 93, 98, 99]
hard_code_indices_13 = [19, 29, 54, 55, 57, 66, 93, 97]
kp_1_13 = np.array([(1247.521240234375, 660.3178100585938),
                    (2150.78955078125, 647.035888671875),
                    (2002.5169677734375, 633.85205078125),
                    (1093.086181640625, 1240.29296875),
                    (1182.396240234375, 775.0768432617188),
                    (2078.02734375, 1037.068359375),
                    (1485.2257080078125, 1482.1685791015625),
                    (1508.6966552734375, 825.9324340820312)])
kp_3_13 = np.array([(1881.14306640625, 616.7504272460938),
                    (2336.680419921875, 628.4423828125),
                    (2212.57177734375, 607.0022583007812),
                    (1186.9083251953125, 1032.3505859375),
                    (1826.0133056640625, 716.88916015625),
                    (2286.837158203125, 1003.905517578125),
                    (1484.8006591796875, 1291.427978515625),
                    (2088.74853515625, 787.6981201171875)])


# ==================== Part a ====================
def detect_and_compute(image_1, image_2):
    # contrastThreshold = 0.04
    # edgeThreshold = 10
    # sigma = 1.6
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    kp_1, desc_1 = sift.detectAndCompute(image_1, None)
    kp_2, desc_2 = sift.detectAndCompute(image_2, None)
    matches = bf.match(desc_1, desc_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp_1, kp_2, matches


def part_a(img1, img2, indices, file_name):
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
    # Match img1 and img2.
    kp_1, kp_2, matches = detect_and_compute(img1, img2)
    hard_code_matches = [matches[i] for i in indices]
    # test code to manually pick matching points
    # for idx in range(0, 20):
    #     matched = cv2.drawMatches(img1, kp_1, img2, kp_2, [matches[idx]],
    #                               None, flags=2)
    #     cv2.imwrite(OUT_DIR + f"a_test{idx}.jpg", matched)
    # all matches
    matched = cv2.drawMatches(img1, kp_1, img2, kp_2, matches[:20], None,
                              flags=2)
    cv2.imwrite(OUT_DIR + f"a_{file_name}.jpg", matched)
    # hard-coded matches
    matched = cv2.drawMatches(img1, kp_1, img2, kp_2, hard_code_matches, None,
                              flags=2)
    cv2.imwrite(OUT_DIR + f"a_{file_name}_8_points.jpg", matched)

    kp_1_coords = np.array([kp_1[m.queryIdx].pt for m in hard_code_matches])
    kp_2_coords = np.array([kp_2[m.trainIdx].pt for m in hard_code_matches])
    # kp_1_coords = np.int32(kp_1_coords)
    # kp_2_coords = np.int32(kp_2_coords)

    print(kp_1_coords)
    print(kp_2_coords)

    return kp_1_coords, kp_2_coords


# ==================== Part b ====================
def part_b(l_pts, r_pts, full_matrix=True):
    """
    Using standard 8-point algorithm, calculate:
    - the fundamental matrix F12 for image pair (I1, I2).
    - the fundamental matrix F13 for image pair (I1, I3).
    """
    A = np.ones((8, 9))
    for i in range(len(l_pts)):
        x_l, y_l = l_pts[i]
        x_r, y_r = r_pts[i]
        A[i] = np.array([
            x_r * x_l, x_r * y_l, x_r,
            y_r * x_l, y_r * y_l, y_r,
            x_l, y_l, 1
        ])
    u, s, vh = np.linalg.svd(A, full_matrices=full_matrix)
    f = vh[-1]
    F = f.reshape((3, 3))

    u, s, vh = np.linalg.svd(F, full_matrices=full_matrix)
    s[2] = 0
    s = np.diag(s)
    F = np.matmul(np.matmul(u, s), vh)
    return F / F[2, 2]


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


def part_c(img_r: np.ndarray, kp_l: np.ndarray, kp_r: np.ndarray,
           F: np.ndarray, file_name: str):
    """
    Using F12, calculate the epipolar lines in the right image for each of the 8
    points in the left image and plot them on the right image.
    """
    lines = cv2.computeCorrespondEpilines(kp_l.reshape((-1, 1, 2)), 1, F)
    lines = lines.reshape(-1, 3)
    img_r_with_lines = draw_lines(img_r, lines, np.int32(kp_r), np.int32(kp_l))
    cv2.imwrite(OUT_DIR + f"c_{file_name}_line.jpg", img_r_with_lines)

    # lines = cv2.computeCorrespondEpilines(kp_l, 1, F)
    # lines = lines.reshape(-1, 3)
    # img_2_with_lines = draw_lines(img_r, lines,
    #                               np.int32(kp_r), np.int32(kp_l))
    # cv2.imwrite(OUT_DIR + f"c_{file_name}_line.jpg", img_2_with_lines)
    return lines


# ==================== Part d ====================
def part_d(r_image, l_kp, r_kp, F, file_name):
    """
    - Using F12, rectify I2 with I1 and visualize the resulting image
      side by side with I1.
    - Do the same for I3 using F13.
    """
    ret, H_l, H_r = cv2.stereoRectifyUncalibrated(l_kp, r_kp, F, r_image.shape)
    h, w = r_image.shape
    t1 = np.array([1, 0, 800, 0, 1, 1500, 0, 0, 1]).reshape((3, 3))
    # t2 = np.array([1, 0, -100, 0, 1, -100, 0, 0, 1]).reshape((3, 3))
    H = t1.dot(H_r)

    warped_img = cv2.warpPerspective(r_image, H, (h + 1500, w + 1500))

    cv2.imwrite(OUT_DIR + f"{file_name}_warped.jpg", warped_img)

    return ret, H_l, H_r


# ==================== Part e ====================
def part_e(l_kps, r_kps):
    """
    ​Using OpenCV, compute F’12 and F’13 and compare with your results.
    """
    F, mask = cv2.findFundamentalMat(l_kps, r_kps, cv2.FM_8POINT)
    return F


# ==================== Part f ====================
def part_f(r_image, l_kp, r_kp, F, file_name):
    """
    Using OpenCV, rectify the images using F’12 and F’13 and compare with your
    rectifications (part d).
    """
    return part_d(r_image, l_kp, r_kp, F, file_name)


if __name__ == '__main__':
    PLOT = False
    DEBUG = True
    # if PLOT:
    #     fig, axes = plt.subplots(nrows=2, ncols=3)
    #     axes[0, 0].imshow(img_1)
    #     axes[0, 1].imshow(img_2)
    #     axes[0, 2].imshow(img_3)
    #     axes[1, 0].imshow(gray_img_1, cmap='gray')
    #     axes[1, 1].imshow(gray_img_2, cmap='gray')
    #     axes[1, 2].imshow(gray_img_3, cmap='gray')
    #     plt.show()
    #     plt.clf()

    # Part a
    # print("{0} Part A Start {0}".format("=" * 20))
    # print("{0} Part A 1-2 {0}".format("=" * 10))
    # part_a(gray_img_1, gray_img_2, hard_code_indices_12, "match_12")
    # print("{0} Part A 1-3 {0}".format("=" * 10))
    # part_a(gray_img_1, gray_img_3, hard_code_indices_13, "match_13")
    # print("{0} Part A End {0}".format("=" * 20))

    # Part b
    print("{0} Part B Start {0}".format("=" * 20))
    print("{0} Part B 1-2 {0}".format("=" * 10))
    F_12 = part_b(kp_1_12, kp_2_12, full_matrix=True)
    # verify that all points multiplication close to 0
    for i in range(len(kp_1_12)):
        pt_l = np.hstack([kp_1_12[i], 1])
        pt_r = np.hstack([kp_2_12[i], 1])
        print(np.matmul(np.matmul(pt_r.T, F_12), pt_l))
    print("{0} Part B 1-3 {0}".format("=" * 10))
    F_13 = part_b(kp_1_13, kp_3_13, full_matrix=False)
    for i in range(len(kp_1_13)):
        pt_l = np.hstack([kp_1_13[i], 1])
        pt_r = np.hstack([kp_3_13[i], 1])
        print(np.matmul(np.matmul(pt_r.T, F_13), pt_l))
    print("{0} Part B End {0}".format("=" * 20))

    # Part c
    print("{0} Part C Start {0}".format("=" * 20))
    part_c(gray_img_2, kp_1_12, kp_2_12, F_12, "image_2")
    print("{0} Part C End {0}".format("=" * 20))

    # Part d
    print("{0} Part D Start {0}".format("=" * 20))
    print("{0} Part D 1-2 {0}".format("=" * 10))
    part_d(gray_img_2, kp_1_12, kp_2_12, F_12, "d_img_2")
    print("{0} Part D 1-3 {0}".format("=" * 10))
    part_d(gray_img_3, kp_1_13, kp_3_13, F_13, "d_img_3")
    print("{0} Part D Start {0}".format("=" * 20))

    # Part e
    print("{0} Part E Start {0}".format("=" * 20))
    print("{0} Part E 1-2 {0}".format("=" * 10))
    F_12_cv = part_e(kp_1_12, kp_2_12)
    # for i in range(len(kp_1_13)):
    #     pt_l = np.hstack([kp_1_12[i], 1])
    #     pt_r = np.hstack([kp_2_12[i], 1])
    #     print(np.matmul(np.matmul(pt_r.T, F_12_cv), pt_l))
    print("{0} Part E 1-3 {0}".format("=" * 10))
    F_13_cv = part_e(kp_1_13, kp_3_13)
    # for i in range(len(kp_1_13)):
    #     pt_l = np.hstack([kp_1_13[i], 1])
    #     pt_r = np.hstack([kp_3_13[i], 1])
    #     print(np.matmul(np.matmul(pt_r.T, F_13_cv), pt_l))
    print(f"F_12:\n{F_12}\nF_12_cv:\n{F_12_cv}")
    print(f"F_13:\n{F_13}\nF_13_cv:\n{F_13_cv}")
    print("{0} Part E End {0}".format("=" * 20))

    # Part f
    print("{0} Part F Start {0}".format("=" * 20))
    print("{0} Part F 1-2 {0}".format("=" * 10))
    part_f(gray_img_2, kp_1_12, kp_2_12, F_12_cv, "f_img_2")
    print("{0} Part F 1-3 {0}".format("=" * 10))
    part_f(gray_img_3, kp_1_13, kp_3_13, F_13_cv, "f_img_3")
    print("{0} Part F Start {0}".format("=" * 20))

    part_c(gray_img_2, kp_1_12, kp_2_12, F_12_cv, "image_2_cv")

