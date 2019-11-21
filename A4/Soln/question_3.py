import cv2
import matplotlib.pyplot as plt

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


# ==================== Part a ====================
def part_a() -> None:
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
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    kp_1, desc_1 = sift.detectAndCompute(gray_img_1, None)
    kp_2, desc_2 = sift.detectAndCompute(gray_img_2, None)
    kp_3, desc_3 = sift.detectAndCompute(gray_img_3, None)

    # Match img1 and img2.
    matches_12 = bf.match(desc_1, desc_2)
    matches_12 = sorted(matches_12, key=lambda x: x.distance)
    hard_code_indices_12 = [1, 2, 3, 4, 5, 6, 9, 10]
    print(hard_code_indices_12)
    hard_code_matches_12 = [matches_12[i] for i in hard_code_indices_12]
    print([match.distance for match in hard_code_matches_12])
    print(sorted([kp_1[match.queryIdx].pt for match in hard_code_matches_12],
                 key=lambda x: x[0]))
    match_12 = cv2.drawMatches(gray_img_1, kp_1, gray_img_2, kp_2,
                               hard_code_matches_12, None, flags=2)
    cv2.imwrite(OUT_DIR + "match_12.jpg", match_12)

    # Match img1 and img3.
    matches_13 = bf.match(desc_1, desc_3)
    matches_13 = sorted(matches_13, key=lambda x: x.distance)
    hard_code_indices_13 = [0, 2, 3, 5, 6, 9, 15, 20]
    print(hard_code_indices_13)
    hard_code_matches_13 = [matches_13[i] for i in hard_code_indices_13]
    print([match.distance for match in hard_code_matches_13])
    print(sorted([kp_1[match.queryIdx].pt for match in hard_code_matches_13],
                 key=lambda x: x[0]))
    match_13 = cv2.drawMatches(gray_img_1, kp_1, gray_img_3, kp_3,
                               hard_code_matches_13, None, flags=2)
    cv2.imwrite(OUT_DIR + "match_13.jpg", match_13)


if __name__ == '__main__':
    PLOT = False
    DEBUG = False
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

    # part_a()
