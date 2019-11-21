import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==================== Constants ====================
OPENCV_ENABLE_NONFREE = True
IMAGE_DIR = "./q2_images/"
LEFT_IMAGE_NAME = "000020_left.jpg"
RIGHT_IMAGE_NAME = "000020_right.jpg"
OUT_DIR = "./out/q2/"

# ==================== Parameters ====================
# bounding box defined in ./q2_images/000020.txt
bounding_box = [685.05, 181.43, 804.68, 258.21]
# parameters are defined in ./q2_images/000020_allcalib.txt
f = 721.537700
px = 609.559300
py = 172.854000
baseline = 0.5327119288
top_left = (int(bounding_box[0]), int(bounding_box[1]))
bottom_right = (int(bounding_box[2]) + 1, int(bounding_box[3]) + 1)

# ==================== Images ====================
left_img = cv2.imread(IMAGE_DIR + LEFT_IMAGE_NAME, cv2.IMREAD_COLOR)
right_img = cv2.imread(IMAGE_DIR + RIGHT_IMAGE_NAME, cv2.IMREAD_COLOR)
left_gray_img = cv2.imread(IMAGE_DIR + LEFT_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
right_gray_img = cv2.imread(IMAGE_DIR + RIGHT_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
left_rectangle_image = cv2.rectangle(left_img, top_left, bottom_right,
                                     (0, 255, 0), 2)


# ==================== Part a ====================
def SSD(l_patch: np.ndarray, r_patch: np.ndarray) -> float:
    """
    The Sum Squared Difference between the left patch and the right patch
    Note: When using SSD, look for minima

    :param l_patch: the left patch
    :param r_patch: the right patch
    :return: the sum squared difference
    """
    assert l_patch.shape == r_patch.shape, \
        f"The left shape {l_patch.shape} and right shape {r_patch.shape} are " \
        f"not equal"
    return float(np.sum(np.square(l_patch - r_patch), axis=None))


def NC(l_patch: np.ndarray, r_patch: np.ndarray) -> float:
    """
    The Normalized Coefficient between the left patch and the right patch
    Note: When using NC, look for maxima

    :param l_patch: the left patch
    :param r_patch: the right patch
    :return: the normalized coefficient
    """
    assert l_patch.shape == r_patch.shape, \
        f"The left shape {l_patch.shape} and right shape {r_patch.shape} are " \
        f"not equal"
    norm_prod = np.linalg.norm(l_patch) * np.linalg.norm(r_patch)
    if norm_prod == 0:
        return 0
    return float(np.sum(l_patch * r_patch) / norm_prod)


def part_a() -> None:
    """
    Compute the depth for each pixel in the given bounding box of car.

    Algorithm:
        - Given a left patch, compare it with all the patches on the right
          image’s scanline.
        - To reduce computation complexity, we use a small patch size, or sample
          patches (e.g. every other pixel) from scanline instead of comparing
          with all possible patches.

    Report:
        - patch size, sampling method, and matching cost function.
        - how depth is computed for each pixel.
        - visualize the depth information. report outliers from incorrect
          point correspondences?
    """
    print("{0} Part A Start {0}".format("=" * 20))

    step = 3
    patch_size = 5
    half_patch_size = patch_size // 2

    SSD_mask = np.zeros(left_gray_img.shape)
    SSD_depth = np.zeros(left_gray_img.shape)
    NC_mask = np.zeros(left_gray_img.shape)
    NC_depth = np.zeros(left_gray_img.shape)

    # for each pixel in the bounding box in the left image, find the
    # corresponding pixel in the right image along the scanline
    for x in range(top_left[1], bottom_right[1]):
        print(f"Processing row {x} / {bottom_right[1]}")
        for y in range(top_left[0], bottom_right[0]):
            l_patch = left_gray_img[
                      x - half_patch_size: x + half_patch_size + 1,
                      y - half_patch_size: y + half_patch_size + 1]

            # keep track of all computed cost/correlations along the scanline
            SSDs = np.full((right_gray_img.shape[1],), np.inf)
            NCs = np.full((right_gray_img.shape[1],), -np.inf)

            # only loop where r_patch's shape will match with that of l_patch
            for ry in range(half_patch_size,
                            min(y, right_gray_img.shape[1] - half_patch_size),
                            step):
                r_patch = right_gray_img[
                          x - half_patch_size: x + half_patch_size + 1,
                          ry - half_patch_size: ry + half_patch_size + 1]
                SSDs[ry] = SSD(l_patch, r_patch)
                NCs[ry] = NC(l_patch, r_patch)

            min_SSD, min_SSD_idx = np.min(SSDs), np.argmin(SSDs)
            max_NC, max_NC_idx = np.max(NCs), np.argmax(NCs)

            # Mark the corresponding location for debugging
            SSD_mask[x, min_SSD_idx] = 255
            NC_mask[x, max_NC_idx] = 255

            # calculate depth and map to depth matrix
            if y - min_SSD_idx == 0:
                SSD_depth[x, y] = 0
            else:
                SSD_depth[x, y] = f * baseline / (y - min_SSD_idx)
            if y - max_NC_idx == 0:
                NC_depth[x, y] = 0
            else:
                NC_depth[x, y] = f * baseline / (y - max_NC_idx)

            if DEBUG:
                print(f"SSD: {min_SSD}, SSD idx: {min_SSD_idx}, "
                      f"depth: {SSD_depth[x, y]}")
                print(f"NC: {max_NC}, NC idx: {max_NC_idx}, "
                      f"depth: {NC_depth[x, y]}")

    if PLOT:
        plt.clf()
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].imshow(SSD_mask, cmap='gray')
        axes[0, 1].imshow(NC_mask, cmap='gray')
        axes[1, 0].imshow(SSD_depth, cmap='gray')
        axes[1, 1].imshow(NC_depth, cmap='gray')
        plt.show()

    cv2.imwrite(OUT_DIR + f"{patch_size}_{step}_SSD_mask.jpg", SSD_mask)
    cv2.imwrite(OUT_DIR + f"{patch_size}_{step}_NC_mask.jpg", NC_mask)
    cv2.imwrite(OUT_DIR + f"{patch_size}_{step}_SSD_depth.jpg", SSD_depth)
    cv2.imwrite(OUT_DIR + f"{patch_size}_{step}_NC_depth.jpg", NC_depth)

    print("{0} Part A End {0}".format("=" * 20))


if __name__ == '__main__':
    PLOT = True
    DEBUG = True
    if PLOT and False:
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].imshow(left_img)
        axes[0, 1].imshow(right_img)
        axes[1, 0].imshow(left_gray_img, cmap='gray')
        axes[1, 1].imshow(right_gray_img, cmap='gray')

        plt.show()
        plt.clf()
        plt.imshow(left_rectangle_image)
        plt.show()
        plt.clf()

    part_a()
