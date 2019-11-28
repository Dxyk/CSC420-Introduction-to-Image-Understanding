import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==================== Constants ====================
IMAGE_DIR = "./q2_images/"
LEFT_IMAGE_NAME = "000020_left.jpg"
RIGHT_IMAGE_NAME = "000020_right.jpg"
OUT_DIR = "./out/q2/"
BOUNDING_BOX_FILE_NAME = "000020.txt"
ALL_CALIB_FILE_NAME = "000020_allcalib.txt"

# ==================== Parameters ====================
with open(IMAGE_DIR + BOUNDING_BOX_FILE_NAME) as file:
    x_min, y_min, x_max, y_max = [float(i) for i in file.read().split()[1:]]
# parameters are defined in ./q2_images/000020_allcalib.txt
with open(IMAGE_DIR + ALL_CALIB_FILE_NAME) as file:
    f, px, py, baseline = [float(i) for i in file.read().split()[1::2]]
# corners: (x, y) coords
top_left_float = np.array([x_min, y_min])
top_left = top_left_float.astype(np.int)
bottom_right_float = np.array([x_max, y_max])
bottom_right = bottom_right_float.astype(np.int)
# box shape: (#row, #col)
box_shape = (bottom_right[1] - top_left[1], bottom_right[0] - top_left[0])

# ==================== Images ====================
left_img = cv2.imread(IMAGE_DIR + LEFT_IMAGE_NAME, cv2.IMREAD_COLOR)
right_img = cv2.imread(IMAGE_DIR + RIGHT_IMAGE_NAME, cv2.IMREAD_COLOR)
left_gray_img = cv2.imread(IMAGE_DIR + LEFT_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
right_gray_img = cv2.imread(IMAGE_DIR + RIGHT_IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
left_copy = np.copy(left_img)
left_rectangle_image = cv2.rectangle(left_copy, (top_left[0], top_left[1]),
                                     (bottom_right[0], bottom_right[1]),
                                     (0, 255, 0), 2)
left_cropped = np.copy(
    left_img[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]])


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


def part_a(l_img: np.ndarray, r_img: np.ndarray, patch_size: int = 3):
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

    assert len(l_img.shape) == len(r_img.shape), \
        "Images have different channels"

    if len(l_img.shape) == 2:
        l_img = l_img[:, :, np.newaxis]
        r_img = r_img[:, :, np.newaxis]

    half_patch_size = patch_size // 2

    # init masks and depth images
    SSD_mask = np.zeros(r_img.shape[:2])
    SSD_depth = np.zeros(box_shape)
    NC_mask = np.zeros(r_img.shape[:2])
    NC_depth = np.zeros(box_shape)

    # for each pixel in the bounding box in the left image, find the
    # corresponding pixel in the right image along the scanline
    for x in range(top_left[1], bottom_right[1]):
        print(f"Processing row {x} / {bottom_right[1]}")
        for ly in range(top_left[0], bottom_right[0]):
            l_patch = l_img[x - half_patch_size: x + half_patch_size + 1,
                      ly - half_patch_size: ly + half_patch_size + 1, :]

            # keep track of all computed cost/correlations along the scanline
            SSDs = np.full((r_img.shape[1],), np.inf)
            NCs = np.full((r_img.shape[1],), -np.inf)

            # Only scan till ly.
            scan_line_end = min(ly, r_img.shape[1] - half_patch_size)
            # only loop where r_patch's shape will match with that of l_patch
            for ry in range(half_patch_size, scan_line_end):
                r_patch = r_img[x - half_patch_size: x + half_patch_size + 1,
                          ry - half_patch_size: ry + half_patch_size + 1, :]
                SSDs[ry] = SSD(l_patch, r_patch)
                NCs[ry] = NC(l_patch, r_patch)

            # pick the index that's closest to the end of scanline
            min_SSD, SSD_ry = np.min(SSDs), np.argmin(SSDs)
            max_NC, NC_ry = np.max(NCs), np.argmax(NCs)

            # Mark the corresponding location for debugging
            # print((x, SSD_ry), (x, NC_ry))
            SSD_mask[x, SSD_ry] = 255
            NC_mask[x, NC_ry] = 255

            # calculate depth and map to depth matrix
            depth_x, depth_y = x - top_left[1], ly - top_left[0]
            if ly - SSD_ry == 0:
                # avoid 0-division
                SSD_depth[depth_x, depth_y] = 0
            else:
                SSD_depth[depth_x, depth_y] = f * baseline / (ly - SSD_ry)
            if ly - NC_ry == 0:
                # avoid 0-division
                NC_depth[depth_x, depth_y] = 0
            else:
                NC_depth[depth_x, depth_y] = f * baseline / (ly - NC_ry)
    SSD_depth[SSD_depth > 255] = 255
    NC_depth[NC_depth > 255] = 255
    print("{0} Part A End {0}".format("=" * 20))
    return SSD_mask, NC_mask, SSD_depth, NC_depth


def stitch_images():
    patch_sizes = [1, 3, 5, 7, 9]
    plt.clf()
    # fig, axes = plt.subplots(nrows=5, ncols=2)
    SSD_1 = cv2.imread(OUT_DIR + f"a_1_SSD_depth_jet.jpg")
    NC_1 = cv2.imread(OUT_DIR + f"a_1_NC_depth_jet.jpg")
    res = np.concatenate((SSD_1, NC_1), axis=1)
    for i in range(1, len(patch_sizes)):
        patch_size = patch_sizes[i]
        SSD_img = cv2.imread(OUT_DIR + f"a_{patch_size}_SSD_depth_jet.jpg")
        # axes[i][0].imshow(SSD_img, cmap="jet")
        # axes[i][0].axis('off')
        NC_img = cv2.imread(OUT_DIR + f"a_{patch_size}_NC_depth_jet.jpg")
        # axes[i][1].imshow(NC_img, cmap="jet")
        # axes[i][1].axis('off')
        curr_h_cat = np.concatenate((SSD_img, NC_img), axis=1)
        res = np.concatenate((res, curr_h_cat), axis=0)

    cv2.imwrite(OUT_DIR + "a_all.jpg", res)
    # plt.savefig(OUT_DIR + "a_all.jpg")


# ==================== Part b ====================
def part_b():
    """
    Model: https://github.com/ucbdrive/hd3
    See prediction implementation in q2.ipynb
    :return:
    """
    print("{0} Part B Start {0}".format("=" * 20))
    threshold = 17
    vec_orig = cv2.imread(OUT_DIR + "b_orig_vec.png", cv2.IMREAD_COLOR)
    vec_orig = cv2.cvtColor(vec_orig, cv2.COLOR_BGR2RGB)
    depth_img = f * baseline / vec_orig[:, :, 0]
    plt.clf()
    plt.imshow(depth_img, cmap="jet")
    plt.savefig(OUT_DIR + "b_depth_jet.jpg")
    plt.clf()
    depth_cropped = depth_img[top_left[1]: bottom_right[1],
                    top_left[0]: bottom_right[0]]
    plt.imshow(depth_cropped, cmap="jet")
    plt.savefig(OUT_DIR + "b_depth_cropped_jet.jpg")
    plt.clf()
    mask = np.copy(depth_cropped)
    mask[depth_cropped < threshold] = 255
    mask[depth_cropped >= threshold] = 0
    cv2.imwrite(OUT_DIR + "d_mask.jpg", mask)

    depth_masked = np.copy(left_cropped)
    depth_masked[mask == 0] = 0

    cv2.imwrite(OUT_DIR + "d_depth_masked.jpg", depth_masked)
    print("{0} Part B End {0}".format("=" * 20))


if __name__ == '__main__':
    PLOT = False
    DEBUG = False
    # if PLOT:
    #     fig, axes = plt.subplots(nrows=2, ncols=2)
    #     axes[0, 0].imshow(left_img)
    #     axes[0, 1].imshow(right_img)
    #     axes[1, 0].imshow(left_gray_img, cmap='gray')
    #     axes[1, 1].imshow(right_gray_img, cmap='gray')
    #     plt.show()
    #     plt.clf()
    #
    #     plt.imshow(left_rectangle_image)
    #     plt.show()
    #     plt.clf()

    # Part a
    # for patch_size in [1, 3, 5, 7, 9]:
    #     res = part_a(left_gray_img, right_gray_img, patch_size)
    #     SSD_mask, NC_mask, SSD_depth, NC_depth = res
    #     cv2.imwrite(OUT_DIR + f"a_{patch_size}_SSD_mask.jpg", SSD_mask)
    #     cv2.imwrite(OUT_DIR + f"a_{patch_size}_NC_mask.jpg", NC_mask)
    #     cv2.imwrite(OUT_DIR + f"a_{patch_size}_SSD_depth.jpg", SSD_depth)
    #     cv2.imwrite(OUT_DIR + f"a_{patch_size}_NC_depth.jpg", NC_depth)
    #     plt.imshow(SSD_depth, cmap='jet')
    #     plt.savefig(OUT_DIR + f"a_{patch_size}_SSD_depth_jet.jpg")
    #     plt.clf()
    #     plt.imshow(NC_depth, cmap='jet')
    #     plt.savefig(OUT_DIR + f"a_{patch_size}_NC_depth_jet.jpg")
    #     plt.clf()

    stitch_images()

    # part_b()
