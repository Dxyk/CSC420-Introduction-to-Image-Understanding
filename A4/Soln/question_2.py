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
# corners: (x, y) coords
top_left = (int(bounding_box[0]), int(bounding_box[1]))
bottom_right = (int(bounding_box[2]) + 1, int(bounding_box[3]) + 1)
# box shape: (#row, #col)
box_shape = (bottom_right[1] - top_left[1], bottom_right[0] - top_left[0])

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


def part_a(block_size: int = 3, step: int = 1, save_image: bool = True) -> None:
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

    half_block_size = block_size // 2

    SSD_mask = np.zeros(left_gray_img.shape)
    SSD_depth = np.zeros(left_gray_img.shape)
    SSD_depth_crop = np.zeros(box_shape)
    NC_mask = np.zeros(left_gray_img.shape)
    NC_depth = np.zeros(left_gray_img.shape)
    NC_depth_crop = np.zeros(box_shape)

    # for each pixel in the bounding box in the left image, find the
    # corresponding pixel in the right image along the scanline
    for x in range(top_left[1], bottom_right[1]):
        print(f"Processing row {x} / {bottom_right[1]}")
        for ly in range(top_left[0], bottom_right[0]):
            l_patch = left_gray_img[
                      x - half_block_size: x + half_block_size + 1,
                      ly - half_block_size: ly + half_block_size + 1]

            # keep track of all computed cost/correlations along the scanline
            SSDs = np.full((right_gray_img.shape[1],), np.inf)
            NCs = np.full((right_gray_img.shape[1],), -np.inf)

            # only loop where r_patch's shape will match with that of l_patch
            for ry in range(half_block_size,
                            min(ly, right_gray_img.shape[1] - half_block_size),
                            step):
                r_patch = right_gray_img[
                          x - half_block_size: x + half_block_size + 1,
                          ry - half_block_size: ry + half_block_size + 1]
                SSDs[ry] = SSD(l_patch, r_patch)
                NCs[ry] = NC(l_patch, r_patch)

            min_SSD, SSD_ry = np.min(SSDs), np.argmin(SSDs)
            max_NC, NC_ry = np.max(NCs), np.argmax(NCs)

            # Mark the corresponding location for debugging
            # print((x, SSD_ry), (x, NC_ry))
            SSD_mask[x, SSD_ry] = 255
            NC_mask[x, NC_ry] = 255

            # calculate depth and map to depth matrix
            depth_x, depth_y = x - top_left[1], ly - top_left[0]
            camera_center = right_gray_img.shape[1] // 2
            ly_c = ly - camera_center
            SSD_ry_c = SSD_ry - camera_center
            NC_ry_c = NC_ry - camera_center
            if ly_c - SSD_ry_c == 0:
                SSD_depth[x, ly] = 0
                SSD_depth_crop[depth_x, depth_y] = 0
            else:
                SSD_depth[x, ly] = f * baseline / (ly_c - SSD_ry_c)
                SSD_depth_crop[depth_x, depth_y] = f * baseline / (
                        ly_c - SSD_ry_c)
            if ly_c - NC_ry_c == 0:
                NC_depth[x, ly] = 0
                NC_depth_crop[depth_x, depth_y] = 0
            else:
                NC_depth[x, ly] = f * baseline / (ly_c - NC_ry_c)
                NC_depth_crop[depth_x, depth_y] = f * baseline / (
                        ly_c - NC_ry_c)

            if DEBUG:
                print(f"SSD: {min_SSD}, SSD idx: {SSD_ry}, "
                      f"depth: {SSD_depth[x, ly]}")
                print(f"NC: {max_NC}, NC idx: {NC_ry}, "
                      f"depth: {NC_depth[x, ly]}")

    # Convert depth maps to appropriate scale
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(SSD_depth_crop)
    SSD_depth_crop -= min_val
    SSD_depth_crop = cv2.convertScaleAbs(SSD_depth_crop, None,
                                         255. / float(max_val - min_val))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(NC_depth_crop)
    NC_depth_crop -= min_val
    NC_depth_crop = cv2.convertScaleAbs(NC_depth_crop, None,
                                        255. / float(max_val - min_val))

    if PLOT:
        plt.clf()
        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes[0, 0].imshow(SSD_mask, cmap='gray')
        axes[0, 1].imshow(NC_mask, cmap='gray')
        axes[1, 0].imshow(SSD_depth, cmap='gray')
        axes[1, 1].imshow(NC_depth, cmap='gray')
        plt.show()
        plt.clf()

        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(SSD_depth_crop, cmap='gray')
        axes[1].imshow(NC_depth_crop, cmap='gray')
        plt.show()
        plt.clf()

    if save_image:
        cv2.imwrite(OUT_DIR + f"a_{block_size}_{step}_SSD_mask.jpg", SSD_mask)
        cv2.imwrite(OUT_DIR + f"a_{block_size}_{step}_NC_mask.jpg", NC_mask)
        cv2.imwrite(OUT_DIR + f"a_{block_size}_{step}_SSD_depth.jpg", SSD_depth)
        cv2.imwrite(OUT_DIR + f"a_{block_size}_{step}_NC_depth.jpg", NC_depth)
        cv2.imwrite(OUT_DIR + f"a_{block_size}_{step}_SSD_depth_crop.jpg",
                    SSD_depth_crop)
        cv2.imwrite(OUT_DIR + f"a_{block_size}_{step}_NC_depth_crop.jpg",
                    NC_depth_crop)

    print("{0} Part A End {0}".format("=" * 20))


def stitch_images():
    block_sizes = [1, 3, 5]
    patch_sizes = [1, 3, 5]
    plt.clf()
    fig, axes = plt.subplots(nrows=9, ncols=2)
    for i in range(len(block_sizes)):
        for j in range(len(patch_sizes)):
            block_size = block_sizes[i]
            patch_size = patch_sizes[j]
            curr_img = cv2.imread(
                OUT_DIR + f"a_{block_size}_{patch_size}_SSD_depth_crop.jpg")
            axes[3 * i + j][0].imshow(curr_img, cmap='gray')
            axes[3 * i + j][0].axis('off')
            curr_img = cv2.imread(
                OUT_DIR + f"a_{block_size}_{patch_size}_NC_depth_crop.jpg")
            axes[3 * i + j][1].imshow(curr_img, cmap='gray')
            axes[3 * i + j][1].axis('off')

    plt.savefig(OUT_DIR + "a_all.jpg")


# ==================== Part b ====================
def part_b():
    """
    Model: https://github.com/ucbdrive/hd3
    See prediction implementation in q2.ipynb
    :return:
    """
    print("{0} Part B Start {0}".format("=" * 20))
    vec_orig = cv2.imread(OUT_DIR + "b_orig_vis.png", cv2.IMREAD_COLOR)
    cropped_out = vec_orig[top_left[1]:bottom_right[1],
                  top_left[0]:bottom_right[0]]
    cv2.imwrite(OUT_DIR + "b_cropped_vis.jpg", cropped_out)
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

    # for block_size in [1, 3, 5]:
    #     for step in [1, 3, 5]:
    #         part_a(block_size, step, save_image=True)
    # stitch_images()

    part_b()
