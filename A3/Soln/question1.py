from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ==================== CONSTANTS ====================
TRAIN_INPUT_PATH = "./cat_data/Train/input"
TRAIN_MASK_PATH = "./cat_data/Train/mask"
TEST_INPUT_PATH = "./cat_data/Test/input"
TEST_MASK_PATH = "./cat_data/Test/mask"
STANDARD_DIMS = (128, 128, 3)


def resize_data() -> None:
    """
    Resize all the cat data to 128 * 128

    :return: None
    """
    print("{0} Start Processing {0}".format("=" * 10))
    paths = [TRAIN_INPUT_PATH, TRAIN_MASK_PATH, TEST_INPUT_PATH, TEST_MASK_PATH]
    for dir_path in paths:
        curr_dir = Path(dir_path)
        print("{0} processing {1} {0}".format("=" * 5, curr_dir))
        if curr_dir.is_dir():
            for img_path in curr_dir.iterdir():
                curr_img = cv2.imread(str(img_path))
                if curr_img is not None:
                    if curr_img.shape != STANDARD_DIMS:
                        print("Resizing {0}".format(img_path))
                        resized_img = cv2.resize(curr_img, STANDARD_DIMS)
                        cv2.imwrite(str(img_path), resized_img)
                else:
                    print("WARNING: couldn't open {} "
                          "as an image".format(img_path))
        else:
            raise FileNotFoundError("{0} is not a directory".format(dir_path))

    print("{0} Done Processing {0}".format("=" * 10))


def load_data() -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the data into numpy arrays from the given cat data

    :return: the loaded numpy data (x_train, y_train, x_test, y_test)
    """
    train_input_dir = Path(TRAIN_INPUT_PATH)
    train_mask_dir = Path(TRAIN_MASK_PATH)
    test_input_dir = Path(TEST_INPUT_PATH)
    test_mask_dir = Path(TEST_MASK_PATH)

    # Training init
    num_train = len([f for f in train_input_dir.iterdir()])
    x_train = np.zeros((num_train, 128, 128, 3), dtype=np.int8)
    y_train = np.zeros((num_train, 128, 128, 3), dtype=np.int8)

    # Testing init
    num_test = len([f for f in test_input_dir.iterdir()])
    x_test = np.zeros((num_test, 128, 128, 3), dtype=np.int8)
    y_test = np.zeros((num_test, 128, 128, 3), dtype=np.int8)

    dirs = [train_input_dir, train_mask_dir, test_input_dir, test_mask_dir]
    for i in range(len(dirs)):
        curr_dir = dirs[i]
        idx = 0
        for img_file in curr_dir.iterdir():
            if i == 0:
                x_train[idx, :, :, :] = cv2.imread(str(img_file))
            elif i == 1:
                y_train[idx, :, :, :] = cv2.imread(str(img_file))
            elif i == 2:
                x_test[idx, :, :, :] = cv2.imread(str(img_file))
            else:
                y_test[idx, :, :, :] = cv2.imread(str(img_file))
            idx += 1

    return x_train, y_train, x_test, y_test


def part1():
    """ Part 1 """
    pass


def main():
    part1()


if __name__ == '__main__':
    # resize_data()

    x_train, y_train, x_test, y_test = load_data()

    main()
