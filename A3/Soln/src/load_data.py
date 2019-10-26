from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor  # using torch.Tensor is annoying
from torch.utils.data import Dataset
from torchvision import transforms

# ==================== CONSTANTS ====================
TRAIN_PATH = "./cat_data/Train/"
TEST_PATH = "./cat_data/Test/"
CHECKPOINT_PATH = "./Checkpoints/"
IMAGE_NAME = "cat.{}.jpg"
LABEL_NAME = "mask_cat.{}.jpg"
INPUT = "input"
MASK = "mask"
# H x W x C
STANDARD_DIMS = (128, 128)


def resize_data() -> None:
    """
    Resize all the cat data to 128 * 128

    :return: None
    """
    print("{0} Start Processing {0}".format("=" * 10))
    paths = [Path(TRAIN_PATH).joinpath(INPUT), Path(TRAIN_PATH).joinpath(MASK),
             Path(TEST_PATH).joinpath(INPUT), Path(TEST_PATH).joinpath(MASK)]
    for curr_dir in paths:
        print("{0} processing {1} {0}".format("=" * 5, curr_dir))
        if curr_dir.is_dir():
            for img_path in curr_dir.iterdir():
                # remove .DS_Store
                if str(img_path).endswith(".DS_Store"):
                    img_path.unlink()
                else:
                    curr_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if curr_img is not None:
                        if curr_img.shape != STANDARD_DIMS:
                            print("Resizing {0}".format(img_path))
                            resized_img = cv2.resize(curr_img, STANDARD_DIMS)
                            cv2.imwrite(str(img_path), resized_img)
                    else:
                        print("WARNING: couldn't open {} "
                              "as an image".format(img_path))
        else:
            raise FileNotFoundError("{0} is not a directory".format(curr_dir))

    print("{0} Done Processing {0}".format("=" * 10))


# ==================== Dataset ====================
class CatDataset(Dataset):
    """
    Custom dataset for Cat Data
    """
    input_dir: Path
    mask_dir: Path
    num_data: int
    transform: transforms
    X: np.ndarray
    Y: np.ndarray

    def __init__(self, root_dir: str, transform: transforms = None,
                 is_train: bool = True) -> None:
        """
        Initialize the dataset with the given directory and the transformation

        :param root_dir: the root directory
        :param transform: the transformations
        :param is_train: true if the current dataset is the training set
        """
        self.input_dir = Path(root_dir).joinpath(INPUT)
        self.mask_dir = Path(root_dir).joinpath(MASK)
        self.num_data = len([f for f in self.input_dir.iterdir()])
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        """
        Returns the size of the dataset

        :return: the size of the dataset
        """
        return self.num_data

    def __getitem__(self, idx: Union[int, Tensor]) \
            -> Tuple[Tensor, np.ndarray]:
        """
        Get the dataset at index idx
        :param idx: the index
        :return: the dictionary of the input and the mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.is_train:
            # Test images start at index 60
            idx += 60

        image_path = Path(self.input_dir).joinpath(IMAGE_NAME.format(idx))
        label_path = Path(self.mask_dir).joinpath(LABEL_NAME.format(idx))
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = image[np.newaxis, :, :].astype(float)

        label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = np.zeros((2, label_img.shape[0], label_img.shape[1]))
        label[0] = (label_img != 0).astype(float)
        label[1] = (label_img == 0).astype(float)

        return image, label


if __name__ == '__main__':
    resize_data()
