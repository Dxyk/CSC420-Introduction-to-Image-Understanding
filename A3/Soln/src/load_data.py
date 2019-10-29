from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor  # using torch.Tensor is annoying
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional

# ==================== CONSTANTS ====================
CHECKPOINT_PATH = "./Checkpoints/"
TRAIN = "Train"
TEST = "Test"
INPUT = "input"
MASK = "mask"
CAT_DATA = "../cat_data"
CAT_IMAGE_NAME = "cat.{}.jpg"
CAT_MASK_NAME = "mask_cat.{}.jpg"
MEMBRANE_DATA = "../membrane"
MEMBRANE_IMAGE_NAME = "{}.png"
# H x W x C
STANDARD_DIMS = (128, 128)


def process_membrane(data_path: str) -> None:
    test_path = Path(data_path).joinpath(TEST)
    test_input_path = test_path.joinpath(INPUT)
    test_mask_path = test_path.joinpath(MASK)
    test_input_path.mkdir(parents=True, exist_ok=True)
    test_mask_path.mkdir(parents=True, exist_ok=True)

    for img_path in test_path.iterdir():
        # membrane remove _predict
        if str(img_path).endswith("_predict.png"):
            img_path.replace(test_mask_path.joinpath(img_path.name))
        elif str(img_path).endswith("png"):
            img_path.replace(test_input_path.joinpath(img_path.name))


def resize_data(data_path: str) -> None:
    """
    Resize all the cat data to 128 * 128

    :return: None
    """
    print("{0} Start Processing {0}".format("=" * 10))
    paths = [Path(data_path).joinpath(TRAIN).joinpath(INPUT),
             Path(data_path).joinpath(TRAIN).joinpath(MASK),
             Path(data_path).joinpath(TEST).joinpath(INPUT),
             Path(data_path).joinpath(TEST).joinpath(MASK)]
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
    transforms: transforms
    X: np.ndarray
    Y: np.ndarray

    def __init__(self, root_dir: str, transforms: transforms = None,
                 is_train: bool = True, is_cat: bool = True) -> None:
        """
        Initialize the dataset with the given directory and the transformation

        :param root_dir: the root directory
        :param transforms: the transformations
        :param is_train: true if the current dataset is the training set
        """
        self.input_dir = Path(root_dir).joinpath(INPUT)
        self.mask_dir = Path(root_dir).joinpath(MASK)
        self.num_data = len([f for f in self.input_dir.iterdir()])
        self.transforms = transforms
        self.is_train = is_train
        self.is_cat = is_cat

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

        if self.is_cat and not self.is_train:
            # Test images start at index 60
            idx += 60

        img_name = CAT_IMAGE_NAME if self.is_cat else MEMBRANE_IMAGE_NAME
        mask_name = CAT_MASK_NAME if self.is_cat else MEMBRANE_IMAGE_NAME

        image_path = Path(self.input_dir).joinpath(img_name.format(idx))
        label_path = Path(self.mask_dir).joinpath(mask_name.format(idx))
        if not image_path.exists():
            raise FileNotFoundError("{} not found".format(image_path))
        if not label_path.exists():
            raise FileNotFoundError("{} not found".format(image_path))
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = image[np.newaxis, :, :].astype(float)

        label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = np.zeros((2, label_img.shape[0], label_img.shape[1]))
        label[0] = (label_img != 0).astype(float)
        label[1] = (label_img == 0).astype(float)

        image = torch.from_numpy(image).type(torch.float32)
        label = torch.from_numpy(label).type(torch.float32)

        if self.is_train and self.transforms:
            pil_img = transforms.functional.to_pil_image(image)
            pil_lbl = transforms.functional.to_pil_image(label)
            image = self.transforms(pil_img)
            label = self.transforms(pil_lbl)

        return image, label


if __name__ == '__main__':
    resize_data(CAT_DATA)

    process_membrane(MEMBRANE_DATA)
    resize_data(MEMBRANE_DATA)
