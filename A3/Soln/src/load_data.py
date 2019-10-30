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
MEMBRANE_DATA = "../membrane"
CAT_NAME = "{}.jpg"
MEMBRANE_NAME = "{}.png"
# H x W x C
STANDARD_DIMS = (128, 128)


# ==================== Dataset ====================
class CatDataset(Dataset):
    """
    Custom dataset for Cat Data
    """
    input_dir: Path
    mask_dir: Path
    num_data: int
    all_transforms: transforms
    is_train: bool
    sample_type: str

    def __init__(self, root_dir: str, all_transforms: transforms = None,
                 is_train: bool = True, sample_type: str = "cat") -> None:
        """
        Initialize the dataset with the given directory and the transformation

        :param root_dir: the root directory
        :param all_transforms: the transformations
        :param is_train: true if the current dataset is the training set
        :param sample_type: the type of input data. Either "cat" or "membrane"
        """
        self.input_dir = Path(root_dir).joinpath(INPUT)
        self.mask_dir = Path(root_dir).joinpath(MASK)
        self.num_data = len([f for f in self.input_dir.iterdir()])
        self.all_transforms = all_transforms
        self.is_train = is_train
        self.sample_type = sample_type

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

        img_name = CAT_NAME if self.sample_type == "cat" else MEMBRANE_NAME

        image_path = Path(self.input_dir).joinpath(img_name.format(idx))
        label_path = Path(self.mask_dir).joinpath(img_name.format(idx))
        if not image_path.exists():
            raise FileNotFoundError("{} not found".format(image_path))
        if not label_path.exists():
            raise FileNotFoundError("{} not found".format(image_path))
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = image[np.newaxis, :, :].astype(float)

        label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = np.zeros((2, label_img.shape[0], label_img.shape[1]))
        # activated
        label[0] = (label_img != 0).astype(float)
        # dark
        label[1] = (label_img == 0).astype(float)

        image = torch.from_numpy(image).type(torch.float32)
        label = torch.from_numpy(label).type(torch.float32)

        if self.is_train and self.all_transforms:
            pil_img = transforms.functional.to_pil_image(image)
            pil_lbl = transforms.functional.to_pil_image(label)
            image = self.all_transforms(pil_img)
            label = self.all_transforms(pil_lbl)

        return image, label


def process_membrane() -> None:
    print("{0} Start Processing MEMBRANE {0}".format("=" * 10))
    data_path = Path(MEMBRANE_DATA)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError("membrane directory does not exist")
    train_dir_path = data_path.joinpath(TRAIN)
    test_dir_path = data_path.joinpath(TEST)
    input_dirs = [train_dir_path.joinpath(INPUT), test_dir_path.joinpath(INPUT),
                  train_dir_path.joinpath(MASK), test_dir_path.joinpath(MASK)]
    for curr_dir in input_dirs:
        for img_path in curr_dir.iterdir():
            if img_path.suffix == ".png":
                if curr_dir.resolve().parents[0].name == TEST and \
                        curr_dir.name == MASK and "_predict" in img_path.name:
                    img_name = img_path.name[
                               :-len("_predict.png")] + img_path.suffix
                    img_path.rename(curr_dir.joinpath(img_name))
                    img_path = curr_dir.joinpath(img_name)

                _resize_img(img_path)
    print("{0} Done Processing MEMBRANE {0}".format("=" * 10))


def process_cat() -> None:
    print("{0} Start Processing CAT {0}".format("=" * 10))
    data_path = Path(CAT_DATA)
    if not data_path.exists() or not data_path.is_dir():
        raise FileNotFoundError("cat_data directory does not exist")
    train_dir_path = data_path.joinpath(TRAIN)
    test_dir_path = data_path.joinpath(TEST)
    input_dirs = [train_dir_path.joinpath(INPUT), test_dir_path.joinpath(INPUT),
                  train_dir_path.joinpath(MASK), test_dir_path.joinpath(MASK)]
    for curr_dir in input_dirs:
        for img_path in curr_dir.iterdir():
            if img_path.suffix == ".jpg":
                if curr_dir.name == INPUT and "cat." in img_path.name:
                    img_name = img_path.name[len("cat."):]
                    img_path.rename(curr_dir.joinpath(img_name))
                    img_path = curr_dir.joinpath(img_name)
                elif curr_dir.name == MASK and "mask_cat." in img_path.name:
                    img_name = img_path.name[len("mask_cat."):]
                    img_path.rename(curr_dir.joinpath(img_name))
                    img_path = curr_dir.joinpath(img_name)
                if curr_dir.resolve().parents[0].name == TEST:
                    idx = int(img_path.stem)
                    if idx >= 60:
                        idx -= 60
                    new_name = str(idx) + img_path.suffix
                    img_path.rename(curr_dir.joinpath(new_name))
                    img_path = curr_dir.joinpath(new_name)

                _resize_img(img_path)

    print("{0} Done Processing CAT {0}".format("=" * 10))


def _resize_img(img_path):
    curr_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if curr_img is not None:
        if curr_img.shape != STANDARD_DIMS:
            print("Resizing {0}".format(img_path))
            resized_img = cv2.resize(curr_img, STANDARD_DIMS)
            cv2.imwrite(str(img_path), resized_img)
    else:
        print("WARNING: couldn't open {} "
              "as an image".format(img_path))


if __name__ == '__main__':
    process_cat()
    process_membrane()
