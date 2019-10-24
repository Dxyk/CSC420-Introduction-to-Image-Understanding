from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn, optim, Tensor  # using torch.Tensor is annoying
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from UNet import UNet

# ==================== CONSTANTS ====================
TRAIN_INPUT_PATH = "./cat_data/Train/input"
TRAIN_MASK_PATH = "./cat_data/Train/mask"
TEST_INPUT_PATH = "./cat_data/Test/input"
TEST_MASK_PATH = "./cat_data/Test/mask"
TRAIN_PATH = "./cat_data/Train"
TEST_PATH = "./cat_data/Test"
CHECKPOINT_PATH = "./Checkpoints"
INPUT = "input"
MASK = "mask"
STANDARD_DIMS = (128, 128, 3)


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

    def __init__(self, root_dir: str, transform: transforms = None) -> None:
        """
        Initialize the dataset with the given directory and the transformation
        :param root_dir:
        :param transform:
        """
        self.transform = transform
        self.input_dir = Path(root_dir).joinpath(INPUT)
        self.mask_dir = Path(root_dir).joinpath(MASK)
        self.num_data = len([f for f in self.input_dir.iterdir()])
        self.X = np.zeros((self.num_data, 1, 128, 128), dtype=np.float)
        self.Y = np.zeros((self.num_data, 1, 128, 128), dtype=np.float)

        # fill in data
        for curr_dir in [self.input_dir, self.mask_dir]:
            idx = 0
            for img_file in curr_dir.iterdir():
                if curr_dir == self.input_dir:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    reshaped_img = img[np.newaxis, :, :]
                    self.X[idx, :, :, :] = reshaped_img
                elif curr_dir == self.mask_dir:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    reshaped_img = img[np.newaxis, :, :]
                    self.Y[idx, :, :, :] = reshaped_img
                idx += 1

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

        return self.X[idx], self.Y[idx]


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


def dice_loss(prediction, label):
    # Reference:
    # https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
    smooth = 1.

    iflat = prediction.view(-1)
    tflat = label.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def part1():
    """ Part 1 """
    unet_model = UNet().float()

    if use_gpu:
        unet_model = unet_model.cuda()

    criterion1 = nn.NLLLoss()
    criterion2 = dice_loss

    optimizer = optim.Adam(unet_model.parameters(), lr=0.003)

    # Training
    unet_model.train()

    for e in range(epochs):
        running_loss = 0
        for images, labels in tqdm(train_data_loader):
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            prediction = unet_model(images.float())
            print(prediction.shape)
            print(labels.shape)

            loss = criterion1(prediction, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            print(f"Training loss: {running_loss / len(train_data_loader)}")


def main():
    part1()


if __name__ == '__main__':
    # resize_data()

    # Hyper-Parameters
    epochs = 3
    use_gpu = False

    train_dataset = CatDataset(TRAIN_PATH)
    test_dataset = CatDataset(TEST_PATH)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    main()
