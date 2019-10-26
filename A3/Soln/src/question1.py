from pathlib import Path
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim, Tensor  # using torch.Tensor is annoying
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from UNet import UNet

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
            -> Tuple[str, str, Tensor, np.ndarray]:
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
        image = image[np.newaxis, :, :]

        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = label[np.newaxis, :, :]
        # return image, label
        return image_path.name, label_path.name, image, label


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


def dice_loss(prediction, label):
    # Reference:
    # https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
    smooth = 1.

    iflat = prediction.view(-1)
    tflat = label.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def bce_loss(prediction, label):
    prediction_flat = prediction.view(-1)
    label_flat = label.view(-1)
    return nn.BCELoss()(prediction_flat, label_flat)


def train(criterion):
    unet_model = UNet().float()

    if use_gpu:
        unet_model = unet_model.cuda()

    optimizer = optim.Adam(unet_model.parameters(), lr=0.003)

    for e in range(epochs):
        unet_model.train()
        running_loss = 0
        for images, labels in tqdm(train_data_loader):
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            prediction = unet_model(images.float())

            loss = criterion(prediction, labels.float())
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            print(f"Training loss: {running_loss / len(train_data_loader)}")

            if 1:
                target_path = Path(CHECKPOINT_PATH).joinpath(
                    'CP{}.pth'.format(e + 1))
                torch.save(unet_model.state_dict(),
                           str(target_path))
                print('Checkpoint {} saved !'.format(e + 1))


def part1():
    """ Part 1 """
    criterion1 = bce_loss
    # criterion1 = nn.CrossEntropyLoss()
    criterion2 = dice_loss
    train(criterion1)
    train(criterion2)


def main():
    part1()


if __name__ == '__main__':
    resize_data()

    # ==================== Hyper-Parameters ====================
    epochs = 3
    use_gpu = False
    batch_size = 3
    transform = None
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])

    # ==================== Load Data ====================
    print("{0} Loading Data {0}".format("=" * 10))
    train_dataset = CatDataset(TRAIN_PATH, transform=transform, is_train=True)
    test_dataset = CatDataset(TEST_PATH, transform=transform, is_train=False)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=True)
    print("{0} Done {0}".format("=" * 10))

    # ==================== Test Data ====================
    # img_path, label_path, image, label = next(iter(train_data_loader))
    # # img_path, label_path, image, label = next(iter(test_data_loader))
    # print(img_path)
    # print(label_path)
    # print(image.shape, label.shape)
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(image[0, 0, :], cmap="gray")
    # axarr[1].imshow(label[0, 0, :], cmap="gray")
    # plt.show()

    main()
