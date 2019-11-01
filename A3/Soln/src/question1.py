import random
import time
from pathlib import Path
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor  # using torch.Tensor is annoying
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from UNet import UNet, DownConv, DoubleConv

# ==================== CONSTANTS ====================
CHECKPOINT_PATH = "./Checkpoints/"
EVAL_PATH = "./eval/"
TRAIN = "Train"
TEST = "Test"
INPUT = "input"
MASK = "mask"
CAT_DATA = "./cat_data"
MEMBRANE_DATA = "./membrane"
CAT_NAME = "{}.jpg"
MEMBRANE_NAME = "{}.png"
# H x W x C
STANDARD_DIMS = (128, 128)


# ==================== Process/Resize Image ====================
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


# ==================== Attributes ====================
class AttrDict(dict):
    """
    A wrapper class for dictionary defining the attributes of the network
    """
    src_dir: str
    gpu: bool
    checkpoint: str
    kernel: int
    num_filters: int
    learn_rate: float
    batch_size: int
    epochs: int
    seed: int
    output_name: str
    sample_type: str

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        # The defaults
        self.update({
            'src_dir': CAT_DATA,
            'gpu': False,
            'checkpoint': "",
            'kernel': 3,
            'num_filters': 64,
            'learn_rate': 1e-3,
            'batch_size': 5,
            'epochs': 25,
            'seed': 0,
            'plot': True,
            'output_name': 'unet',
            'sample_type': "cat"
        })


# ==================== Custom Dataset ====================
class CatDataset(Dataset):
    """
    Custom dataset for Cat Data
    """
    input_dir: Path
    mask_dir: Path
    num_data: int
    augment: bool
    is_train: bool
    sample_type: str

    def __init__(self, root_dir: str, augment: bool = False,
                 is_train: bool = True, sample_type: str = "cat") -> None:
        """
        Initialize the dataset with the given directory and the transformation

        :param root_dir: the root directory
        :param augment: true if augment the image
        :param is_train: true if the current dataset is the training set
        :param sample_type: the type of input data. Either "cat" or "membrane"
        """
        self.input_dir = Path(root_dir).joinpath(INPUT)
        self.mask_dir = Path(root_dir).joinpath(MASK)
        self.num_data = len([f for f in self.input_dir.iterdir()])
        self.augment_methods = [horizontal_flip, vertical_flip, rotate_img,
                                noise_img, lambda x: x]
        self.augment = augment
        self.is_train = is_train
        self.sample_type = sample_type

    def __len__(self) -> int:
        """
        Returns the size of the dataset

        :return: the size of the dataset
        """
        if self.is_train and self.augment:
            return self.num_data * len(self.augment_methods)
        return self.num_data

    def __getitem__(self, idx: Union[int, Tensor]) \
            -> Tuple[Tensor, Tensor]:
        """
        Get the dataset at index idx

        :param idx: the index
        :return: the dictionary of the input and the mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        augment_method = lambda x: x
        if self.is_train and self.augment:
            augment_method = self.augment_methods[
                (idx + 1) % len(self.augment_methods)]
            idx = idx // len(self.augment_methods)

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
        label_img = label_img[np.newaxis, :, :].astype(float)

        if self.is_train and self.augment:
            image = augment_method(image)
            label_img = augment_method(label_img)

        label = np.zeros((2, label_img.shape[1], label_img.shape[2]))
        # activated
        label[0] = (label_img != 0).astype(float)
        # dark
        label[1] = (label_img == 0).astype(float)

        image = np.clip(image, a_min=0, a_max=255)
        label = np.clip(label, a_min=0, a_max=255)

        image = torch.from_numpy(image).type(torch.float32)
        label = torch.from_numpy(label).type(torch.float32)

        return image, label


def get_data_loaders(args: AttrDict, augment: bool = False) -> \
        Tuple[DataLoader, DataLoader]:
    """
    Get The data loaders for the training and testing data

    :param args: the arguments for the network
    :param augment: true if augment
    :return: the data loader
    """
    train_dataset = CatDataset(Path(args.src_dir).joinpath(TRAIN),
                               augment=augment,
                               is_train=True,
                               sample_type=args.sample_type)
    test_dataset = CatDataset(Path(args.src_dir).joinpath(TEST),
                              augment=augment,
                              is_train=False,
                              sample_type=args.sample_type)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    return train_data_loader, test_data_loader


# ==================== Augmentation Functions ====================
def horizontal_flip(img: np.ndarray) -> np.ndarray:
    return np.flip(img, 2)


def vertical_flip(img: np.ndarray) -> np.ndarray:
    return np.flip(img, 1)


def rotate_img(img: np.ndarray) -> np.ndarray:
    angle = random.uniform(0, 45)

    rows, cols = img.shape[1:]
    center = tuple(np.array([rows, cols]) / 2)

    R = cv2.getRotationMatrix2D(center, angle, 1.)
    rotated = np.copy(img)
    rotated[0] = cv2.warpAffine(img[0], R, (cols, rows))

    return rotated


def noise_img(img: np.ndarray) -> np.ndarray:
    img_cpy = np.copy(img)
    img_cpy = np.divide(img_cpy, 255.)
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img_cpy.shape)
    gauss = gauss.reshape(*img_cpy.shape)
    noisy = img_cpy + gauss
    noisy = np.multiply(noisy, 255.)
    return noisy


# ==================== Loss Functions ====================
def dice_loss(prediction: Tensor, label: Tensor) -> Tensor:
    """
    The dice loss function

    :param prediction: the prediction generated by network
    :param label: the true value
    :return: the dice loss
    """
    smooth = 1.
    pred_flat = prediction.view(-1)
    lab_flat = label.view(-1)
    intersection = (pred_flat * lab_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred_flat.sum() + lab_flat.sum() + smooth))


def bce_loss(prediction: Tensor, label: Tensor) -> Tensor:
    """
    The Binary Cross Entropy Loss

    :param prediction: the prediction generated by network
    :param label: the true value
    :return: the BCE Loss
    """
    prediction_flat = prediction.view(-1)
    label_flat = label.view(-1)
    return nn.BCELoss()(prediction_flat, label_flat)


def train(args, criterion, train_data_loader, test_data_loader, model=None):
    # Load Model
    if model is None:
        model = UNet(num_channels=1, num_classes=2, num_filters=64)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    print("Beginning training ...")
    if args.gpu:
        model.cuda()
    start = time.time()

    train_losses, valid_losses, valid_accuracies = [], [], []
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        losses = []
        for i, (images, labels) in enumerate(train_data_loader):
            if args.gpu:
                images, labels = images.cuda(), labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)

            loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        # Report training result
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print('Epoch [%d/%d], Loss: %.4f, Time (s): %d' % (
            epoch, args.epochs, float(avg_loss), time_elapsed))

        # Evaluate the model
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        val_loss, val_acc = run_validation_step(model, criterion, args.gpu,
                                                test_data_loader)

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)
        print('Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d' % (
            epoch, args.epochs, float(val_loss), val_acc, time_elapsed))

    if args.checkpoint:
        print('Saving model...')
        torch.save(model.state_dict(), args.checkpoint)

    return model


def compute_loss(criterion, outputs, labels, batch_size):
    """
    Helper function to compute the loss. Since this is a pixel-wise
    prediction task we need to reshape the output and ground truth
    tensors into a 2D tensor before passing it in to the loss criterion.
    Args:
      criterion: pytorch loss criterion
      outputs (pytorch tensor): predicted labels from the model
      labels (pytorch tensor): ground truth labels
      batch_size (int): batch size used for training
    Returns:
      pytorch tensor for loss
    """
    loss_out = outputs.transpose(1, 3) \
        .contiguous() \
        .view([batch_size * 128 * 128, 2])
    loss_lab = labels.transpose(1, 3) \
        .contiguous() \
        .view([batch_size * 128 * 128, 2])
    return criterion(loss_out, loss_lab)


def run_validation_step(cnn, criterion, gpu, test_data_loader):
    correct = 0.0
    total = 0.0
    losses = []
    for i, (images, labels) in enumerate(test_data_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()

        outputs = cnn(images)

        val_loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=labels.size(0))
        losses.append(val_loss.data.item())

        predicted = torch.argmax(outputs.data, 1, keepdim=True)
        total += outputs.numel()

        output_class = (outputs > 0.5).float()
        correct += (output_class == labels).float().sum()

    val_loss = np.mean(losses)
    # TODO: val always 100??
    val_acc = 100 * correct / total
    return val_loss, val_acc


def report_model(args: AttrDict, plot_name: str) -> None:
    print("Loading model from {}".format(args.checkpoint))
    args.batch_size = 1
    _, test_loader = get_data_loaders(args)

    model = UNet(num_channels=1, num_classes=2, num_filters=64)
    loaded_weights = torch.load(args.checkpoint)
    model.load_state_dict(loaded_weights)
    model.eval()

    fig, axarr = plt.subplots(12, 3, figsize=(100, 100))

    losses = []
    i = 0
    total = correct = 0
    for images, labels in test_loader:
        if i == 11:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            fig.savefig(plot_name + "_1.png")
            fig, axarr = plt.subplots(10, 3, figsize=(100, 100))
            i = 0

        pred = model(images)

        loss = compute_loss(dice_loss, pred, labels, batch_size=1)
        losses.append(loss.detach().numpy())
        np_pred = pred.detach().numpy()

        total += pred.numel()

        output_class = (pred > 0.5).float()
        correct += (output_class == labels).float().sum()

        axarr[i, 0].imshow(images[0][0], cmap="gray")
        axarr[i, 1].imshow(labels[0][0], cmap="gray")
        axarr[i, 2].imshow(np_pred[0][0], cmap="gray")
        i += 1

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(plot_name + "_2.png")

    print("Average Loss = {0}".format(np.mean(losses)))


# ==================== Part 1 ====================
def part1() -> None:
    """ Part 1 """
    print("{0} Part 1 {0}".format("=" * 10))
    # =============== Args ===============
    args = AttrDict()
    args_dict = {
        'src_dir': CAT_DATA,
        'kernel': 3,
        'num_filters': 64,
        'learn_rate': 1e-3,
        'batch_size': 5,
        'epochs': 20,
        'output_name': 'unet',
        'sample_type': 'cat',
    }
    args.update(args_dict)

    # =============== Train with BCE ===============
    print("{0} Training {1} {0}".format("=" * 5, "bce"))
    train_data_loader, test_data_loader = get_data_loaders(args)
    args.checkpoint = CHECKPOINT_PATH + "1_1_bce.pt"
    trained_network = train(args, bce_loss, train_data_loader, test_data_loader)
    print("{0} Done {0}".format("=" * 5))

    # =============== EVAL Dice ===============
    args.checkpoint = CHECKPOINT_PATH + "1_1_bce.pt"
    report_model(args, EVAL_PATH + "1_1_bce")

    # =============== Train with Dice ===============
    print("{0} Training {1} {0}".format("=" * 5, "dice"))
    args.checkpoint = CHECKPOINT_PATH + "1_1_dice.pt"
    args.batch_size = 2
    args.epochs = 100

    train_data_loader, test_data_loader = get_data_loaders(args)

    trained_network = train(args, dice_loss, train_data_loader,
                            test_data_loader)
    print("{0} Done {0}".format("=" * 5))

    # =============== EVAL Dice ===============
    args.checkpoint = CHECKPOINT_PATH + "1_1_dice.pt"
    report_model(args, EVAL_PATH + "1_1_dice")

    print("{0} Part 1 Done {0}".format("=" * 10))


# ==================== Part 2 ====================
def part2() -> None:
    """ Part 2 """
    # ==================== Load Data ====================
    print("{0} Part 2 {0}".format("=" * 10))
    # =============== Args ===============
    args = AttrDict()
    args_dict = {
        'src_dir': CAT_DATA,
        'kernel': 3,
        'num_filters': 64,
        'learn_rate': 1e-3,
        'batch_size': 20,
        'epochs': 100,
        'output_name': 'unet',
        'sample_type': 'cat',
    }
    args.update(args_dict)

    # =============== Train with Dice ===============
    print("{0} Training {1} {0}".format("=" * 5, "dice"))
    train_data_loader, test_data_loader = \
        get_data_loaders(args, augment=True)
    args.checkpoint = CHECKPOINT_PATH + "1_2_dice_transforms{}.pt".format(
        args.epochs)
    trained_network = train(args, dice_loss, train_data_loader,
                            test_data_loader)
    print("{0} Done {0}".format("=" * 5))
    print("{0} Part 2 Done {0}".format("=" * 10))

    args.checkpoint = CHECKPOINT_PATH + "1_2_dice_transforms{}.pt".format(
        args.epochs)
    report_model(args, EVAL_PATH + "1_2_transforms" + str(args.epochs))

    print("{0} Part 2 Done {0}".format("=" * 10))


# ==================== Part 3 ====================
def part3() -> None:
    """ Part 3 """
    print("{0} Part 3 {0}".format("=" * 10))
    # =============== Args ===============
    args = AttrDict()
    args_dict = {
        'src_dir': MEMBRANE_DATA,
        'kernel': 3,
        'num_filters': 64,
        'learn_rate': 1e-3,
        'batch_size': 5,
        'epochs': 20,
        'seed': 0,
        'output_name': 'unet',
        'sample_type': 'membrane',
    }
    args.update(args_dict)

    # =============== Training on Membrane with Dice ===============
    print("{0} Training {1} {0}".format("=" * 10, "membrane"))
    train_data_loader, test_data_loader = get_data_loaders(args)
    args.checkpoint = CHECKPOINT_PATH + "1_3_dice_membrane.pt"
    trained_unet = train(args, dice_loss, train_data_loader, test_data_loader)
    print("{0} Done {0}".format("=" * 10))

    # =============== Load pre-trained model ===============
    print("{0} Loading Pre-trained {0}".format("=" * 10))
    pre_trained_unet = UNet(num_channels=1, num_classes=2,
                            num_filters=args.num_filters)
    loaded_weights = torch.load(CHECKPOINT_PATH +
                                "/1_3_dice_membrane.pt")
    pre_trained_unet.load_state_dict(loaded_weights)
    pre_trained_unet.eval()
    print("{0} Done {0}".format("=" * 10))

    # TODO: set this to middle layer
    # reset the necessary layers
    pre_trained_unet.up_sample = nn.Upsample(scale_factor=2, mode="bilinear",
                                             align_corners=True)
    pre_trained_unet.down_conv2 = DownConv(2 * args.num_filters,
                                           4 * args.num_filters)
    pre_trained_unet.down_conv3 = DownConv(4 * args.num_filters,
                                           8 * args.num_filters)
    pre_trained_unet.down_conv4 = DownConv(8 * args.num_filters,
                                           8 * args.num_filters)
    pre_trained_unet.up_conv1 = DoubleConv(16 * args.num_filters,
                                           4 * args.num_filters)
    pre_trained_unet.up_conv2 = DoubleConv(8 * args.num_filters,
                                           2 * args.num_filters)
    pre_trained_unet.up_conv3 = DoubleConv(4 * args.num_filters,
                                           args.num_filters)
    pre_trained_unet.up_conv4 = DoubleConv(2 * args.num_filters,
                                           args.num_filters)

    pre_trained_unet.out_conv = nn.Conv2d(args.num_filters, 2, 1)

    # =============== Training on pre-trained on Cat with Dice ===============
    print("{0} Training {1} {0}".format("=" * 10, "cat"))
    train_data_loader, test_data_loader = get_data_loaders(args)
    args.sample_type = 'cat'
    args.src_dir = CAT_DATA
    args.checkpoint = CHECKPOINT_PATH + "1_3_dice_cat.pt"

    trained_network = train(args, dice_loss, train_data_loader,
                            test_data_loader, model=pre_trained_unet)
    print("{0} Done {0}".format("=" * 10))

    args.checkpoint = CHECKPOINT_PATH + "1_3_dice_cat.pt"
    report_model(args, EVAL_PATH + "1_3_dice_cat")

    print("{0} Part 3 Done {0}".format("=" * 10))


def part4():
    """ Part 4 """
    # =============== Args ===============
    args = AttrDict()
    args_dict = {
        'src_dir': CAT_DATA,
        'kernel': 3,
        'num_filters': 64,
        'learn_rate': 1e-3,
        'batch_size': 5,
        'epochs': 20,
        'seed': 0,
        'output_name': 'unet',
        'sample_type': 'cat',
    }
    args.update(args_dict)

    # loads the network
    trained_network = UNet(num_channels=1, num_classes=2, num_filters=64)
    loaded_weights = torch.load(CHECKPOINT_PATH + "1_3_dice_cat.pt")
    trained_network.load_state_dict(loaded_weights)
    trained_network.cuda()
    trained_network.eval()
    print("loaded network")

    # test for network
    train_data_loader, test_data_loader = get_data_loaders(args)

    for i in range(20):
        img_color = cv2.imread(
            CAT_DATA + "/" + TEST + "/" + INPUT + "/" + CAT_NAME.format(i))
        img_gray = np.copy(img_color)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
        img_tensor = torch.from_numpy(img_gray[np.newaxis, np.newaxis, :, :])
        img_tensor = img_tensor.float()

        # predict
        pred = trained_network(img_tensor.cuda())

        np_pred = pred.cpu().detach().numpy()[0][0]
        np_pred = np.uint8(np_pred * 255.)

        # find contours
        retval, dst = cv2.threshold(np_pred, 255 / 2, 255, 0)
        _, contours, _ = cv2.findContours(dst, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
        # plt.imshow(img_color)
        plt.imsave(EVAL_PATH + "1_4_{}.png".format(i), img_color)


def main() -> None:
    """ Main """
    part1()
    part2()
    part3()
    part4()


if __name__ == '__main__':
    # process_membrane()
    # process_cat()

    # ==================== Main ====================
    main()
