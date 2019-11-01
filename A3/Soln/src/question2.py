# imports
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from torch import nn, Tensor  # using torch.Tensor is annoying
from torch.utils.data import Dataset, DataLoader

from CNN import CNN

CHECKPOINT_PATH = "./Checkpoints/"
EVAL_PATH = "./eval"
Q2_DATA = "../q2_data"
TRAIN = "Train"
TEST = "Test"
INPUT = "input"
LABEL = "label"
FILE_NAME = "{}.png"

FILE_BATCH_SIZE = 100
IMG_DIM = 200


class AttrDict(dict):
    """
    A wrapper class for dictionary defining the attributes of the network
    """
    gpu: bool
    checkpoint: str
    kernel: int
    num_filters: int
    learn_rate: float
    batch_size: int
    epochs: int
    seed: int
    total_img: int

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        # The defaults
        self.update({
            'gpu': True,  # set to False if machine does not support cuda
            'checkpoint': "",
            'kernel': 3,
            'num_filters': 64,
            'learn_rate': 1e-3,
            'batch_size': 5,
            'epochs': 25,
            'seed': 0,
            'total_img': 5000
        })


class MyDataSet(Dataset):
    """
    Custom dataset for Circles
    """
    input_dir: Path
    label_dir: Path
    num_data: int
    is_train: bool

    def __init__(self, root_dir, is_train) -> None:
        """
        Initialize the dataset with the given directory and the transformation

        :param root_dir: the root directory
        """
        # self.input_dir = Path(root_dir).joinpath(INPUT)
        # self.label_dir = Path(root_dir).joinpath(LABEL)
        # self.num_data = len([f for f in self.input_dir.iterdir()]) * FILE_BATCH_SIZE
        self.is_train = is_train
        if self.is_train:
            img_path = Path(root_dir).joinpath(FILE_NAME.format("train_input"))
            label_path = Path(root_dir).joinpath(
                FILE_NAME.format("train_label"))
        else:
            img_path = Path(root_dir).joinpath(FILE_NAME.format("test_input"))
            label_path = Path(root_dir).joinpath(FILE_NAME.format("test_label"))

        self.images = np.load(str(img_path))
        self.labels = np.load(str(label_path))

    def __len__(self) -> int:
        """
        Returns the size of the dataset

        :return: the size of the dataset
        """
        return self.images.shape[0]
        # return self.num_data

    def __getitem__(self, idx: Union[int, Tensor]) \
            -> Tuple[Tensor, Tensor]:
        """
        Get the dataset at index idx

        :param idx: the index
        :return: the dictionary of the input and the mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = torch.from_numpy(self.images[idx]).type(torch.float32)
        labels = torch.from_numpy(self.labels[idx]).type(torch.float32)

        return images, labels


def get_data_loaders(args: AttrDict):
    """
    Get The data loaders for the training and testing data

    :param args: the arguments for the network
    :param augment: true if augment
    :return: the data loader
    """
    train_dataset = MyDataSet(Q2_DATA, True)
    test_dataset = MyDataSet(Q2_DATA, False)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                  shuffle=True)
    return train_data_loader, test_data_loader


def generate_data(args):
    data_path = Path(Q2_DATA)
    train_path = data_path.joinpath(TRAIN)
    test_path = data_path.joinpath(TEST)
    train_input_path = train_path.joinpath(INPUT)
    train_label_path = train_path.joinpath(LABEL)
    test_input_path = test_path.joinpath(INPUT)
    test_label_path = test_path.joinpath(LABEL)

    train_input_path.mkdir(exist_ok=True, parents=True)
    train_label_path.mkdir(exist_ok=True, parents=True)
    test_input_path.mkdir(exist_ok=True, parents=True)
    test_label_path.mkdir(exist_ok=True, parents=True)

    split_idx = int(args.total_img * .7)
    train_input, train_label, test_input, test_label = [], [], [], []
    for i in range(args.total_img):
        params, img = noisy_circle(IMG_DIM, 50, 2)
        img = img[np.newaxis, :, :]
        if i < split_idx:
            train_input.append(img)
            train_label.append(np.array(params, dtype=np.float))
        else:
            test_input.append(img)
            test_label.append(np.array(params, dtype=np.float))
    np.save(str(data_path.joinpath("train_input")), np.array(train_input))
    np.save(str(data_path.joinpath("train_label")), np.array(train_label))
    np.save(str(data_path.joinpath("test_input")), np.array(test_input))
    np.save(str(data_path.joinpath("test_label")), np.array(test_label))


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = ((rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1]))
    img[rr[valid], cc[valid]] = val[valid]
    return img


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return shape0.intersection(shape1).area / shape0.union(shape1).area


def train(args, criterion, train_data_loader, test_data_loader, model=None):
    # Load Model
    if model is None:
        model = CNN(kernel=args.kernel, num_filters=args.num_filters,
                    num_colours=3, num_in_channels=1)

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

            loss = criterion(outputs, labels)
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


def euclidean_distance_loss(output, labels):
    return torch.sum(torch.sqrt(torch.sum((output - labels) ** 2)))


def run_validation_step(cnn, criterion, gpu, test_data_loader):
    correct = 0.0
    total = 0.0
    losses = []
    for i, (images, labels) in enumerate(test_data_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()

        outputs = cnn(images)

        val_loss = criterion(outputs, labels)
        losses.append(val_loss.data.item())

        total += outputs.numel()

        output_class = (outputs > 0.5).float()
        correct += (output_class == labels).float().sum()

    val_loss = np.mean(losses)
    val_acc = 100 * correct / total
    return val_loss, val_acc


def train_model():
    args = AttrDict()
    args_dict = {
        'kernel': 3,
        'num_filters': 32,
        'learn_rate': 1e-3,
        'batch_size': 20,
        'epochs': 25,
    }
    args.update(args_dict)
    train_data_loader, test_data_loader = get_data_loaders(args)

    args.checkpoint = CHECKPOINT_PATH + "2_1_{}.pt".format(args.epochs)
    return train(args, nn.MSELoss(), train_data_loader,
                 test_data_loader)


def find_circle(img, model):
    pred = model(img.cuda())
    np_pred = pred.cpu().detach().numpy()

    return np_pred[0][0], np_pred[0][1], np_pred[0][2]
    # return 100, 100, 30


def main():
    # generate_data(args)

    # train
    trained_model = train_model()

    # eval
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(trained_model)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())


if __name__ == '__main__':
    main()
