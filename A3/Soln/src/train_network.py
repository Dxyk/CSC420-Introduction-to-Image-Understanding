import time

import numpy as np
import torch

from UNet import UNet
from load_data import CAT_DATA


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
            'gpu': True,  # set to False if machine does not support cuda
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
        total += labels.size(0) * 128 * 128
        torch.set_printoptions(profile="full")
        torch.set_printoptions(profile="default")

        correct += (predicted == labels.data).sum()

    # TODO: val_loss sometimes > 1??
    val_loss = np.mean(losses)
    # TODO: val_acc always 100??
    val_acc = 100 * correct / total
    return val_loss, val_acc
