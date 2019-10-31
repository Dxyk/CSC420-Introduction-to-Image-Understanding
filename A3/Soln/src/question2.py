import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa

from CNN import CNN
from train_network import AttrDict


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


def find_circle(img):
    # TODO
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return shape0.intersection(shape1).area / shape0.union(shape1).area


def train_model():
    args = AttrDict()
    args_dict = {
        'kernel': 3,
        'num_filters': 64,
        'learn_rate': 1e-3,
        'batch_size': 20,
        'epochs': 75,
        'output_name': 'unet',
        'sample_type': 'cat',
    }
    args.update(args_dict)
    cnn_model = CNN(args.kernel, args.num_filters, 2, 1)


def get_data_sets(load=True):
    if load:
        print("Loading Data")
        train_circles = np.load("../q2_data/train_circles.npz")
        train_labels = np.load("../q2_data/train_labels.npz")
        test_circles = np.load("../q2_data/test_circles.npz")
        test_labels = np.load("../q2_data/test_labels.npz")
        print("Done Loading Data")
    else:
        print("Generating Data")

        all_circles = np.zeros((100000, 200, 200))
        all_labels = np.zeros((100000, 3))  # row col rad
        for i in range(100000):
            params, img = noisy_circle(200, 50, 2)
            all_circles[i] = img
            all_labels[i] = np.array(params, dtype=np.float)

        train_circles, train_labels = all_circles[:70000], all_labels[:70000]
        test_circles, test_labels = all_circles[70000:], all_labels[70000:]

        np.save("../q2_data/train_circles.npz", train_circles)
        np.save("../q2_data/train_labels.npz", train_labels)
        np.save("../q2_data/test_circles.npz", test_circles)
        np.save("../q2_data/test_labels.npz", test_labels)
        print("Done Generating Data")

    return train_circles, train_labels, test_circles, test_labels


def main():
    # get datasets
    get_data_sets(False)

    # train

    # eval
    # results = []
    # for _ in range(1000):
    #     params, img = noisy_circle(200, 50, 2)
    #     detected = find_circle(img)
    #     results.append(iou(params, detected))
    # results = np.array(results)
    # print((results > 0.7).mean())


if __name__ == '__main__':
    main()
