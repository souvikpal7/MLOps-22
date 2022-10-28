import numpy as np
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

def plot_images(dataset, labels, nrows=1, ncols=4, figsize=(10,3)):
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, image, label in zip(axes, dataset, labels):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def show_plots():
    plt.show()


def get_data_and_labels(dataset_name):
    supported_datasts = ["digits"]
    assert dataset_name in ["digits"], f"dataset should be one of {supported_datasts}"
    if dataset_name == "digits":
        digits = datasets.load_digits()
        X = digits.images
        Y = digits.target

    return X, Y

    
def get_train_dev_test_split(X_data, y_data, train_split, valid_split, test_split, shuffle=True):
    assert train_split + valid_split + test_split == 1.0
    assert len(X_data) == len(y_data), "Feature set and labels should have same size"
    
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X_data, 
        y_data, 
        train_size=train_split, 
        shuffle=shuffle, 
        random_state=711
        )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, 
        y_test_val, 
        train_size = valid_split / (valid_split + test_split),
        shuffle=True, 
        random_state=711
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_object(item, save_path):
    dir_name = os.path.dirname(save_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(save_path, 'wb') as file:
        pickle.dump(item, file)

def load_object(save_path):
    with open(save_path) as file:
        item = pickle.load(file)
    return item

def get_min_max_mean_median(accs):
        min_val = np.min(accs)
        max_val = np.max(accs)
        mean_val = np.mean(accs)
        median_val = np.median(accs)
        return min_val, max_val, mean_val, median_val

def split_non_overlapping(data, labels, splits):
    assert len(data) == len(labels)
    n_samples = len(data)
    rand_idxs = np.random.permutation(n_samples)
    len_splits = n_samples // splits
    x_data = []
    y_data = []
    for split in range(splits):
        split_idxs = rand_idxs[len_splits * split : len_splits  * (split + 1)]
        x_split = data[split_idxs]
        y_split = labels[split_idxs]
        x_data.append(x_split)
        y_data.append(y_split)
    
    return x_data, y_data
    
