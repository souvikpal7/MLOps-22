import matplotlib.pyplot as plt

def plot_images(dataset, labels, nrows=1, ncols=4, figsize=(10,3)):
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, image, label in zip(axes, dataset, labels):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def show_plot():
    plt.show()
