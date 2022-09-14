# import modeules
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize


# data split
TRAIN_SPLIT = 0.8
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1

# Defining model hyper-parameters
C = [0.25, 0.5, 1.0, 1.25]
GAMMA = [1e-4, 0.0001, 0.001, 0.01]


# loading data
digits = datasets.load_digits()


# pre-processing the data with different size
n_samples = len(digits.images)
original_images = digits.images
resized_imgs1 = np.array([resize(img, (4, 4), anti_aliasing=True) for img in original_images])
resized_imgs2 = np.array([resize(img, (16, 16), anti_aliasing=True) for img in original_images])
resized_imgs3 = np.array([resize(img, (32, 32), anti_aliasing=True) for img in original_images])
dataset_list = [original_images, resized_imgs1, resized_imgs2, resized_imgs3]                               # to contain different image sizes


for images in dataset_list:
        print(f"******************* Input image size = {images[0].shape} *******************\n")
        # flattening the images
        data = images.reshape((n_samples, -1))

        # Split data into Train validation and test datasets
        X_train, X_test_val, y_train, y_test_val = train_test_split(
                    data, digits.target, train_size=TRAIN_SPLIT, shuffle=True, random_state=711
                    )
        X_val, X_test, y_val, y_test = train_test_split(
                X_test_val, y_test_val, train_size = VALID_SPLIT / (VALID_SPLIT + TEST_SPLIT),
                shuffle=True, random_state=711
                )


        # Creating a SVM Classifier
        best_acc = -np.inf
        best_model = None
        best_hyper_params = None
        best_row_idx = None
        result_df = pd.DataFrame(columns=["train", "dev", "test"])
        for hyper_params in list(product(C, GAMMA)):
                param_dict = {'C': hyper_params[0], 'gamma': hyper_params[1]}
                cur_row_idx = str(param_dict)
                clf = svm.SVC()
                clf.set_params(**param_dict)
                clf.fit(X_train, y_train)
                train_acc = clf.score(X_train, y_train)
                dev_acc = clf.score(X_val, y_val)
                test_acc = clf.score(X_test, y_test)
                result_df.loc[cur_row_idx] = [train_acc, dev_acc, test_acc]
                if dev_acc > best_acc:
                        best_acc = dev_acc
                        best_hyper_params = param_dict
                        best_model = clf
                        best_row_idx = cur_row_idx

        # Converting the accuracy into percentage and printing the result
        print(f"Printing the results:")
        result_df = result_df * 100
        print(result_df)

        print(f"\nBest set of Hyper Parameters {best_hyper_params} as per dev set\n")
        print(f"Results obtained with {best_hyper_params}:")
        print(f"{result_df.loc[best_row_idx]}")
        print("---------------------------------------------------------- \n")



