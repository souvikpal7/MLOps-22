# import modeules
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import utils
import datetime as dt

# setting as global variable
MODEL_DIR = "./models"

def svm_model_fit(
        X_train, 
        X_val, 
        X_test, 
        y_train, 
        y_val, 
        y_test,
        C=[0.25, 0.5, 1.0, 1.25], 
        GAMMA=[1e-4, 0.0001, 0.001, 0.01]
        ):

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
        print("\n")

        for col_name in result_df.columns:
                col_min, col_max, col_mean, col_median = utils.get_min_max_mean_median(result_df[col_name])
                print(f"Column : {col_name}")
                print(f"\tMinimum Accuracy: {col_min}")
                print(f"\tMaximum Accuracy: {col_max}")
                print(f"\tMean Accuracy: {col_mean}")
                print(f"\tMedian Accuracy: {col_median}")
                

        print(f"\nBest set of Hyper Parameters {best_hyper_params} as per dev set\n")
        print(f"Results obtained with {best_hyper_params}:")
        print(f"{result_df.loc[best_row_idx]}")
        print("---------------------------------------------------------- \n")

        best_model_file_name = f"svm_digit_clf_"
        for key in best_hyper_params:
                best_model_file_name += f"{key}:{best_hyper_params[key]}_"
        best_model_file_name += ".pkl"
        model_path = os.path.join(MODEL_DIR, best_model_file_name)
        utils.save_object(best_model, model_path)
        print(f"Path of best model: {model_path}")
        return model_path


def dtree_model_fit(
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        hyper_param_set={}
):
        assert len(hyper_param_set) > 0
        h_param_names = list(hyper_param_set.keys())
        list_of_hparams = [hyper_param_set[h_param] for h_param in h_param_names]
        result_df = pd.DataFrame(columns=["train", "dev", "test"])
        best_acc = -np.inf
        best_model = None
        best_hyper_params = None
        best_row_idx = None
        for h_param in list(product(*list_of_hparams)):
                hypam_dict = {k:v for k, v in zip(h_param_names, h_param)}
                print(hypam_dict)
                d_clf = DecisionTreeClassifier()
                d_clf.set_params(**hypam_dict)
                d_clf.fit(X_train, y_train)
                train_acc = d_clf.score(X_train, y_train)
                dev_acc = d_clf.score(X_val, y_val)
                test_acc = d_clf.score(X_test, y_test)
                cur_row_idx = str(hypam_dict)
                result_df.loc[cur_row_idx] = [train_acc, dev_acc, test_acc]
                if dev_acc > best_acc:
                        best_acc = dev_acc
                        best_hyper_params = hypam_dict
                        best_model = d_clf
                        best_row_idx = cur_row_idx

        # Converting the accuracy into percentage and printing the result
        print(f"Printing the results:")
        result_df = result_df * 100
        print(result_df)
        print("\n")

        for col_name in result_df.columns:
                col_min, col_max, col_mean, col_median = utils.get_min_max_mean_median(result_df[col_name])
                print(f"Column : {col_name}")
                print(f"\tMinimum Accuracy: {col_min}")
                print(f"\tMaximum Accuracy: {col_max}")
                print(f"\tMean Accuracy: {col_mean}")
                print(f"\tMedian Accuracy: {col_median}")
                

        print(f"\nBest set of Hyper Parameters {best_hyper_params} as per dev set\n")
        print(f"Results obtained with {best_hyper_params}:")
        print(f"{result_df.loc[best_row_idx]}")
        print("---------------------------------------------------------- \n")

        best_model_file_name = f"dtree_digit_clf_"
        for key in best_hyper_params:
                best_model_file_name += f"{key}:{best_hyper_params[key]}_"
        best_model_file_name += ".pkl"
        model_path = os.path.join(MODEL_DIR, best_model_file_name)
        utils.save_object(best_model, model_path)
        print(f"Path of best model: {model_path}")
        return model_path



if __name__ == "__main__":
        
        TRAIN_SPLIT = 0.8
        VALID_SPLIT = 0.1
        TEST_SPLIT = 0.1

        # Defining model hyper-parameters
        C = [0.25, 0.5, 1.0, 1.25]
        GAMMA = [1e-4, 0.0001, 0.001, 0.01]

        # do we want to plot the samples 
        PLOT = False

        # loading data
        dataset = "digits"
        data, labels = utils.get_data_and_labels(dataset)

        # plotting the images
        if PLOT:
                utils.plot_images(data, labels)

        #flattening the image
        n_samples = len(data)
        data = data.reshape((n_samples, -1))

        # splitting the data
        X_train, X_val, X_test, y_train, y_val, y_test = utils.get_train_dev_test_split(
                data, labels, TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT
        )

        model_path = svm_model_fit(X_train, X_val, X_test, y_train, y_val, y_test, C, GAMMA)
        print(model_path)

        h_param_set = {
                "max_depth": [2, 4, 8, None],
                "min_samples_split": [2, 8, 12],
                "min_impurity_decrease": [0.0, 0.01, 0.001]
        }
        d_model_path = dtree_model_fit(X_train, X_val, X_test, y_train, y_val, y_test, h_param_set)
        print(d_model_path)
        