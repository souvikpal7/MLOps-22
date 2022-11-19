# import modeules..
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
        return model_path, best_model


def dtree_model_fit(
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        hyper_param_set={
                "max_depth": [2, 4, 8, None],
                "min_samples_split": [2, 8, 12],
                "min_impurity_decrease": [0.0, 0.01, 0.001]
        }
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
        return model_path, best_model


def compare_models(data, labels, splits=5):
        x_splits, y_splits = utils.split_non_overlapping(data, labels, splits)
        svm_scores = []
        d_tree_scores = []

        for x_split, y_split in zip(x_splits, y_splits):
                X_train, X_val, X_test, y_train, y_val, y_test = utils.get_train_dev_test_split(
                x_split, y_split, 0.8, 0.1, 0.1
                )
                _, svm_model = svm_model_fit(X_train, X_val, X_test, y_train, y_val, y_test)
                a_test_score = svm_model.score(X_test, y_test)
                svm_scores.append(a_test_score)

                _, d_tree_model = dtree_model_fit(X_train, X_val, X_test, y_train, y_val, y_test)
                b_test_score = d_tree_model.score(X_test, y_test)
                d_tree_scores.append(b_test_score)


        svm_mean = np.mean(svm_scores)
        svm_std = np.std(svm_scores)
        
        d_tree_mean = np.mean(d_tree_scores)
        d_tree_std = np.std(d_tree_scores)

        df = pd.DataFrame({'svm': svm_scores, 'decision_tree': d_tree_scores})
        print(df)

        print(f"SVM model mean and std are {svm_mean} and {svm_std}")
        print(f"Decision Tree model mean and std are {d_tree_mean} and {d_tree_std}")

if __name__ == "__main__":
        np.random.seed(711)

        # loading data
        dataset = "digits"
        data, labels = utils.get_data_and_labels(dataset)

        #flattening the image
        n_samples = len(data)
        data = data.reshape((n_samples, -1))

        compare_models(data, labels, 5)

        