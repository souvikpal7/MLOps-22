from email import utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
import hand_written_digits
import utils
import pickle
import os

def test_bias_diff_cls():
    digits = datasets.load_digits()
    X = digits.images
    Y = digits.target
    _, _, X_test, _, _, y_test = utils.get_train_dev_test_split(X, Y, 0.8, 0.1, 0.1)

    model_path = hand_written_digits.svm_model_fit()
    assert os.path.exists(model_path), "Model does not exist at returned path"

    # load the model
    with open(model_path, "rb") as file:
        svm_model = pickle.load(file)

    n_samples = len(X_test)
    X_test = X_test.reshape((n_samples, -1))
    y_t_pred = svm_model.predict(X_test)
    assert len(set(y_t_pred)) >1, "model is not perdicting more than one class"


def test_all_class_prediction():
    digits = datasets.load_digits()
    X = digits.images
    Y = digits.target
    _, _, X_test, _, _, y_test = utils.get_train_dev_test_split(X, Y, 0.8, 0.1, 0.1)

    model_path = hand_written_digits.svm_model_fit()
    assert os.path.exists(model_path), "Model does not exist at returned path"

    # load the model
    with open(model_path, "rb") as file:
        svm_model = pickle.load(file)

    n_samples = len(X_test)
    X_test = X_test.reshape((n_samples, -1))
    y_t_pred = svm_model.predict(X_test)
    assert set(y_t_pred) == set(y_test), "all classes are not predicting"
    


    
