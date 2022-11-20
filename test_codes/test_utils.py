import pickle
import os
from sklearn import svm
# import sys
# sys.path.append("..")
import hand_written_digits
import utils


def test_saved_model_type():
    data, labels = utils.get_data_and_labels("digits")
    #flattening the image
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    TRAIN_SPLIT = 0.8
    VALID_SPLIT = 0.1
    TEST_SPLIT = 0.1
    X_train, X_val, X_test, y_train, y_val, y_test = utils.get_train_dev_test_split(
        data,
        labels,
        TRAIN_SPLIT,
        VALID_SPLIT,
        TEST_SPLIT
    )

    model_path, _ = hand_written_digits.svm_model_fit(
        X_train, X_val, X_test, 
        y_train, y_val, y_test
    )
    assert os.path.exists(model_path), "Model does not exist at returned path"

    # load the model
    with open(model_path, "rb") as file:
        svm_model = pickle.load(file)

    assert isinstance(svm_model, svm.SVC), "The model is not an instance of skleran,svm.SVC"
    
def test_check_random_seed():
    data, labels = utils.get_data_and_labels("digits")
    TRAIN_SPLIT = 0.8
    VALID_SPLIT = 0.1
    TEST_SPLIT = 0.1

    # producing split for first time
    X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = utils.get_train_dev_test_split(
        data,
        labels,
        TRAIN_SPLIT,
        VALID_SPLIT,
        TEST_SPLIT
    )

    # producing split for second time
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = utils.get_train_dev_test_split(
        data,
        labels,
        TRAIN_SPLIT,
        VALID_SPLIT,
        TEST_SPLIT
    )

    assert (X_train1 == X_train2).all(), "X_train is not matched"
    assert (X_val1 == X_val2).all(), "X_val is not matched"
    assert (X_test1 == X_test2).all(), "X_test is not matched"

    assert (y_train1 == y_train2).all(), "y_train is not matched"
    assert (y_val1 == y_val2).all(), "y_val is not matched"
    assert (y_test1==y_test2).all(), "y_test is not matched"