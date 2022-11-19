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
    
