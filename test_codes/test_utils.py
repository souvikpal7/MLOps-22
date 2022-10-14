import pickle
import os
from sklearn import svm
# import sys
# sys.path.append("..")
import hand_written_digits


def test_saved_model_type():
    model_path = hand_written_digits.svm_model_fit()
    assert os.path.exists(model_path), "Model does not exist at returned path"

    # load the model
    with open(model_path, "rb") as file:
        svm_model = pickle.load(file)

    assert isinstance(svm_model, svm.SVC), "The model is not an instance of skleran,svm.SVC"
    
