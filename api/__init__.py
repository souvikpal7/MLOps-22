from flask import Flask
from flask import request
import pickle
import numpy as np
from sklearn import svm
import os
import logging
# from . import digit_classifier_svm

# Digits_SVM = digit_classifier_svm.Digits_SVM
# svm_model = Digits_SVM().train()

app = Flask(__name__)
with open("models/svm_api_model.pkl", "rb") as file:
    svm_model = pickle.load(file)


def predict(x):
    x = np.array(x).reshape((1, -1))
    pred = svm_model.predict(x)
    return pred[0]    

@app.route("/")
def hello_world():
    return "Hello World"

@app.route("/check_same", methods=['POST'])
def check_same():
    image1 = request.json['image1']
    image2 = request.json['image2']
    cls_img1 = predict(image1)
    cls_img2 = predict(image2)
    ret_dict = {
        "same_digit": str(cls_img1==cls_img2),
        "image1_class": str(cls_img1),
        "image2_class": str(cls_img2)
        }
    return ret_dict


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

    #if __name__ == "__main__":
    #app.run(host ='0.0.0.0', port = 5001, debug = True) 