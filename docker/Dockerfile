# FROM ubuntu:latest
FROM python:3.8.1
EXPOSE 5001
WORKDIR /exp
COPY ./models models
COPY ./api api
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
ENV FLASK_APP=./api/app.py
CMD ["python3", "./api/app.py" ]
#CMD ["flask", "run", "--debug", "-p", "5001"]


#FROM ubuntu:latest
#FROM python:3.8.1
#COPY /api/*.py /exp/
#COPY ./requirements.txt /exp/requirements.txt
#COPY ./svm_gamma=0.0004_C=3.0.joblib /exp/svm_gamma=0.0004_C=3.0.joblib
#RUN pip3 install --no-cache-dir -r /exp/requirements.txt
#WORKDIR /exp
#EXPOSE 8080
#CMD ["python3", "./app.py" ]