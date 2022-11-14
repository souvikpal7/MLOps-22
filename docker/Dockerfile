# FROM ubuntu:latest
FROM python:3.8.1
EXPOSE 5001
WORKDIR /exp
COPY ./models/* /exp/models
COPY ./api/* /exp/
COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
CMD ["python3", "__init__.py" ]
