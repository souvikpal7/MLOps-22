# FROM ubuntu:latest
FROM python:3.8.1
EXPOSE 5001
WORKDIR /exp
COPY ./models .
COPY ./*.py .
COPY ./api .
COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "api/__init__.py" ]
