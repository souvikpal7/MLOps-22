#FROM ubuntu:14.04
#COPY ./*.sh /exp/
#ENTRYPOINT ["/exp/run.sh"]


FROM python:3.8.1
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
COPY ./*.py /exp/
COPY ./*.sh /exp/
WORKDIR /exp
ENTRYPOINT ["/exp/run.sh"]