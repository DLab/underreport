FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

WORKDIR /underreport

RUN apt update -y
RUN apt upgrade -y
RUN apt install python3.8 python3.8-distutils python3.8-dev python3-pip wget  -y
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir /underreport/output

COPY underreport.py /underreport
COPY adjusted_cfr.csv /underreport