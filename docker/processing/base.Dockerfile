FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get -y install python3 python3-pip vim nano git

COPY requirements_cpu.txt .
RUN pip3 install -r requirements_cpu.txt
RUN python3 -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

WORKDIR /opt/ml/code

# Make sure python doesn't buffer stdout so we get logs ASAP.
ENV PYTHONUNBUFFERED=TRUE

