ARG BASE_IMAGE
FROM ${BASE_IMAGE}


WORKDIR /opt/ml/code

COPY tools/train_net.py /opt/ml/code/
COPY configs/xview3 /opt/ml/code/configs
COPY src /opt/ml/code/src
RUN python3 -m pip install /opt/ml/code/src

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM train_net.py