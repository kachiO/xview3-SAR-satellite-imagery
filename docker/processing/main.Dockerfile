
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /opt/ml/code

COPY src /opt/ml/code/src
RUN pip install /opt/ml/code/src

COPY tools/ /opt/ml/code/
