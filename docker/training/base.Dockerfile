FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.9.1-gpu-py38-cu111-ubuntu20.04
LABEL author="kachio@amazon.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get -y install python3 python3-pip git python3-setuptools \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -U sagemaker
RUN pip install -U --upgrade torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -U --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
RUN pip install -U boto3==1.17.18 pandas rasterio zarr
RUN pip install -U --no-cache-dir pycocotools~=2.0.0


ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta"

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"

