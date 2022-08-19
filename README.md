## TODO: TITLE

### Overview

Build a scalable machine learning pipeline using Amazon SageMaker to train and deploy an object detection model for gigapixel-scale satellite imagery 
 
In this repository, we use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) to build, train, and deploy a custom `Faster-RCNN` (FRCNN) model for gigapixel-scale satellite imagery. The custom FRCNN model is built using [Detectron2](https://github.com/facebookresearch/detectron2), an open-source object detection library released by Meta AI Research. 

This repository shows how to do the following:

* Build Detectron2 Docker images and push them to [Amazon ECR](https://aws.amazon.com/ecr/) to run training and inference jobs on Amazon SageMaker.
* Run a SageMaker Training job to finetune pre-trained model weights on a custom dataset.
* TODO

### Get Started

### Instructions

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the Apache 2.0 License. See the LICENSE file.


