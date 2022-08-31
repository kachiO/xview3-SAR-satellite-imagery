## xView3 SAR - Object detection on giga-pixel satellite imagery

### Overview

Build a scalable machine learning pipeline using Amazon SageMaker to train and deploy an object detection model for gigapixel-scale satellite imagery 
 
In this repository, we use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) to build, train, and deploy a custom `Faster-RCNN` model for gigapixel-scale satellite imagery. The custom FRCNN model is built using [Detectron2](https://github.com/facebookresearch/detectron2), an open-source object detection library released by Meta AI Research. 

This repository shows how to do the following:

* Build Docker images and push them to [Amazon ECR](https://aws.amazon.com/ecr/) to run Amazon SageMaker jobs for processing, training, and runnng inference on the xView3 SAR dataset. 
* Run Amazon SageMaker Training job to preprocess xView3 SAR satellite imagery for ob
* Customize the FRCNN model forRun Amazon SageMaker Training job to finetune pre-trained model weights on xView3 SAR satellite imagery.

### Installation Instructions
- create virtual environment:
```
python3 -m venv venv-xview3
source venv-xview3/bin/activate
python3 -m pip install --upgrade pip
```
- install dependencies:
```
pip install -r requirements_cpu.txt
``` 
OR 
```
pip install -r requirements_gpu.txt
```

- install detectron2:
```
pip install git+https://github.com/facebookresearch/detectron2.git
```

- install repository as package:
```
pip install -e
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the Apache 2.0 License. See the LICENSE file.


