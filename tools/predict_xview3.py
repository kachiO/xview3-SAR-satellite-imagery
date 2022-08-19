"""Inference code for batch transform job"""

#import gzip
import json
import logging
import tarfile
import tempfile
from http.client import gzip
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np
import torch
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from xview3_d2.engine.defaults import DefaultPredictor
from xview3_d2.utils import extract_tarball, save_hdf5

logger = logging.getLogger("detectron2.inference")

    
def _load_tarball_and_extract1(request_body, destination:str='/opt/ml/input/'):
    with tarfile.open(fileobj=request_body.raw  , mode="r:gz") as  tarobj:
        tarobj.extractall(destination) 
        

def _load_tarball_and_extract2(request_body, destination:str='/opt/ml/input/'):
    fp = tempfile.NamedTemporaryFile()
    with gzip.GzipFile(fileobj=request_body) as d, open(fp, 'wb') as f:
        f.write(d.read())
    
    extract_tarball(fp.name, destination=destination, delete_tar=False)
        


def model_fn(model_dir: str) -> DefaultPredictor:
    r"""Load trained model

    Parameters
    ----------
    model_dir (str): location of the model directory

    Returns
    -------
    DefaultPredictor
        PyTorch model created by using Detectron2 API
    """
    cfg_file = list(Path(model_dir).rglob('*config.yaml*'))[0]
    assert cfg_file.is_file(), "Detectron2 config file not found!"
    model_file = list(Path(model_dir).rglob('*model_best.pth'))[0]
    assert model_file.is_file(), "Model file is missing!"
    
    logger.info(f"Using configuration specified in {str(cfg_file)}")
    logger.info(f"Using model saved at {str(model_file)}")
            
    with cfg_file.open('r') as f:
        cfg = CN().load_cfg(f)

    cfg.MODEL.WEIGHTS = str(model_file)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)


def input_fn(request_body: BinaryIO, request_content_type: str) -> np.ndarray:
    r"""Parse input data

    Parameters
    ----------
    request_body : BinaryIO
        encoded input image
    request_content_type : str
        type of content

    Returns
    -------
    np.ndarray
        input image

    Raises
    ------
    ValueError
        ValueError if the content type is not `application/x-image`
    """
    if request_content_type == "application/x-image":
        data = _load_tarball_and_extract1(request_body)
    else:
        err_msg = f"Type [{request_content_type}] not support this type yet"
        logger.error(err_msg)
        raise ValueError(err_msg)
    return data


def predict_fn(input_object: np.ndarray, predictor: DefaultPredictor) -> Mapping:
    r"""Run Detectron2 prediction

    Parameters
    ----------
    input_object : np.ndarray
        input image
    predictor : DefaultPredictor
        Detectron2 default predictor (see Detectron2 documentation for details)

    Returns
    -------
    Mapping
        a dictionary that contains: the image shape (`image_height`, `image_width`), the predicted
        bounding boxes in format x1y1x2y2 (`pred_boxes`), the confidence scores (`scores`), the
        labels associated with the bounding boxes (`pred_boxes`), and the predicted lengths (`pred_lengths`).
    """
    logger.info(f"Prediction on image of shape {input_object.shape}")
    outputs = predictor(input_object)
    fmt_out = {
        "image_height": input_object.shape[0],
        "image_width": input_object.shape[1],
        "pred_boxes": outputs["instances"].pred_boxes.tensor.tolist(),
        "scores": outputs["instances"].scores.tolist(),
        "pred_classes": outputs["instances"].pred_classes.tolist(),
        "pred_lengths": outputs["instances"].pred_lengths.tolist()
    }
    logger.info(f"Number of detected boxes: {len(fmt_out['pred_boxes'])}")
    return fmt_out


# pylint: disable=unused-argument
def output_fn(predictions, response_content_type):
    r"""Serialize the prediction result into the desired response content type"""
    return json.dumps(predictions)
