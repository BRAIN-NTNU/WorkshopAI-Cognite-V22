import glob
import logging
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import SSD, FasterRCNN, fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large

from brain_ntnu_workshop_ai_2022.utils import coco_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ssdlite_model() -> SSD:
    """
    Constructs an SSDlite model with input size 320x320 and a MobileNetV3 Large backbone.

    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_ and
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    <https://arxiv.org/abs/1801.04381>`_.

    This model will run much faster on your computer
    """
    return ssdlite320_mobilenet_v3_large(pretrained=True)


def get_fasterrcnn_model() -> FasterRCNN:
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    Reference: `"Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks" <https://arxiv.org/abs/1506.01497>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    Returns:
        model (FasterRCNN): A pre-trained Faster R-CNN model with a ResNet-50-FPN backbone.
    """
    return fasterrcnn_resnet50_fpn(pretrained=True)


def predict(model: Union[FasterRCNN, SSD], x: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """
    Make predictions on the input images using the model provided.
    Args:
        model (Union[FasterRCNN, SSD]): Model for making the predictions
        x (List[np.ndarray]): List of images as numpy arrays to predict

    Returns:
        predictions (List[Dict[str, np.ndarray]]): A list of dictionaries containing the bounding boxes,
        labels and scores.
    """

    # Set the model in evaluation mode. This is a kind of switch for some specific layers/parts of the model
    # that behave differently during training and inference (evaluating) time. For example,
    # Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval()
    # will do it for you.
    model.eval()

    # The model expects the input to be a list of torch.Tensor. Convert from numpy to tensor.
    x_tensor: List[torch.Tensor] = [torch.from_numpy(img).to(torch.float) for img in x]

    # Normalize images to be in the range 0-1. Max value of a pixel in an image is 255.
    # This is very important! If values are not in the range as the network was trained with, the predictions will
    # be very bad.
    x_tensor = [tensor / 255 for tensor in x_tensor]

    # The common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to
    # turn off gradients computation:
    with torch.no_grad():
        # TODO (Task 3): Fill in your code ########################################################
        ##################################################################################
        predictions: List[Dict[str, torch.Tensor]] = model(x_tensor)
        ##################################################################################
        predictions_np = [{key: value.numpy() for key, value in prediction.items()} for prediction in predictions]

    return predictions_np


def add_bounding_box_to_frame(frame: np.ndarray, box: Tuple[int, int, int, int], label: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    frame = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, coco_labels[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def main() -> None:
    # TODO (Task 5): Fill in your code ########################################################
    ##################################################################################
    # Make predictions on files stored in the data folder

    # List the files in the data folder
    files: List[str] = glob.glob("data/*.png")

    # Read files as numpy arrays
    images: List[np.ndarray] = [np.array(Image.open(file).convert("RGB")) for file in files]

    # Reshape all images to what the torchvision models expect [Channel, Height, Width].
    images = [np.moveaxis(image, -1, 0) for image in images]

    # Load model
    model = get_ssdlite_model()

    # Make predictions
    result: List[Dict[str, np.ndarray]] = predict(model, images)

    # Visualize one of the images
    pred = result[0]
    logger.info("Listing predictions for one image")
    for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
        if score >= 0.5:
            box = [int(v) for v in box]
            logger.info(f"{box}: {coco_labels[label]} ({round(score * 100, 2)} %)")
    ##################################################################################


if __name__ == "__main__":
    main()
