import cv2
import numpy as np
import pygame
import torch
from cognite.client.data_classes import FileMetadata
from djitellopy import Tello
from dotenv import load_dotenv

from brain_ntnu_workshop_ai_2022.solution.object_detection import get_fasterrcnn_model, get_ssdlite_model

load_dotenv()


def main() -> None:
    Tello()
    get_fasterrcnn_model()
    get_ssdlite_model()
    lst = [pygame.KEYUP, cv2.INTER_MAX, np.zeros(1), torch.Tensor(), FileMetadata(name="test")]
    len(lst)
    print("You are good to go!")


if __name__ == "__main__":
    main()
