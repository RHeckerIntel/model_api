#!/usr/bin/env python3

import sys

from model_api import ClassificationModel
import cv2

def main():
    if len(sys.argv) != 3:
        raise RuntimeError(f"Usage: {sys.argv[0]} <path_to_model> <path_to_image>")


    image = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2RGB)
    if image is None:
        raise RuntimeError("Failed to read the image")

    model = ClassificationModel.create_model(sys.argv[1])
    result = model(image)

if __name__ == "__main__":
    main()
