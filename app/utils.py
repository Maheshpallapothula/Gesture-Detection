import cv2
import numpy as np

def load_image(image_data: bytes):
    """
    Load an image from raw bytes.

    :param image_data: Image data in bytes.
    :return: OpenCV image.
    """
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
