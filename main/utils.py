"""Utility file for views"""
import base64
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model


def init():
    """Loads the model from directory"""
    model = load_model('model.h5', compile=True)
    return model


def getimage(request):
    """
    Gets the image from front-end as base-64
    decodes base-64 into bytes data then converts
    it to an image and saves the image on directory.

    :param request: incoming image via POST method
    :return: N/A
    """
    img_data = request.POST.get('img')
    base64_data = re.sub('^data:image/.+;base64,', '', img_data)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    img.save('output.png')


def png2jpeg(rgba, background=(255, 255, 255)):
    """
    Takes a png image and converts it into a jpeg image.
    :param rgba: image in rgba format
    :param background: value to turn the background to 255,255,255 is white
    :return: rgb image
    """
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')


def detect_numbers():
    """
    Reads image from directory detects numbers on the image by using
    contours. Draws a counding box around the numbers and crops them.
    Rescales the image caredully so it doesnt lose information.
    Pastes the detected numbers on 400x400 empty image separately
    and dialted the image using morphological process, to make the number
    more rich in detail.
    :return: Image with detected numbers, list of images of detected numbers.
    """
    img = cv2.imread('output.png', cv2.IMREAD_UNCHANGED)
    img = png2jpeg(img)
    img = np.invert(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img_copy = gray.copy()

    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # Detect the contours in the image
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    to_remove = []

    """Finding inner contour"""
    count = 0
    for i in heirarchy[0]:
        if i[3] != -1:
            to_remove.append(count)
        count += 1

    """Deleting Contour"""
    for i in reversed(to_remove):
        del contours[i]

    # Iterate through all the contours
    detected_numbers = []
    for contour in contours:
        # Find bounding rectangles
        x, y, w, h = cv2.boundingRect(contour)
        # print(x, y, w, h, end="\n\n")

        """Cropping Number"""
        number = gray[y:y + h, x:x + w]
        number = cv2.resize(number, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        height, width = number.shape[:2]

        """Making number fit into 400x400"""
        if height > 400:
            number = cv2.resize(number, (width, 300), interpolation=cv2.INTER_CUBIC)
            height, width = number.shape[:2]
        if width > 400:
            number = cv2.resize(number, (300, height), interpolation=cv2.INTER_CUBIC)
            height, width = number.shape[:2]

        """Making empty image"""
        empty = np.zeros((400, 400))

        """Pasting the number onto empty image"""
        e_height, e_width = empty.shape[0], empty.shape[1]
        n_height, n_width = height, width
        row_to_place = int((e_height / 2) - (n_height / 2))
        col_to_place = int((e_width / 2) - (n_width / 2))
        empty[row_to_place:row_to_place + height, col_to_place:col_to_place + width] = number

        """Dilation of images"""
        kernel = np.ones((5, 5), np.uint8)
        empty = cv2.dilate(empty, kernel, iterations=5)

        """Appending detected numbers"""
        detected_numbers.append(empty)

        """Draw the rectangle"""
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return img_copy, detected_numbers
