"""
Utility function for preprocessing the image before feeding them into neural
network
"""

import cv2
import imutils
from keras.preprocessing.image import img_to_array


def preprocessing_function(image):
    """Preprocess image by converting image to array and normalizing

    Arguments:
        image {tensor} -- Image numpy tensor with rank 3

    Returns:
        tensor -- Processed image numpy tensor with same rank as input
    """
    image_array = img_to_array(image)
    image_array /= 255.0

    return image_array


def process_sample(image_path, depth=3):
    """
    Preprocess sample image for identifying featurewise center

    Arguments:
        image_path {string} -- Path of the sample image

    Keyword Arguments:
        depth {int} -- number of image channels (default: {3})

    Returns:
        tensor -- Processed image numpy tensor
    """

    image = cv2.imread(image_path)

    if depth == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = preprocessing_function(image)
    return image


def preprocess(image, inner_width=180, inner_height=180, width=224, height=224):
    """
    Pads given image, applies threshold and resizes it to given width and height

    Arguments:
        image {tensor} -- Image numpy tensor with rank 3


    Keyword Arguments:
        inner_width {int} -- width of the given image after processing (default: {180})
        inner_height {int} -- height of the given image after processing (default: {180})
        width {int} -- width of the image after padding (default: {224})
        height {int} -- height of the image after padding (default: {224})

    Returns:
        tensor -- Processed image numpy tensor with same rank as input
    """

    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    original_height, original_width = image.shape[:2]

    # Resize image to square
    if original_width > original_height:
        image = imutils.resize(image, width=inner_width)
    else:
        image = imutils.resize(image, height=inner_height)

    # Calculate the padding values for width and height
    width_padding = int((width - image.shape[1]) / 2.0)
    height_padding = int((height - image.shape[0]) / 2.0)

    # Apply threshold to convert image to true black/white
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Add padding to image
    bordered = cv2.copyMakeBorder(thresh,
                                  height_padding, height_padding,
                                  width_padding, width_padding,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Revert image to RGB
    colored = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

    # Resize image to given width and height
    resized = cv2.resize(colored, (width, height))

    return resized
