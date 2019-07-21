"""
import default models from keras
"""

from keras.applications.resnet50 import Resnet50


class Models:

    def __init__(self, height, width, depth, num_classes):
        self.height = height
        self.width = width
        self.depth = depth
        self.num_classes = num_classes

    def resnet50(self):
        model = Resnet50(inclu)
