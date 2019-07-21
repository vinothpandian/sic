"""
import default models from keras
"""

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3


class Models:

    def __init__(self, height, width, depth, classes):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes

    def resnet50(self):
        model = ResNet50(include_top=True,
                         weights=None,
                         input_shape=(self.height, self.width, self.depth),
                         classes=self.classes)
        return model

    def inception(self):
        model = InceptionV3(include_top=True,
                            weights=None,
                            input_shape=(self.height, self.width, self.depth),
                            classes=self.classes)

        return model
