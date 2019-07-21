"""
import default models from keras
"""

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19

from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


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

    def vgg(self):
        base_model = VGG19(include_top=False,
                           weights="imagenet",
                           input_shape=(self.height, self.width, self.depth))

        X = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, output=x)
        return model
