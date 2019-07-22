"""
import default models from keras
"""

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169
from keras.applications.nasnet import NASNetLarge
from keras.applications.mobilenet_v2 import MobileNetV2

from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model, Sequential


class Models:

    def __init__(self, height, width, depth, classes):
        self.height = height
        self.width = width
        self.depth = depth
        self.classes = classes

    def resnet50(self):
        base_model = ResNet50(include_top=True,
                              weights="imagenet",
                              pooling="avg",
                              input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    def inception(self):
        base_model = InceptionV3(include_top=True,
                                 weights="imagenet",
                                 input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    def vgg(self):
        base_model = VGG19(include_top=False,
                           weights="imagenet",
                           input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model


    def vgg16(self):
        base_model = VGG16(include_top=False,
                           weights="imagenet",
                           input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    def densenet(self):
        base_model = DenseNet169(include_top=False,
                           weights="imagenet",
                           input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    def mobilenet(self):
        base_model = VGG16(include_top=False,
                           weights="imagenet",
                           input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model

    
    def NASNet(self):
        base_model = NASNetLarge(include_top=False,
                           weights="imagenet",
                           input_shape=(self.height, self.width, self.depth))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))

        return model