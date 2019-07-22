"""
Train Metamorph neural network
# https://transfer.sh/12Z5E1/weights.hdf5
"""
import os

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.applications.vgg19 import preprocess_input as vgg_preprocess_input
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint)
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.applications.vgg19 import VGG19

from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model, Sequential

from keras.utils.data_utils import get_file


ImageFile.LOAD_TRUNCATED_IMAGES = True


###################################################################################################
# Arguments for setting parameters while running array batch job
###################################################################################################


NAME = "VGG19"

# Set verbosity
VERBOSITY = 1

# Set model configuration
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]

# Dataset folder information
TRAINING_CSV = "../input/training-labels.csv"
DATASET_FOLDER = "../input/train/output_combined2"
TESTSET_FOLDER = "../input/test/Test"


WEIGHTS_FOLDER = "."


WEIGHTS_PATH = get_file('pretrained_weights',
                        'https://transfer.sh/12Z5E1/weights.hdf5')

# Image augmentation parameters

HEIGHT = 256
WIDTH = 256
DEPTH = 3
SHIFT = 20.0
ROTATION = 10.0
VAL_AUG_FACTOR = 0.2

# Hyperparameters
# Set epochs from args if set else from config file
EPOCHS = 16

BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROP_EVERY = 12
DROP_FACTOR = 0.25
MOMENTUM = 0.9

# Image generator information
VALIDATION_SPLIT = 0.1

##################################################################################################
# Read details from CSV
###################################################################################################

DATASET = pd.read_csv(TRAINING_CSV, dtype=str)
TRAIN, VALIDATION = train_test_split(DATASET, test_size=VALIDATION_SPLIT)

TESTSET_ARRAY = [[filename, "0"]
                 for filename in os.listdir(TESTSET_FOLDER)]
TESTSET = pd.DataFrame(TESTSET_ARRAY, columns=["Id", "Expected"])


###################################################################################################
#  Create data generator to augment images for training and validation
###################################################################################################

preprocessing_function = vgg_preprocess_input

TRAINING_DATA_GENERATOR = ImageDataGenerator(rotation_range=ROTATION,
                                             width_shift_range=SHIFT,
                                             height_shift_range=SHIFT,
                                             preprocessing_function=preprocessing_function)

VALIDATION_DATA_GENERATOR = ImageDataGenerator(rotation_range=ROTATION *
                                               (1+VAL_AUG_FACTOR),
                                               width_shift_range=SHIFT *
                                               (1+VAL_AUG_FACTOR),
                                               height_shift_range=SHIFT *
                                               (1+VAL_AUG_FACTOR),
                                               preprocessing_function=preprocessing_function)

TEST_DATA_GENERATOR = ImageDataGenerator(
    preprocessing_function=preprocessing_function)

COLOR_MODE = "grayscale" if DEPTH == 1 else "rgb"

print("[INFO] Creating training data generator")
TRAINING_DATA = TRAINING_DATA_GENERATOR.flow_from_dataframe(dataframe=TRAIN,
                                                            directory=DATASET_FOLDER,
                                                            x_col="Filename",
                                                            y_col="Drscore",
                                                            class_mode="categorical",
                                                            color_mode=COLOR_MODE,
                                                            target_size=(
                                                                WIDTH, HEIGHT),
                                                            batch_size=BATCH_SIZE)

print("[INFO] Creating validation data generator")
VALIDATION_DATA = VALIDATION_DATA_GENERATOR.flow_from_dataframe(dataframe=VALIDATION,
                                                                directory=DATASET_FOLDER,
                                                                x_col="Filename",
                                                                y_col="Drscore",
                                                                class_mode="categorical",
                                                                color_mode=COLOR_MODE,
                                                                target_size=(
                                                                    WIDTH, HEIGHT),
                                                                batch_size=BATCH_SIZE)

print("[INFO] Creating test data generator")
TEST_DATA = TEST_DATA_GENERATOR.flow_from_dataframe(dataframe=TESTSET,
                                                    directory=TESTSET_FOLDER,
                                                    x_col="Id",
                                                    y_col="Expected",
                                                    class_mode="categorical",
                                                    target_size=(
                                                        WIDTH, HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)

NUM_OF_TRAINING_SAMPLES = len(TRAIN)
NUM_OF_VALIDATION_SAMPLES = len(VALIDATION)
NUM_OF_TEST_SAMPLES = len(TEST_DATA.classes)//BATCH_SIZE+1
CLASSES = 5


###################################################################################################
# Cohen Kappa metrics
###################################################################################################


def cohen_kappa(y_true, y_pred):
    y_true_classes = tf.argmax(y_true, 1)
    y_pred_classes = tf.argmax(y_pred, 1)
    return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, CLASSES)[1]


###################################################################################################
# Compile MetaMorph model
###################################################################################################

BASE_MODEL = VGG19(include_top=False, input_shape=(HEIGHT, WIDTH, DEPTH))
MODEL = Sequential()
MODEL.add(BASE_MODEL)
MODEL.add(GlobalAveragePooling2D())
MODEL.add(Dense(1024, activation='relu'))
MODEL.add(Dense(512, activation='relu'))
MODEL.add(Dense(CLASSES, activation='softmax'))

MODEL.load_weights(WEIGHTS_PATH)

OPTIMISER = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)

MODEL.compile(loss=LOSS, optimizer=OPTIMISER, metrics=[*METRICS, cohen_kappa])

K.get_session().run(tf.local_variables_initializer())

###################################################################################################
# Define callbacks
###################################################################################################


def step_decay(epoch):
    """Generate step decay callback function

    Arguments:
        epoch {int} -- Current epoch value

    Returns:
        float -- Updated learning rate alpha after decay
    """
    alpha = LEARNING_RATE * (DROP_FACTOR ** np.floor((1 + epoch) / DROP_EVERY))
    return float(alpha)


# Learning rate decay in steps
DECAY = LearningRateScheduler(step_decay)

# Checkpoint model callback
WEIGHT_NAME = "./weights.hdf5"
CHECKPOINT = ModelCheckpoint(WEIGHT_NAME, monitor="val_cohen_kappa", mode="max",
                             save_best_only=True, verbose=1)

EARLY_STOP = EarlyStopping(monitor='val_cohen_kappa',
                           min_delta=0.001,
                           patience=5,
                           mode='max',
                           verbose=1)

CALLBACKS = [EARLY_STOP, DECAY, CHECKPOINT]

###################################################################################################
# Train the model
###################################################################################################

print("[INFO] Training the model....")
HISTORY = MODEL.fit_generator(generator=TRAINING_DATA,
                              steps_per_epoch=NUM_OF_TRAINING_SAMPLES//BATCH_SIZE,
                              epochs=EPOCHS,
                              callbacks=CALLBACKS,
                              validation_data=VALIDATION_DATA,
                              validation_steps=NUM_OF_VALIDATION_SAMPLES//BATCH_SIZE,
                              verbose=VERBOSITY)


###################################################################################################
# Storing the model to output
###################################################################################################

print("[INFO] Storing trained model....")
MODEL.save("./trained_model.hdf5")
MODEL.save_weights("./trained_weights.hdf5")

###################################################################################################
# Evaluate the model and store the report and history log
###################################################################################################

print("[INFO] Evaluating the model....")
PREDICTIONS = MODEL.predict_generator(generator=TEST_DATA,
                                      steps=NUM_OF_TEST_SAMPLES,
                                      verbose=VERBOSITY)
Y_PREDICTIONS = np.argmax(PREDICTIONS, axis=1)

TEST_DATA

TESTSET["Expected"] = Y_PREDICTIONS
TESTSET.to_csv("Submission.csv", index=False)

ACCURACY = HISTORY.history["acc"][-1] * 100
VALIDATION_ACCURACY = HISTORY.history["val_acc"][-1] * 100
LOSS = HISTORY.history["loss"][-1]
COHEN_KAPPA = HISTORY.history["cohen_kappa"][-1]
VALIDATION_LOSS = HISTORY.history["val_loss"][-1]
VALIDATION_COHEN_KAPPA = HISTORY.history["val_cohen_kappa"][-1]

REPORT = [
    80*"#",
    "\n",
    "REPORT".center(80),
    f'Training with {NAME} config'.center(80),
    "\n",
    80*"#",
    "\n",
    "DETAILS",
    "\n",
    f'Accuracy: {ACCURACY:.2f}',
    f'Validation accuracy: {VALIDATION_ACCURACY:.2f}',
    f'Loss: {LOSS:.4f}',
    f'Validation Loss: {VALIDATION_LOSS:.4f}',
    f'COHEN_KAPPA: {COHEN_KAPPA:.4f}',
    f'VALIDATION_COHEN_KAPPA: {VALIDATION_COHEN_KAPPA:.4f}',
    "\n",
    80*"#",
]

for line in REPORT:
    print(line)

print("\n")
print("#"*80)
print("\n")
print("\n")
