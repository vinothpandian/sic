"""
Train Metamorph neural network
"""
from utils.plot_confusion_matrix import confusion_matrix_analysis
from nn.models import Models
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable
from PIL import Image, ImageFile
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import argparse
import configparser
import datetime
import json
import os
import shutil
import sys

import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input as vgg_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input


ImageFile.LOAD_TRUNCATED_IMAGES = True

matplotlib.use("Agg")


# Disable annoying error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################################################################################
# Arguments for setting parameters while running array batch job
###################################################################################################

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-c", "--config_file", type=str, required=True,
                    help="Configuration file path")
PARSER.add_argument("-e", "--epochs", type=int, default=None,
                    help="Epochs")


ARGS = vars(PARSER.parse_args())

CONFIG_FILE = ARGS["config_file"]
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_FILE)


# Set verbosity
NAME = CONFIG["general"]["name"]
VERBOSITY = int(CONFIG["general"]["verbosity"])

# Set model configuration
MODEL_NAME = CONFIG["model"]["name"]
TRANSFER_LEARNING = CONFIG["model"].getboolean(
    "transfer_learning", fallback=True)
PRETRAINED_MODEL_PATH = CONFIG["model"]["pretrained_model_path"]
LOSS = CONFIG["model"]["loss"]
METRICS = CONFIG["model"]["metrics"].split(",")

# Dataset folder information
DATASET_INFORMATION = CONFIG["dataset_information"]

TRAINING_CSV = DATASET_INFORMATION["training_csv"]
DATASET_FOLDER = DATASET_INFORMATION["dataset_folder"]

# Disabled sample temporarily -
# add later for enhancements in featurewise center
# SAMPLE_FOLDER = DATASET_INFORMATION["samples folder"]

CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
PID = os.getpid()

OUTPUT_FOLDERNAME = f'{NAME}__{CURRENT_TIMESTAMP}__{PID}'

OUTPUT_FOLDER = os.path.join(
    DATASET_INFORMATION["output_folder"], OUTPUT_FOLDERNAME)
WEIGHTS_FOLDER = os.path.join(OUTPUT_FOLDER, "weights")
LOGS_FOLDER = os.path.join(OUTPUT_FOLDER, "logs")

# Create output directories
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)


# Image augmentation parameters
IMAGE_AUGMENTATION = CONFIG["image_augmentation"]

RESNET_HEIGHT, RESNET_WIDTH = 224, 224
INCEPTION_HEIGHT, INCEPTION_WIDTH = 299, 299
VGG_HEIGHT, VGG_WIDTH = 256, 256

HEIGHT = int(IMAGE_AUGMENTATION["height"])
WIDTH = int(IMAGE_AUGMENTATION["width"])

if TRANSFER_LEARNING:
    if MODEL_NAME == "resnet":
        HEIGHT, WIDTH = RESNET_HEIGHT, RESNET_WIDTH
    elif MODEL_NAME == "inception":
        HEIGHT, WIDTH = INCEPTION_HEIGHT, INCEPTION_WIDTH
    elif MODEL_NAME == "vgg":
        HEIGHT, WIDTH = VGG_HEIGHT, VGG_WIDTH
    else:
        HEIGHT, WIDTH = VGG_HEIGHT, VGG_WIDTH

DEPTH = int(IMAGE_AUGMENTATION["depth"])
SHIFT = float(IMAGE_AUGMENTATION["shift"])
ROTATION = float(IMAGE_AUGMENTATION["rotation"])
VAL_AUG_FACTOR = float(
    IMAGE_AUGMENTATION["validation_data_augmentation_factor"])

# Hyperparameters
HYPERPARAMETERS = CONFIG["hyperparameters"]

# Set epochs from args if set else from config file
EPOCHS = ARGS["epochs"] if ARGS["epochs"] else int(HYPERPARAMETERS["epochs"])

BATCH_SIZE = int(HYPERPARAMETERS["batch_size"])
LEARNING_RATE = float(HYPERPARAMETERS["learning_rate"])
DROP_EVERY = float(HYPERPARAMETERS["learning_rate_decay_after_x_epoch"])
DROP_FACTOR = float(HYPERPARAMETERS["decay_rate"])
MOMENTUM = float(HYPERPARAMETERS["momentum"])

# Image generator information
TRAIN_TEST_VAL_SPLIT = CONFIG["train_test_val_split"]

TEST_SPLIT = float(TRAIN_TEST_VAL_SPLIT["test_split"])
VALIDATION_SPLIT = float(TRAIN_TEST_VAL_SPLIT["validation_split"])


###################################################################################################
# print details of the model for log
###################################################################################################

print(80*"#")
print(f'Training - with {NAME} config file'.center(80))
print(CURRENT_TIMESTAMP.center(80))
print(f'Training results stored in {OUTPUT_FOLDERNAME}'.center(80))
print(80*"#")

PRETTY = PrettyTable()
PRETTY.field_names = ["Purpose", "Parameter", "Values"]
PRETTY.align = "l"
PRETTY.add_row(["Model", "Name", MODEL_NAME])
PRETTY.add_row(["", "", ""])
PRETTY.add_row(["Image Augmentation", "Shift", SHIFT])
PRETTY.add_row(["Image Augmentation", "Rotation", ROTATION])
PRETTY.add_row(
    ["Image Augmentation", "Factor in validation set", VAL_AUG_FACTOR])
PRETTY.add_row(["", "", ""])
PRETTY.add_row(["Training", "Epochs", EPOCHS])
PRETTY.add_row(["Training", "Batch size", BATCH_SIZE])
PRETTY.add_row(["", "", ""])
PRETTY.add_row(["Learning rate", "Initial learning rate", LEARNING_RATE])
PRETTY.add_row(["Learning rate", "Drop after every ", DROP_EVERY])
PRETTY.add_row(["Learning rate", "Drop factor ", DROP_FACTOR])
PRETTY.add_row(["Learning rate", "Momentum", MOMENTUM])

DETAILS = PRETTY.get_string(title='ConvNet details')
print(DETAILS)


###################################################################################################
# Read details from CSV
###################################################################################################

DATASET = pd.read_csv(TRAINING_CSV, dtype=str)
TRAIN_VALIDATION, TEST = train_test_split(DATASET, test_size=TEST_SPLIT)
TRAIN, VALIDATION = train_test_split(
    TRAIN_VALIDATION, test_size=VALIDATION_SPLIT)


###################################################################################################
#  Create data generator to augment images for training and validation
###################################################################################################

preprocessing_function = None

if TRANSFER_LEARNING:
    if MODEL_NAME == "resnet":
        preprocessing_function = resnet_preprocess_input
    elif MODEL_NAME == "inception":
        preprocessing_function = inception_preprocess_input
    elif MODEL_NAME == "vgg":
        preprocessing_function = vgg_preprocess_input
    else:
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

# Load sample images to fit image generator for identifying featurewise center
# SAMPLES = [process_sample(image_path, depth=DEPTH)
#            for image_path in paths.list_images(SAMPLE_FOLDER)]
# TRAINING_DATA_GENERATOR.fit(SAMPLES)
# VALIDATION_DATA_GENERATOR.fit(SAMPLES)

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
TEST_DATA = TEST_DATA_GENERATOR.flow_from_dataframe(dataframe=TEST,
                                                    directory=DATASET_FOLDER,
                                                    x_col="Filename",
                                                    y_col="Drscore",
                                                    class_mode="categorical",
                                                    target_size=(
                                                        WIDTH, HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)


NUM_OF_TRAINING_SAMPLES = 64  # len(TRAIN)
NUM_OF_VALIDATION_SAMPLES = 64  # len(VALIDATION)
NUM_OF_TEST_SAMPLES = len(TEST_DATA.classes)//BATCH_SIZE+1
CLASSES = len(DATASET["Drscore"].unique())


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
print("[INFO] Compiling model....")
Models = Models(height=HEIGHT, width=WIDTH, depth=DEPTH, classes=CLASSES)

if PRETRAINED_MODEL_PATH != "None":
    print("[INFO] Loading pre-trained model")
    MODEL = load_model(PRETRAINED_MODEL_PATH)
else:
    if MODEL_NAME == "resnet":
        MODEL = Models.resnet50()
    elif MODEL_NAME == "inception":
        MODEL = Models.inception()
    elif MODEL_NAME == "vgg":
        MODEL = Models.vgg()
    else:
        MODEL = Models.vgg()

OPTIMISER = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)

MODEL.compile(loss=LOSS, optimizer=OPTIMISER, metrics=[*METRICS, cohen_kappa])

K.get_session().run(tf.local_variables_initializer())

print(80*"#")
print("\n")
print(MODEL.name.center(80))
print("\n")
print(80*"#")
MODEL.summary()


###################################################################################################
# Define callbacks
###################################################################################################

# Training monitor
FIGURE_PATH = os.path.sep.join([LOGS_FOLDER, "{}.png".format(PID)])
JSON_PATH = os.path.sep.join([LOGS_FOLDER, "{}.json".format(PID)])


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
WEIGHT_NAME = os.path.join(WEIGHTS_FOLDER, "weights.hdf5")
CHECKPOINT = ModelCheckpoint(WEIGHT_NAME, monitor="val_cohen_kappa", mode="max",
                             save_best_only=True, verbose=1)

EARLY_STOP = EarlyStopping(monitor='val_cohen_kappa',
                           min_delta=0.001,
                           patience=5,
                           mode='max',
                           verbose=1)

TENSORBOARD = TensorBoard(log_dir=LOGS_FOLDER,
                          histogram_freq=0,
                          # write_batch_performance=True,
                          write_graph=True,
                          write_images=True)

CALLBACKS = [EARLY_STOP, DECAY, CHECKPOINT, TENSORBOARD]


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
MODEL.save(os.path.join(OUTPUT_FOLDER, "trained_model.hdf5"))
MODEL.save_weights(os.path.join(OUTPUT_FOLDER, "trained_weights.hdf5"))


###################################################################################################
# Evaluate the model and store the report and history log
###################################################################################################

print("[INFO] Evaluating the model....")
# Predict only on existing images - len(TEST_DATA.classes)
PREDICTIONS = MODEL.predict_generator(generator=TEST_DATA,
                                      steps=NUM_OF_TEST_SAMPLES,
                                      verbose=VERBOSITY)
Y_PREDICTIONS = np.argmax(PREDICTIONS, axis=1)

CONFUSION_MATRIX_FILENAME = os.path.join(OUTPUT_FOLDER, "confusion_matrix")

confusion_matrix_analysis(y_true=TEST_DATA.classes,
                          y_predicted=Y_PREDICTIONS,
                          filename=CONFUSION_MATRIX_FILENAME,
                          labels=[0, 1, 2, 3, 4])

CLASSIFICATION_REPORT = classification_report(TEST_DATA.classes,
                                              Y_PREDICTIONS)
print(CLASSIFICATION_REPORT)


ACCURACY = HISTORY.history["acc"][-1] * 100
VALIDATION_ACCURACY = HISTORY.history["val_acc"][-1] * 100
LOSS = HISTORY.history["loss"][-1]
VALIDATION_LOSS = HISTORY.history["val_loss"][-1]

REPORT = [
    80*"#",
    "\n",
    "REPORT".center(80),
    f'Training with {NAME} config'.center(80),
    f'Model used {MODEL_NAME}'.center(80),
    f'Config file : {CONFIG_FILE}'.center(80),
    f'Model name: {MODEL.name}'.center(80),
    f'Time: {CURRENT_TIMESTAMP}'.center(80),
    f'Training results stored in {OUTPUT_FOLDERNAME}'.center(80),
    "\n",
    80*"#",
    "\n",
    DETAILS,
    "\n",
    f'Accuracy: {ACCURACY:.2f}',
    f'Validation accuracy: {VALIDATION_ACCURACY:.2f}',
    f'Loss: {LOSS:.4f}',
    f'Validation Loss: {VALIDATION_LOSS:.4f}',
    "\n",
    80*"#",
    "\n",
    "Evaluation Metrics",
    "\n",
    "Classification report",
    CLASSIFICATION_REPORT,
    "\n",
    80*"#",
]

for line in REPORT:
    print(line)

REPORT_SHORTFORM = f'{ACCURACY:.0f}_{VALIDATION_ACCURACY:.0f}_{LOSS:.2f}_{VALIDATION_LOSS:.2f}'

FILENAME = f'REPORT__{REPORT_SHORTFORM}.txt'

print("[INFO] Storing the evaluation results....")
with open(os.path.join(OUTPUT_FOLDER, FILENAME), "w") as eval_result:
    eval_result.write("\n".join(REPORT))

shutil.copy(CONFIG_FILE, OUTPUT_FOLDER)
os.rename(OUTPUT_FOLDER, f'{OUTPUT_FOLDER}__{REPORT_SHORTFORM}')
print("[INFO] Training complete...")
