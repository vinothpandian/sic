"""
Train Metamorph neural network
"""
from nn.metamorph import MetaMorph
from utils.validate_config import validate
from utils.preprocessing import (preprocessing_function,
                                 process_sample)
from utils.plot_confusion_matrix import confusion_matrix_analysis
from callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import json
import os
import sys
import shutil

import matplotlib
matplotlib.use("Agg")


# Disable annoying error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################################################################################
# Arguments for setting parameters while running array batch job
###################################################################################################

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-c", "--config_file", type=str, required=True,
                    help="MetaMorph configuration file path")
PARSER.add_argument("-e", "--epochs", type=int, default=None,
                    help="MetaMorph configuration file path")


ARGS = vars(PARSER.parse_args())

CONFIG_FILE = ARGS["config_file"]

# Exit if error in config file
if not validate(CONFIG_FILE):
    print("Erroneous configuration file. Please recheck")
    sys.exit(0)

###################################################################################################

# Load config file
with open(CONFIG_FILE, "r") as json_file:
    JSON_DATA = json_file.read()

CONFIG = json.loads(JSON_DATA)

# Training mode - Backend or classifier
TRAINING_MODE = CONFIG["training mode"]

PRETRAINED_MODEL = CONFIG["pre-trained model"]

# Dataset folder information
DATASET_INFORMATION = CONFIG["dataset information"]

TRAIN_FOLDER = DATASET_INFORMATION["train folder"]
DEV_FOLDER = DATASET_INFORMATION["dev folder"]
TEST_FOLDER = DATASET_INFORMATION["test folder"]
SAMPLE_FOLDER = DATASET_INFORMATION["samples folder"]
LABELS_FILE = DATASET_INFORMATION["labels file"]

CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
PID = os.getpid()

OUTPUT_FOLDERNAME = f'{TRAINING_MODE.capitalize()}-{CURRENT_TIMESTAMP}-{PID}'

OUTPUT_FOLDER = os.path.join(
    DATASET_INFORMATION["output folder"], OUTPUT_FOLDERNAME)
WEIGHTS_FOLDER = os.path.join(OUTPUT_FOLDER, "weights")
LOGS_FOLDER = os.path.join(OUTPUT_FOLDER, "logs")

# Create output directories
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)


# Image augmentation parameters
IMAGE_AUGMENTATION = CONFIG["image augmentation"]

WIDTH = IMAGE_AUGMENTATION["size"]
HEIGHT = IMAGE_AUGMENTATION["size"]
DEPTH = IMAGE_AUGMENTATION["depth"]
SHIFT = IMAGE_AUGMENTATION["shift"]
ROTATION = IMAGE_AUGMENTATION["rotation"]
VAL_AUG_FACTOR = IMAGE_AUGMENTATION["validation data augmentation factor"]

# Hyperparameters
HYPERPARAMETERS = CONFIG["hyperparameters"]

# Set epochs from args if set else from config file
EPOCHS = ARGS["epochs"] if ARGS["epochs"] else HYPERPARAMETERS["epochs"]

BATCH_SIZE = HYPERPARAMETERS["batch size"]
LEARNING_RATE = HYPERPARAMETERS["learning rate"]
DROP_EVERY = HYPERPARAMETERS["learning rate decay after x epoch"]
DROP_FACTOR = HYPERPARAMETERS["decay rate"]
MOMENTUM = HYPERPARAMETERS["momentum"]

# Image generator information
IMAGE_GENERATOR_INFORMATION = CONFIG["image generator information"]

NUM_OF_TRAINING_SAMPLES = IMAGE_GENERATOR_INFORMATION["num of training samples"]
NUM_OF_TEST_SAMPLES = IMAGE_GENERATOR_INFORMATION["num of test samples"]

# Set verbosity
VERBOSITY = CONFIG["verbosity"]

LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]


###################################################################################################
# print details of the model for log
###################################################################################################

print(80*"#")
print(f'MetaMorph training - {TRAINING_MODE.capitalize()} mode'.center(80))
print(CURRENT_TIMESTAMP.center(80))
print(f'Training results stored in {OUTPUT_FOLDERNAME}'.center(80))
print(80*"#")

PRETTY = PrettyTable()
PRETTY.field_names = ["Purpose", "Parameter", "Values"]
PRETTY.align = "l"

PRETTY.add_row(["Training Mode", "", TRAINING_MODE.capitalize()])
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
# Read labels from file and create Label binarizer
###################################################################################################

with open(LABELS_FILE, "r") as lf:
    LABELS = np.array([x[:-1] for x in lf.readlines()])

CLASSES = len(LABELS)

LB = LabelBinarizer().fit(LABELS)


###################################################################################################
#  Create data generator to augment images for training and validation
###################################################################################################

TRAINING_DATA_GENERATOR = ImageDataGenerator(featurewise_center=True, rotation_range=ROTATION,
                                             width_shift_range=SHIFT, height_shift_range=SHIFT,
                                             preprocessing_function=preprocessing_function)


VALIDATION_DATA_GENERATOR = ImageDataGenerator(featurewise_center=True,
                                               rotation_range=ROTATION *
                                               (1+VAL_AUG_FACTOR),
                                               width_shift_range=SHIFT *
                                               (1+VAL_AUG_FACTOR),
                                               height_shift_range=SHIFT *
                                               (1+VAL_AUG_FACTOR),
                                               preprocessing_function=preprocessing_function)

TEST_DATA_GENERATOR = ImageDataGenerator(
    preprocessing_function=preprocessing_function)

# Load sample images to fit image generator for identifying featurewise center
SAMPLES = [process_sample(image_path, depth=DEPTH)
           for image_path in paths.list_images(SAMPLE_FOLDER)]

TRAINING_DATA_GENERATOR.fit(SAMPLES)
VALIDATION_DATA_GENERATOR.fit(SAMPLES)

COLOR_MODE = "grayscale" if DEPTH == 1 else "rgb"

print("[INFO] Creating training data generator")
TRAINING_DATA = TRAINING_DATA_GENERATOR.flow_from_directory(directory=TRAIN_FOLDER,
                                                            color_mode=COLOR_MODE,
                                                            target_size=(
                                                                WIDTH, HEIGHT),
                                                            batch_size=BATCH_SIZE)

print("[INFO] Creating validation data generator")
VALIDATION_DATA = VALIDATION_DATA_GENERATOR.flow_from_directory(directory=DEV_FOLDER,
                                                                color_mode=COLOR_MODE,
                                                                target_size=(
                                                                    WIDTH, HEIGHT),
                                                                batch_size=BATCH_SIZE)

print("[INFO] Creating test data generator")
TEST_DATA = TEST_DATA_GENERATOR.flow_from_directory(directory=TEST_FOLDER,
                                                    color_mode=COLOR_MODE,
                                                    target_size=(
                                                        WIDTH, HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)


###################################################################################################
# Compile MetaMorph model
###################################################################################################

print("[INFO] Compiling model....")
METAMORPH = MetaMorph()

if PRETRAINED_MODEL != "":
    print("[INFO] Loading pre-trained model")
    MODEL = load_model(PRETRAINED_MODEL)
else:
    if TRAINING_MODE == "backend":
        print("[INFO] Backend training mode")
        MODEL = METAMORPH.backend(width=WIDTH,
                                  height=HEIGHT,
                                  depth=DEPTH,
                                  classes=CLASSES)
    else:
        print("[INFO] Classifier training mode")
        MODEL = METAMORPH.classifier(width=WIDTH,
                                     height=HEIGHT,
                                     depth=DEPTH,
                                     classes=CLASSES)


OPTIMISER = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)

MODEL.compile(loss=LOSS, optimizer=OPTIMISER, metrics=METRICS)

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
TRAINING_MONITOR = TrainingMonitor(FIGURE_PATH, jsonPath=JSON_PATH)


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
CHECKPOINT = ModelCheckpoint(WEIGHT_NAME, monitor="val_loss", mode="min",
                             save_best_only=True, verbose=1)

EARLY_STOP = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=5,
                           mode='min',
                           verbose=1)

TENSORBOARD = TensorBoard(log_dir=LOGS_FOLDER,
                          histogram_freq=0,
                          # write_batch_performance=True,
                          write_graph=True,
                          write_images=False)

CALLBACKS = [EARLY_STOP, TRAINING_MONITOR, DECAY, CHECKPOINT, TENSORBOARD]


###################################################################################################
# Train the model
###################################################################################################

print("[INFO] Training the model....")
HISTORY = MODEL.fit_generator(generator=TRAINING_DATA,
                              steps_per_epoch=NUM_OF_TRAINING_SAMPLES//BATCH_SIZE,
                              epochs=EPOCHS,
                              callbacks=CALLBACKS,
                              validation_data=VALIDATION_DATA,
                              validation_steps=NUM_OF_TEST_SAMPLES//BATCH_SIZE,
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
                                      steps=len(TEST_DATA.classes)//BATCH_SIZE+1)
Y_PREDICTIONS = np.argmax(PREDICTIONS, axis=1)

CONFUSION_MATRIX_FILENAME = os.path.join(OUTPUT_FOLDER, "confusion_matrix")

confusion_matrix_analysis(y_true=TEST_DATA.classes,
                          y_predicted=Y_PREDICTIONS,
                          filename=CONFUSION_MATRIX_FILENAME,
                          labels=np.arange(CLASSES),
                          y_map=LB.classes_)

CLASSIFICATION_REPORT = classification_report(TEST_DATA.classes,
                                              Y_PREDICTIONS,
                                              target_names=LB.classes_)
print(CLASSIFICATION_REPORT)


ACCURACY = HISTORY.history["acc"][-1] * 100
VALIDATION_ACCURACY = HISTORY.history["val_acc"][-1] * 100
LOSS = HISTORY.history["loss"][-1]
VALIDATION_LOSS = HISTORY.history["val_loss"][-1]

REPORT = [
    80*"#",
    "\n",
    "REPORT".center(80),
    f'MetaMorph NN training : {TRAINING_MODE.capitalize()} mode'.center(80),
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
