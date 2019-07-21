"""
Train Metamorph neural network
"""
import argparse
import datetime
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from prettytable import PrettyTable
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from metamorph_utils.plot_confusion_matrix import confusion_matrix_analysis
from metamorph_utils.preprocessing import preprocessing_function
from metamorph_utils.validate_config import validate

# Disable annoying error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################################################################################
# Arguments for setting parameters while running array batch job
###################################################################################################

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-c", "--config_file", type=str, required=True,
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

TEST_FOLDER = DATASET_INFORMATION["test folder"]
LABELS_FILE = DATASET_INFORMATION["labels file"]

CURRENT_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
PID = os.getpid()

OUTPUT_FOLDERNAME = f'Metrics-{TRAINING_MODE.capitalize()}-{CURRENT_TIMESTAMP}-{PID}'

OUTPUT_FOLDER = os.path.join(DATASET_INFORMATION["output folder"], OUTPUT_FOLDERNAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Image augmentation parameters
IMAGE_AUGMENTATION = CONFIG["image augmentation"]

WIDTH = IMAGE_AUGMENTATION["size"]
HEIGHT = IMAGE_AUGMENTATION["size"]
DEPTH = IMAGE_AUGMENTATION["depth"]

# Hyperparameters
HYPERPARAMETERS = CONFIG["hyperparameters"]

BATCH_SIZE = HYPERPARAMETERS["batch size"]

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


TEST_DATA_GENERATOR = ImageDataGenerator(preprocessing_function=preprocessing_function)

COLOR_MODE = "grayscale" if DEPTH == 1 else "rgb"

print("[INFO] Creating test data generator")
TEST_DATA = TEST_DATA_GENERATOR.flow_from_directory(directory=TEST_FOLDER, color_mode=COLOR_MODE,
                                                    target_size=(WIDTH, HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=False)


###################################################################################################
# Compile MetaMorph model
###################################################################################################

print("[INFO] Compiling model....")
MODEL = load_model(PRETRAINED_MODEL)

print(80*"#")
print("\n")
print(MODEL.name.center(80))
print("\n")
print(80*"#")
MODEL.summary()


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


REPORT = [
    80*"#",
    "\n",
    "REPORT".center(80),
    f'MetaMorph NN metrics : {TRAINING_MODE.capitalize()} mode'.center(80),
    f'Model name: {MODEL.name}'.center(80),
    f'Time: {CURRENT_TIMESTAMP}'.center(80),
    f'Metrics results stored in {OUTPUT_FOLDERNAME}'.center(80),
    "\n",
    80*"#",
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

FILENAME = f'REPORT.txt'

print("[INFO] Storing the evaluation results....")
with open(os.path.join(OUTPUT_FOLDER, FILENAME), "w") as eval_result:
    eval_result.write("\n".join(REPORT))
