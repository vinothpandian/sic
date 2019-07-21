"""
Process given labelled dataset and create train/test/dev split
"""

import os
import shutil
from os import path

import cv2
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess


def create_folder_structure(input_folder, output_folder):
    """
    Create folder structure in output folder similar to input folder

    Arguments:
        input_folder {str} -- Path to the input folder
        output_folder {str} -- Path to the output folder
    """
    os.makedirs(output_folder, exist_ok=True)
    directories = os.listdir(input_folder)

    for directory in directories:
        os.makedirs(os.path.join(output_folder, directory), exist_ok=True)


def process_batch(x_batch, batch_path, dataset_path):
    """
    Process images in a given batch

    Arguments:
        x_batch {list} -- Array of image paths in the batch
        batch_path {string} -- Path to the batch folder
        dataset_path {string} -- Path to the dataset folder
    """
    for image_path in x_batch:
        out_path = image_path.replace(dataset_path, batch_path)
        image = cv2.imread(image_path)
        image = preprocess(image)
        cv2.imwrite(out_path, image)


def copy_samples(sample_images, batch_path, sample_path, dataset_path):
    """
    Copy samples from batches

    Arguments:
        sample_images {list} -- Array of image paths of samples
        batch_path {string} -- Path to the batch folder
        sample_path {string} -- Path to the sample folder
        dataset_path {string} -- Path to the dataset folder
    """
    for image_path in sample_images:
        processed_image_path = image_path.replace(dataset_path, batch_path)
        filename = path.basename(processed_image_path)
        sample_image_path = path.join(sample_path, filename)
        shutil.copy(processed_image_path, sample_image_path)


def create_dataset(dataset_path, output_path=None, label_file="labels.txt",
                   dev_split=0.2, test_split=0.2, random_state=42):
    """
    Process given labelled dataset and create train/test/dev split in output
    directory

    Arguments:
        dataset_path {str} -- Path to the dataset folder

    Keyword Arguments:
        output_path {str} -- Path to the output folder (default: {None})
        label_file {str} -- Path to the label file (default: {labels.txt})
        dev_split {float} -- Fraction of dev split. 0.2 == 20% (default: {0.2})
        test_split {float} -- Fraction of test split. 0.2 == 20% (default: {0.2})
        random_state {int} -- Random seed for split (default: {42})
    """

    dev_split = round(dev_split, 1)
    test_split = round(test_split, 1)
    train_split = round(1 - dev_split - test_split, 1)

    if train_split <= 0:
        print("Please check the train/dev/test split.")
        return

    print(f"Dataset will be split into {train_split}% train, {dev_split}% dev, {test_split}% test")

    print("Loading images...")

    dataset_path = dataset_path.strip(os.sep)

    # Create folder names for output if not defined and train/dev/test folders
    if not output_path:
        output_path = f"{dataset_path}_processed"
    else:
        output_path = output_path.strip(os.sep)

    train_path = path.join(output_path, "train")
    dev_path = path.join(output_path, "dev")
    test_path = path.join(output_path, "test")

    os.makedirs(output_path, exist_ok=True)

    # Create folder structure similar to dataset in train/dev/test folders
    create_folder_structure(dataset_path, train_path)
    create_folder_structure(dataset_path, test_path)
    create_folder_structure(dataset_path, dev_path)

    data = []
    labels = []

    for image_path in paths.list_images(dataset_path):
        data.append(image_path)
        labels.append(image_path.split(os.path.sep)[-2])

    print(f"{len(data)} images found")

    # Create samples for finding featurewise mean
    sample_size = int(len(data)*0.25)
    print(f"Approx {sample_size} images will be stored as sample")

    sample_path = path.join(output_path, "samples")
    os.makedirs(sample_path, exist_ok=True)

    x_train, x_test, y_train, _ = train_test_split(data, labels,
                                                   test_size=test_split, random_state=random_state)
    x_train, x_dev, y_train, _ = train_test_split(x_train, y_train,
                                                  test_size=dev_split, random_state=random_state)

    print("Processing images...")

    process_batch(x_train, train_path, dataset_path)
    sample_images = np.random.choice(x_train, int(sample_size*train_split), replace=False)
    copy_samples(sample_images, train_path, sample_path, dataset_path)

    process_batch(x_dev, dev_path, dataset_path)
    sample_images = np.random.choice(x_dev, int(sample_size*dev_split), replace=False)
    copy_samples(sample_images, dev_path, sample_path, dataset_path)

    process_batch(x_test, test_path, dataset_path)
    sample_images = np.random.choice(x_test, int(sample_size*test_split), replace=False)
    copy_samples(sample_images, test_path, sample_path, dataset_path)

    label_list = np.sort(np.unique(labels))

    print("Creating labels file...")
    with open(label_file, "w") as file:
        for label in label_list:
            print(label, file=file)
    print(f"Labels file created at {label_file}...")

    print(f"Processed dataset with train/test/dev split created at {output_path}")
    print("#"*80)
    print(f"Train path: {train_path}")
    print(f"Dev path: {dev_path}")
    print(f"Test path: {test_path}")
    print(f"Samples path: {sample_path}")
