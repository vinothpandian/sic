"""
Create processed dataset with train/dev/test split
"""

import argparse
import os

from metamorph_utils.dataset import create_dataset

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Automatically crop labelled UI sketch elements.')

    PARSER.add_argument("-i", "--input", required=True,
                        dest="input_folder",
                        help="Input folder containing labelled folders of UI sketches")
    PARSER.add_argument("-o", "--output", default=None,
                        dest="output_folder",
                        help="Output folder with processed images in train/dev/test folders")
    PARSER.add_argument("-l", "--label-file", default="labels.txt",
                        dest="label_file",
                        help="Labels list file")
    PARSER.add_argument("-d", "--dev-split", default=0.2, type=float,
                        dest="dev_split",
                        help="Dev split percentage (default: 0.2)")
    PARSER.add_argument("-t", "--test-split", default=0.2, type=float,
                        dest="test_split",
                        help="Test split percentage (default: 0.2)")
    PARSER.add_argument("-s", "--random-seed", default=42, type=int,
                        dest="seed",
                        help="Randomness seed value (default: 42)")

    ARGS = PARSER.parse_args()

    INPUT_FOLDER = ARGS.input_folder
    INPUT_FOLDER = INPUT_FOLDER.strip(os.sep)
    print(f'Input folder: {INPUT_FOLDER}')

    OUTPUT_FOLDER = ARGS.output_folder
    if not OUTPUT_FOLDER:
        OUTPUT_FOLDER = f"{INPUT_FOLDER}_processed"
    else:
        OUTPUT_FOLDER = OUTPUT_FOLDER.strip(os.sep)

    print(f'Output folder: {OUTPUT_FOLDER}')

    LABEL_FILE = ARGS.label_file
    LABEL_FILE = LABEL_FILE.strip(os.sep)
    print(f'Labels file: {LABEL_FILE}')

    DEV_SPLIT = ARGS.dev_split
    print(f'Dev split: {DEV_SPLIT}')

    TEST_SPLIT = ARGS.test_split
    print(f'Test split: {TEST_SPLIT}')

    SEED = ARGS.seed
    print(f'Random seed value: {SEED}')

    create_dataset(dataset_path=INPUT_FOLDER, output_path=OUTPUT_FOLDER, label_file=LABEL_FILE,
                   dev_split=DEV_SPLIT, test_split=TEST_SPLIT, random_state=SEED)
