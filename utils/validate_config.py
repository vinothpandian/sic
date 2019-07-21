import json
from os import path

from jsonschema import ValidationError
from jsonschema import validate as validate_schema


def validate(config_file):
    """
    Validates the metamorph configuration file with the schema

    Arguments:
        config_file {string} -- Path to the metamorph configuration file

    Raises:
        NotADirectoryError -- Training folder given in config is not a directory
        NotADirectoryError -- Dev folder given in config is not a directory
        NotADirectoryError -- Test folder given in config is not a directory
        FileNotFoundError -- Labels file is not found
        NotADirectoryError -- Output folder given in config is not a directory

    Returns:
        boolean -- True if config file is valid else throws exceptions and
        return False
    """
    with open(config_file, "r") as json_file:
        json_data = json_file.read()

    data = json.loads(json_data)

    schema = {
        "type": "object",
        "required": ["training mode",
                     "pre-trained model",
                     "dataset information",
                     "image augmentation",
                     "hyperparameters",
                     "image generator information",
                     "verbosity"],
        "properties": {
            "training mode": {
                "type": "string",
                "pattern": "^(backend|classifier)$"
            },
            "pre-trained model": {
                "type": "string"
            },
            "dataset information": {
                "type": "object",
                "required": ["train folder",
                             "dev folder",
                             "test folder",
                             "samples folder",
                             "labels file",
                             "output folder"],
                "properties": {
                    "train folder": {
                        "type": "string"
                    },
                    "dev folder": {
                        "type": "string"
                    },
                    "test folder": {
                        "type": "string"
                    },
                    "samples folder": {
                        "type": "string"
                    },
                    "labels file": {
                        "type": "string"
                    },
                    "output folder": {
                        "type": "string"
                    }
                }
            },
            "image augmentation": {
                "type": "object",
                "required": ["size",
                             "depth",
                             "shift",
                             "rotation",
                             "validation data augmentation factor"],
                "properties": {
                    "size": {
                        "type": "integer"
                    },
                    "depth": {
                        "type": "integer"
                    },
                    "shift": {
                        "type": "number"
                    },
                    "rotation": {
                        "type": "number"
                    },
                    "validation data augmentation factor": {
                        "type": "number"
                    }
                }
            },
            "hyperparameters": {
                "type": "object",
                "required": ["epochs",
                             "batch size",
                             "learning rate",
                             "learning rate decay after x epoch",
                             "decay rate",
                             "momentum"],
                "properties": {
                    "epochs": {
                        "type": "integer"
                    },
                    "batch size": {
                        "type": "integer"
                    },
                    "learning rate": {
                        "type": "number"
                    },
                    "learning rate decay after x epoch": {
                        "type": "integer"
                    },
                    "decay rate": {
                        "type": "number"
                    },
                    "momentum": {
                        "type": "number"
                    }
                }
            },
            "image generator information": {
                "type": "object",
                "required": ["num of training samples",
                             "num of test samples", ],
                "properties": {
                    "num of training samples": {
                        "type": "integer"
                    },
                    "num of test samples": {
                        "type": "integer"
                    }
                }
            },
            "verbosity": {
                "type": "integer"
            }
        }
    }

    try:
        validate_schema(data, schema)

        pretrained_model = data["pre-trained model"]
        if not pretrained_model == "":
            if not path.isfile(pretrained_model) and not path.exists(pretrained_model):
                raise FileNotFoundError(pretrained_model)

        train_folder = data["dataset information"]["train folder"]
        if not path.isdir(train_folder) and not path.exists(train_folder):
            raise NotADirectoryError(train_folder)

        dev_folder = data["dataset information"]["dev folder"]
        if not path.isdir(dev_folder) and not path.exists(dev_folder):
            raise NotADirectoryError(dev_folder)

        test_folder = data["dataset information"]["test folder"]
        if not path.isdir(test_folder) and not path.exists(test_folder):
            raise NotADirectoryError(test_folder)

        samples_folder = data["dataset information"]["samples folder"]
        if not path.isdir(samples_folder) and not path.exists(samples_folder):
            raise NotADirectoryError(samples_folder)

        label_file = data["dataset information"]["labels file"]
        if not path.isfile(label_file) and not path.exists(label_file):
            raise FileNotFoundError(label_file)

        output_folder = data["dataset information"]["output folder"]
        if not path.isdir(output_folder) and not path.exists(output_folder):
            raise NotADirectoryError(output_folder)

    except NotADirectoryError as error:
        print("Either this path is not a directory or does not exist")
        print(error)
    except FileNotFoundError as error:
        print(f"{error} file is missing")
    except ValidationError as error:
        print("Error in configuration file")
        print("#"*80)
        print(error.message)
    else:
        return True

    return False


if __name__ == "__main__":
    print(validate("./config.json"))
