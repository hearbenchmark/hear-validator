#!/usr/bin/env python3

"""A simple python script template.
"""

import os
import sys
import argparse
import importlib

import torch
import tensorflow as tf


class ModelError(BaseException):
    """Class for errors in models"""

    pass


class ValidateModel:

    ACCEPTABLE_SAMPLE_RATE = [16000, 22050, 44100, 48000]

    def __init__(self, module_name: str, model_file_path: str):
        self.module_name = module_name
        self.model_file_path = model_file_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.module = None
        self.sample_rate = None
        self.model = None

    def __call__(self):
        self.import_model()
        self.check_load_model()
        self.check_sample_rate()
        self.check_embedding_size()

    def import_model(self):
        print(f"Importing {self.module_name}")
        self.module = importlib.import_module(self.module_name)

    def check_load_model(self):
        print("Checking load_model")
        self.model = self.module.load_model(self.model_file_path)

        if not (isinstance(self.model, tf.Module) or isinstance(self.model, torch.nn.Module)):
            raise ModelError(
                f"Model must be either a PyTorch module: "
                f"https://pytorch.org/docs/stable/generated/torch.nn.Module.html "
                f"or a tensorflow module: "
                f"https://www.tensorflow.org/api_docs/python/tf/Module"
            )

    def check_sample_rate(self):
        print("Checking model sample rate")
        if not hasattr(self.model, "sample_rate"):
            raise ModelError("Model must expose expected input audio "
                             "sample rate as an attribute.")

        if self.model.sample_rate not in self.ACCEPTABLE_SAMPLE_RATE:
            raise ModelError(
                f"Input sample rate of {self.sample_rate} is invalid. "
                f"Must be one of {self.ACCEPTABLE_SAMPLE_RATE}"
            )

    def check_embedding_size(self):
        print("Checking model embedding size")
        if not hasattr(self.model, "scene_embedding_size"):
            raise ModelError("Model must expose the output size of the scene "
                             "embeddings as an attribute: scene_embedding_size")

        if not isinstance(self.model.scene_embedding_size, int):
            raise ModelError("Model.scene_embedding_size must be an int")

        if not hasattr(self.model, "timestamp_embedding_size"):
            raise ModelError("Model must expose the output size of the timestamp "
                             "embeddings as an attribute: timestamp_embedding_size")

        if not isinstance(self.model.timestamp_embedding_size, int):
            raise ModelError("Model.timestamp_embedding_size must be an int")


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "module", help="Name of the model package to validate", type=str
    )
    parser.add_argument(
        "--model",
        "-m",
        default="",
        type=str,
        help="Load model weights from this location",
    )
    args = parser.parse_args(arguments)

    ValidateModel(args.module, args.model)()

    print("Looks good!")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
