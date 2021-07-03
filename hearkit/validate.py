#!/usr/bin/env python3
"""
This command-line script can be called to validate any module against the HEAR 2021
API to ensure that all the current functions and attributes are available, can be called
with the expected input, and produce correctly formed output. To see the API that this
was built against, please visit
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api

Usage:
    python3 -m hearkit.valdate <module-to-test> -m <path-to-model-checkpoint-file>

Example usage:
    python -m hearkit.validate hearkit.baseline

TODO:
    - Build this out to support TensorFlow models as well.
"""

import sys
import argparse
import importlib
import warnings

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
        print(f"Running validation using device: {self.device}.")

        self.module = None
        self.model = None
        self.model_type = None

    def __call__(self):
        self.import_model()
        self.check_load_model()
        self.check_sample_rate()
        self.check_embedding_size()
        self.check_timestamp_embeddings()
        self.check_scene_embeddings()

    def import_model(self):
        print(f"Importing {self.module_name}")
        self.module = importlib.import_module(self.module_name)

    def check_load_model(self):
        print("Checking load_model")
        self.model = self.module.load_model(self.model_file_path)

        if isinstance(self.model, tf.Module):
            self.model_type = "tf"
            raise NotImplementedError("TensorFlow validation needs to be implemented")

        elif isinstance(self.model, torch.nn.Module):
            self.model_type = "torch"

        else:
            raise ModelError(
                f"Model must be either a PyTorch module: "
                f"https://pytorch.org/docs/stable/generated/torch.nn.Module.html "
                f"or a tensorflow module: "
                f"https://www.tensorflow.org/api_docs/python/tf/Module"
            )

    def check_sample_rate(self):
        print("Checking model sample rate")
        if not hasattr(self.model, "sample_rate"):
            raise ModelError(
                "Model must expose expected input audio " "sample rate as an attribute."
            )

        if self.model.sample_rate not in self.ACCEPTABLE_SAMPLE_RATE:
            raise ModelError(
                f"Input sample rate of {self.sample_rate} is invalid. "
                f"Must be one of {self.ACCEPTABLE_SAMPLE_RATE}"
            )

    def check_embedding_size(self):
        print("Checking model embedding size")
        if not hasattr(self.model, "scene_embedding_size"):
            raise ModelError(
                "Model must expose the output size of the scene "
                "embeddings as an attribute: scene_embedding_size"
            )

        if not isinstance(self.model.scene_embedding_size, int):
            raise ModelError("Model.scene_embedding_size must be an int")

        if not hasattr(self.model, "timestamp_embedding_size"):
            raise ModelError(
                "Model must expose the output size of the timestamp "
                "embeddings as an attribute: timestamp_embedding_size"
            )

        if not isinstance(self.model.timestamp_embedding_size, int):
            raise ModelError("Model.timestamp_embedding_size must be an int")

    def check_timestamp_embeddings(self):
        print("Checking get_timestamp_embeddings")
        if not hasattr(self.module, "get_timestamp_embeddings"):
            raise ModelError(
                "Your API must include a function: 'get_timestamp_embeddings'"
            )

        if self.model_type == "torch":
            self.torch_timestamp_embeddings()
        else:
            raise NotImplementedError("Not implemented for TF")

    def check_scene_embeddings(self):
        print("Checking get_scene_embeddings")
        if not hasattr(self.module, "get_scene_embeddings"):
            raise ModelError("Your API must include a function: 'get_scene_embeddings'")

        if self.model_type == "torch":
            self.torch_scene_embeddings()
        else:
            raise NotImplementedError("Not implemented for TF")

    def torch_timestamp_embeddings(self):
        # Create a batch of test audio (white noise)
        num_audio = 16
        length = 2.0
        audio_batch = torch.rand(
            (num_audio, int(length * self.model.sample_rate)), device=self.device
        )

        # Audio samples [-1.0, 1.0]
        audio_batch = (audio_batch * 2) - 1.0

        # Try moving model to device
        self.model.to(self.device)

        print(f"  - Passing in audio batch of shape: {audio_batch.shape}")

        # Get embeddings for the batch of white noise
        embeddings, timestamps = self.module.get_timestamp_embeddings(
            audio_batch, self.model
        )

        print(f"  - Received embedding of shape: {embeddings.shape}")

        # Verify the output looks correct
        if embeddings.dtype != torch.float32:
            raise ModelError(
                f"Expected embeddings to be {torch.float32}, received "
                f"{embeddings.dtype}."
            )

        if embeddings.shape[0] != num_audio:
            raise ModelError(
                f"Passed in a batch of {num_audio} audio samples, but "
                f"your model returned {embeddings.shape[0]}. These values "
                f"should be the same."
            )

        if embeddings.shape[1] != timestamps.shape[0]:
            raise ModelError(
                f"Received {embeddings.shape[1]} timestamp embeddings for "
                f"each audio in the batch. But received "
                f"{timestamps.shape[0]} timestamps. These values should "
                f"be the same."
            )

        if embeddings.shape[2] != self.model.timestamp_embedding_size:
            raise ModelError(
                f"Output embedding size is {embeddings.shape[2]}. Your "
                f"model specified an embedding size of "
                f"{self.model.timestamp_embedding_size} in "
                "Model.timestamp_embedding_size. These values "
                "should be the same."
            )

        # Check that there is a consistent spacing between timestamps.
        # Warn if the spacing is greater than 50ms
        timestamp_diff = torch.diff(timestamps)
        average_diff = torch.mean(timestamp_diff.float())
        print(f"  - Interval between timestamps is {average_diff}ms")

        if average_diff > 50.0:
            warnings.warn(
                "We suggest a interval between timestamps less than or equal "
                "to 50ms to accommodate a tolerance of 50ms for music "
                "transcription tasks."
            )

        if not torch.all(torch.abs(timestamp_diff - average_diff) < 1e-3):
            raise ModelError(
                "Timestamps should occur at regular intervals. Found "
                "a deviation larger than 1ms between adjacent timestamps."
            )

    def torch_scene_embeddings(self):
        # Create a batch of test audio (white noise)
        num_audio = 8
        length = 3.74
        audio_batch = torch.rand(
            (num_audio, int(length * self.model.sample_rate)), device=self.device
        )

        # Audio samples [-1.0, 1.0]
        audio_batch = (audio_batch * 2) - 1.0

        # Try moving model to device
        self.model.to(self.device)

        print(f"  - Passing in audio batch of shape: {audio_batch.shape}")

        # Get embeddings for the batch of white noise
        embeddings = self.module.get_scene_embeddings(audio_batch, self.model)

        print(f"  - Received embedding of shape: {embeddings.shape}")

        # Verify the output looks correct
        if embeddings.dtype != torch.float32:
            raise ModelError(
                f"Expected embeddings to be {torch.float32}, received "
                f"{embeddings.dtype}."
            )

        if embeddings.shape[0] != num_audio:
            raise ModelError(
                f"Passed in a batch of {num_audio} audio samples, but "
                f"your model returned {embeddings.shape[0]}. These values "
                f"should be the same."
            )

        if embeddings.shape[1] != self.model.scene_embedding_size:
            raise ModelError(
                f"Output embedding size is {embeddings.shape[1]}. Your "
                f"model specified an embedding size of "
                f"{self.model.scene_embedding_size} in "
                "Model.scene_embedding_size. These values "
                "should be the same."
            )


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

    # Run validation
    ValidateModel(args.module, args.model)()
    print("Looks good!")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
