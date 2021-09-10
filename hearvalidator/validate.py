#!/usr/bin/env python3
"""
This command-line script can be called to validate any module against the HEAR 2021
API to ensure that all the current functions and attributes are available, can be called
with the expected input, and produce correctly formed output. To see the API that this
was built against, please visit
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api

Usage:
    hear-validator <module-to-test> -m <path-to-model-checkpoint-file> -d <device>

Example usage:
    hear-validator hearbaseline -m naive_baseline.pt -d cuda
"""

import argparse
import importlib
import warnings
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch


class ModelError(BaseException):
    """Class for errors in models"""

    pass


class ValidateModel:

    ACCEPTABLE_SAMPLE_RATE = [16000, 22050, 32000, 44100, 48000]

    def __init__(self, module_name: str, model_file_path: str, device: str = None):
        self.module_name = module_name
        self.model_file_path = model_file_path
        self.device = device

        self.module = None
        self.model = None
        self.model_type = None

    def __call__(self):
        self.import_model()
        self.check_load_model()

        # Perform validation. If a tensorflow model was loaded and a specific
        # device was specified, then reload the model on the correct device.
        # This is a bit awkward but I wanted to avoid all the tensorflow initialization
        # stuff before loading a module, which could potentially be a PyTorch module.
        # So we only get into the tensorflow device stuff if we found a tf module.
        if self.model_type == "tf" and self.device is not None:
            with tf.device(self.device):
                # Re-import model using correct device
                print(f"Reloading tf model on {self.device}")
                del self.model
                self.check_load_model()
                self.validate_model()
        else:
            self.validate_model()

    def validate_model(self):
        self.check_sample_rate()
        self.check_embedding_size()
        self.check_timestamp_embeddings()
        self.check_scene_embeddings()

    def import_model(self):
        print(f"Importing {self.module_name}")
        self.module = importlib.import_module(self.module_name)

    def check_load_model(self):
        print("Checking load_model")
        if not hasattr(self.module, "load_model"):
            raise ModelError("Your API must include a function: 'load_model'")

        # Try to load the module. Use a weight file if one was provided
        if self.model_file_path:
            print(f"  - Loading model with weights file: {self.model_file_path}")
            self.model = self.module.load_model(self.model_file_path)
        else:
            print("  - No weight file provided. Using default")
            self.model = self.module.load_model()

        # TensorFlow module
        if isinstance(self.model, tf.Module):
            self.model_type = "tf"
            print(f"  - Received tensorflow Module: {self.model}")

        # PyTorch module -- also setup the device if None was passed
        elif isinstance(self.model, torch.nn.Module):
            self.model_type = "torch"
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(
                f"  - Received torch Module: {self.model}. "
                f"Loading onto device: {self.device}"
            )
            self.model.to(self.device)

        else:
            raise ModelError(
                "Model must be either a PyTorch module: "
                "https://pytorch.org/docs/stable/generated/torch.nn.Module.html "
                "or a tensorflow module: "
                "https://www.tensorflow.org/api_docs/python/tf/Module"
            )

    def check_sample_rate(self):
        print("Checking model sample rate")
        if not hasattr(self.model, "sample_rate"):
            raise ModelError(
                "Model must expose expected input audio sample rate as an attribute."
            )

        print(f"  - Model sample rate is: {self.model.sample_rate}")
        if self.model.sample_rate not in self.ACCEPTABLE_SAMPLE_RATE:
            raise ModelError(
                f"Input sample rate of {self.model.sample_rate} is invalid. "
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

        print(f"  - scene_embedding_size: {self.model.scene_embedding_size}")

        if not hasattr(self.model, "timestamp_embedding_size"):
            raise ModelError(
                "Model must expose the output size of the timestamp "
                "embeddings as an attribute: timestamp_embedding_size"
            )

        print(f"  - timestamp_embedding_size: {self.model.timestamp_embedding_size}")

        if not isinstance(self.model.timestamp_embedding_size, int):
            raise ModelError("Model.timestamp_embedding_size must be an int")

    def check_timestamp_embeddings(self):
        # Run this a few times to check embeddings match timestamps
        self._check_timestamp_embeddings(num_audio=2, length=1.07)
        self._check_timestamp_embeddings(num_audio=2, length=1.98)
        self._check_timestamp_embeddings(num_audio=2, length=4.0)
        # for i in range(20):
        #    import random
        #    self._check_timestamp_embeddings(num_audio=2,
        #        length=(1+random.random() * 4))

        warnings.warn(
            """ IMPORTANT: A common bug we have seen in many codebases
                  involves rounding errors accumulating over longer audio.
                  For example, if you want embeddings every 25ms and have
                  44100Hz audio, then the sample hop length is 1102.5. If
                  you round this before your for loop, the timestamp centers
                  and/or embedding sample centers might be wrong. In this
                  example, you will drift 25ms every 37 minutes of audio.
                  We can't detect this drift with short audio in the
                  validator.
                  """
        )

    def _check_timestamp_embeddings(self, num_audio, length):
        print("Checking get_timestamp_embeddings")
        if not hasattr(self.module, "get_timestamp_embeddings"):
            raise ModelError(
                "Your API must include a function: 'get_timestamp_embeddings'"
            )

        if self.model_type == "torch":
            embeddings, timestamps = self.torch_timestamp_embeddings(
                num_audio=num_audio, length=length
            )
        else:
            embeddings, timestamps = self.tf2_timestamp_embeddings(
                num_audio=num_audio, length=length
            )

        print(f"  - Received embedding of shape: {embeddings.shape}")
        print(f"  - Received timestamps of shape: {timestamps.shape}")

        if embeddings.ndim != 3:
            raise ModelError(
                "Output dimensions of the embeddings from get_timestamp_embeddings is "
                f"incorrect. Expected 3 dimensions, but received shape"
                f"{embeddings.shape}."
            )

        if timestamps.ndim != 2:
            raise ModelError(
                "Output dimensions of the timestamps from get_timestamp_embeddings is "
                f"incorrect. Expected 2 dimensions, but received shape"
                f"{timestamps.shape}."
            )

        if embeddings.shape[0] != num_audio:
            raise ModelError(
                f"Passed in a batch of {num_audio} audio samples, but "
                f"your model returned {embeddings.shape[0]}. These values "
                f"should be the same."
            )

        if embeddings.shape[:2] != timestamps.shape:
            raise ModelError(
                "Output shape of the timestamps from get_timestamp_embeddings is "
                f"incorrect. Expected {embeddings.shape[:2]}, but received "
                f"{timestamps.shape}."
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
        if timestamps.shape[1] > 1:
            timestamp_diff = np.diff(timestamps)
            avg_diff = np.mean(timestamp_diff)
            max_diff = np.max(timestamp_diff)
            print(f"  - Avg interval between timestamps is {avg_diff}ms")
            print(f"  - Max interval between timestamps is {max_diff}ms")

            if max_diff > 50.0:
                warnings.warn(
                    "We suggest a interval between timestamps less than or equal "
                    "to 50ms to accommodate a tolerance of 50ms for music "
                    "transcription tasks."
                )

            timestamp_deviation = np.max(np.abs(timestamp_diff - avg_diff))
            if timestamp_deviation > 1:
                raise ModelError(
                    "Timestamps should occur at regular intervals. Found "
                    f"a deviation {timestamp_deviation}ms larger than 1ms "
                    "between adjacent timestamps. "
                    "If you REALLY want to use a variable hop-size,"
                    "please contact us"
                )

            # These checks are cool but won't catch subtle bugs like the above.
            min_time = np.min(timestamps)
            max_time = np.max(timestamps)
            print(f"  - Min timestamp {min_time}ms")
            print(f"  - Max timestamp {max_time}ms")
            if min_time > avg_diff:
                warnings.warn(
                    f"Your timestamps begin at {min_time}ms, which appears to be "
                    "wrong."
                )
            if (max_time < 1000 * length - avg_diff) or (max_time < 1000 * length - 50):
                warnings.warn(
                    f"Your timestamps end at {max_time}ms, but the "
                    f"audio is {1000 * length} ms. You won't have "
                    f"embeddings for events at the end of the audio."
                )
            if max_time > 1000 * length:
                raise ModelError(
                    f"Your timestamps end at {max_time}ms, but the "
                    f"audio is {1000 * length} ms."
                )

    def check_scene_embeddings(self):
        print("Checking get_scene_embeddings")
        if not hasattr(self.module, "get_scene_embeddings"):
            raise ModelError("Your API must include a function: 'get_scene_embeddings'")

        num_audio = 4
        length = 2.74
        if self.model_type == "torch":
            embeddings = self.torch_scene_embeddings(num_audio, length)
        else:
            embeddings = self.tf2_scene_embeddings(num_audio, length)

        print(f"  - Received embedding of shape: {embeddings.shape}")

        # Verify the output looks correct
        if embeddings.dtype != np.float32:
            raise ModelError(
                f"Expected embeddings to be {np.float32}, received "
                f"{embeddings.dtype}."
            )

        if embeddings.ndim != 2:
            raise ModelError(
                "Output dimensions of the embeddings from get_scene_embeddings is "
                f"incorrect. Expected 2 dimensions, but received shape"
                f"{embeddings.shape}."
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

    def torch_timestamp_embeddings(
        self, num_audio: int, length: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Create a batch of test audio (white noise)
        audio_batch = torch.rand(
            (num_audio, int(length * self.model.sample_rate)), device=self.device
        )

        # Audio samples [-1.0, 1.0]
        audio_batch = (audio_batch * 2) - 1.0

        print(f"  - Passing in audio batch of shape: {audio_batch.shape}")

        # Get embeddings for the batch of white noise
        embeddings, timestamps = self.module.get_timestamp_embeddings(
            audio_batch, self.model
        )

        # Verify the output looks correct
        if embeddings.dtype != torch.float32:
            raise ModelError(
                f"Expected embeddings to be {torch.float32}, received "
                f"{embeddings.dtype}."
            )

        return embeddings.detach().cpu().numpy(), timestamps.detach().cpu().numpy()

    def tf2_timestamp_embeddings(
        self, num_audio: int, length: float
    ) -> Tuple[np.ndarray, np.array]:
        # Create a batch of test audio (white noise)
        audio_batch = tf.random.uniform(
            (num_audio, int(length * self.model.sample_rate))
        )

        # Audio samples [-1.0, 1.0]
        audio_batch = (audio_batch * 2) - 1.0

        print(f"  - Passing in audio batch of shape: {audio_batch.shape}")

        # Get embeddings for the batch of white noise
        embeddings, timestamps = self.module.get_timestamp_embeddings(
            audio_batch, self.model
        )

        # Verify the output looks correct
        if embeddings.dtype != tf.float32:
            raise ModelError(
                f"Expected embeddings to be {tf.float32}, received "
                f"{embeddings.dtype}."
            )

        return embeddings.numpy(), timestamps.numpy()

    def torch_scene_embeddings(self, num_audio: int, length: float) -> np.ndarray:
        # Create a batch of test audio (white noise)
        audio_batch = torch.rand(
            (num_audio, int(length * self.model.sample_rate)), device=self.device
        )

        # Audio samples [-1.0, 1.0]
        audio_batch = (audio_batch * 2) - 1.0

        print(f"  - Passing in audio batch of shape: {audio_batch.shape}")

        # Get embeddings for the batch of white noise
        embeddings = self.module.get_scene_embeddings(audio_batch, self.model)

        # Verify the output looks correct
        if embeddings.dtype != torch.float32:
            raise ModelError(
                f"Expected embeddings to be {torch.float32}, received "
                f"{embeddings.dtype}."
            )

        return embeddings.detach().cpu().numpy()

    def tf2_scene_embeddings(self, num_audio: int, length: float) -> np.ndarray:
        # Create a batch of test audio (white noise)
        num_audio = 4
        length = 2.74
        audio_batch = tf.random.uniform(
            (num_audio, int(length * self.model.sample_rate))
        )

        # Audio samples [-1.0, 1.0]
        audio_batch = (audio_batch * 2) - 1.0

        print(f"  - Passing in audio batch of shape: {audio_batch.shape}")

        # Get embeddings for the batch of white noise
        embeddings = self.module.get_scene_embeddings(audio_batch, self.model)

        # Verify the output looks correct
        if embeddings.dtype != tf.float32:
            raise ModelError(
                f"Expected embeddings to be {tf.float32}, received "
                f"{embeddings.dtype}."
            )

        return embeddings.numpy()


def main():
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
        help="Load model from this location",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        type=str,
        help="Device to run validation on. If not provided will try to use GPU if "
        "available.",
    )
    args = parser.parse_args()

    # Run validation
    ValidateModel(args.module, args.model, device=args.device)()
    print("Looks good!")


if __name__ == "__main__":
    main()
