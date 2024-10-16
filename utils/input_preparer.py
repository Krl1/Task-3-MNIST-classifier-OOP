from typing import Callable, Union

import torch
import numpy as np


def prepare_cnn_input(image: np.ndarray) -> torch.Tensor:
    """Return a tensor of shape (28, 28, 1)."""
    return torch.tensor(image, dtype=torch.float32)


def prepare_rf_input(image: np.ndarray) -> np.ndarray:
    """
    Prepares input for Random Forest classifier.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: A 1D numpy array of length 784 (flattened).
    """
    return image.flatten()


def prepare_rand_input(image: np.ndarray) -> np.ndarray:
    """
    Prepares input for Random model (RandModel) by performing a center crop.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: A 2D numpy array of shape (10, 10), center-cropped from the input.
    """
    return image[9:19, 9:19]


# Dictionary mapping model names to their respective input preparer functions
INPUT_PREPARER_REGISTRY = {
    "ConvolutionalNeuralNetwork": prepare_cnn_input,
    "RandomForest": prepare_rf_input,
    "RandModel": prepare_rand_input,
}


def get_input_preparer(
    classifier_name: str,
) -> Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]:
    """
    Retrieves the input preparer function for the given classifier name.

    Args:
        classifier_name (str): Name of the classifier ("ConvolutionalNeuralNetwork", "RandomForest", or "RandModel").

    Returns:
        Callable[[np.ndarray], Union[np.ndarray, torch.Tensor]]: The appropriate input preparer function.

    Raises:
        ValueError: If the classifier name is unknown.
    """
    try:
        return INPUT_PREPARER_REGISTRY[classifier_name]
    except KeyError:
        raise ValueError(
            f"Unknown classifier: '{classifier_name}'. Available options are: {', '.join(INPUT_PREPARER_REGISTRY.keys())}."
        )
