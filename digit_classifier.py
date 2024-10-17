from typing import Callable

import numpy as np

from utils.classifier_provider import get_classifier


class DigitClassifier:
    """
    A class to perform digit classification using different algorithms.

    Args:
        algorithm (str): The name of the algorithm to use for classification.
    """

    def __init__(self, algorithm: str) -> None:
        """
        Initializes the classifier with the specified algorithm.

        Args:
            algorithm (str): The algorithm to use, e.g., 'cnn', 'rf', or 'rand'.
        """
        self._algorithm: Callable[[np.ndarray], int] = get_classifier(algorithm)

    def predict(self, image: np.ndarray) -> int:
        """
        Predicts the digit from a given image.

        Args:
            image (np.ndarray): A NumPy array representing a 28x28 grayscale image with shape (28, 28, 1).

        Returns:
            int: The predicted digit (0-9).

        Raises:
            ValueError: If the input image does not have the correct shape (28, 28, 1).
        """
        if image.shape != (28, 28, 1):
            raise ValueError(
                f"Expected input image shape (28, 28, 1), but got {image.shape}"
            )

        # Perform prediction using the selected algorithm
        return self._algorithm(image)
