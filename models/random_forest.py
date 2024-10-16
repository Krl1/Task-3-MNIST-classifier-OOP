import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from models.digit_classification_interface import DigitClassificationInterface


class RandomForest(DigitClassificationInterface):
    """
    A Random Forest classifier for digit classification. It takes a 1D numpy array of length 784
    (representing a 28x28 pixel image) and predicts a digit between 0 and 9.
    """

    def __init__(self):
        super().__init__()
        self._model = RandomForestClassifier()

    def predict(self, image: np.ndarray) -> int:
        """
        Predict the digit from a 1D numpy array representing a flattened image (28x28 pixels).

        Args:
            image (np.ndarray): A 1D numpy array of shape (784,).

        Returns:
            int: Predicted digit (0-9). If the model is not trained, returns a random digit.
        """
        if image.ndim != 1 or image.size != 784:
            raise ValueError(
                "Input must be a 1D numpy array of length 784 (28x28 pixels)."
            )

        # Reshape the image to a 2D array with one sample
        x = image.reshape(1, -1)

        try:
            prediction = self._model.predict(x)
            return int(prediction[0])  # Ensure an integer is returned
        except NotFittedError:
            # Return a random prediction if the model is not fitted
            return random.randint(0, 9)
