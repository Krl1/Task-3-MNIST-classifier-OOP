# Model that provides random value (for simplicity) as a result of classification
# with input 10x10 numpy array, the center crop of the image.
from random import randint
from typing import Any

from models.digit_classification_interface import DigitClassificationInterface


class RandModel(DigitClassificationInterface):
    """
    A random model that predicts a digit from 0 to 9 without using any input data.
    Implements the DigitClassificationInterface.
    """

    def predict(self, *args: Any) -> int:
        """
        Predict a random digit between 0 and 9.

        Args:
            *args: Variable length argument list, not used by this model.

        Returns:
            int: A random integer between 0 and 9.
        """
        return randint(0, 9)
