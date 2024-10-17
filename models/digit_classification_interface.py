from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch

from utils.input_preparer import get_input_preparer


class DigitClassificationInterface(ABC):
    """
    Abstract base class for digit classification models. This class defines a common interface
    for models that predict a digit (0-9) from an image, either in NumPy or PyTorch tensor format.
    """

    def __init__(self) -> None:
        """
        Initialize the digit classification interface. This retrieves the appropriate input
        preparer function for the class, based on the class name.
        """
        self._input_preparer = get_input_preparer(self.__class__.__name__)

    def __call__(self, image: np.ndarray) -> int:
        """
        Prepares the input and calls the predict method to classify the image.

        Args:
            image (np.ndarray): Input image, expected to be a NumPy array.

        Returns:
            int: Predicted class label (digit from 0-9).
        """
        prepared_image = self._input_preparer(image)
        return self.predict(prepared_image)

    @abstractmethod
    def predict(self, image: Union[np.ndarray, torch.tensor]) -> int:
        """
        Abstract method to be implemented by subclasses for digit prediction.

        Args:
            image (Union[np.ndarray, torch.Tensor]): The input image, prepared for prediction.

        Returns:
            int: Predicted digit (0-9).
        """
        pass
