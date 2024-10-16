import torch
from torch import nn
import torch.nn.functional as F

from models.digit_classification_interface import DigitClassificationInterface


class ConvolutionalNeuralNetwork(DigitClassificationInterface):
    """
    A Convolutional Neural Network (CNN) for digit classification. This model takes
    an image tensor of shape (28x28x1) and predicts a digit between 0 and 9.
    """

    def __init__(self):
        """
        Initializes the CNN architecture with two convolutional layers, max-pooling,
        and two fully connected layers.
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9

    def _preprocess_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the input image by converting it to a float tensor of size (1, 1, 28, 28).

        Args:
            image (torch.Tensor): A tensor of shape (28, 28, 1) with int type.

        Returns:
            torch.Tensor: A preprocessed float tensor of shape (1, 1, 28, 28).
        """
        # Convert to float and change shape from (28, 28, 1) to (1, 1, 28, 28)
        return image.permute(2, 0, 1).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (1, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits for 10 classes.
        """
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))  # Shape: (32, 28, 28)
        x = self.pool(F.relu(self.conv2(x)))  # Shape: (64, 14, 14)

        # Flatten the feature maps
        x = x.view(-1, 64 * 14 * 14)  # Shape: (1, 12544)

        # Fully connected layers with ReLU and final output
        x = F.relu(self.fc1(x))  # Shape: (1, 128)
        x = self.fc2(x)  # Shape: (1, 10)

        return x

    def predict(self, image: torch.Tensor) -> int:
        """
        Predicts the class (digit) of the input image.

        Args:
            image (torch.Tensor): A tensor of shape (28, 28, 1) representing the input image.

        Returns:
            int: The predicted digit (0-9).
        """
        # Preprocess the input image
        x = self._preprocess_input(image)

        # Forward pass through the network
        logits = self.forward(x)

        # Return the predicted class (digit)
        return torch.argmax(logits, dim=1).item()
