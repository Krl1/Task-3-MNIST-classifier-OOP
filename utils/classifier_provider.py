from typing import Callable

from models.convolutional_neural_network import ConvolutionalNeuralNetwork
from models.random_forest import RandomForest
from models.rand_model import RandModel


# Dictionary to map classifier names to their respective classes
CLASSIFIER_REGISTRY = {
    "cnn": ConvolutionalNeuralNetwork,
    "rf": RandomForest,
    "rand": RandModel,
}


def get_classifier(classifier_name: str) -> Callable:
    """
    Retrieves the classifier class based on the provided classifier name.

    Args:
        classifier_name (str): The name of the classifier. It must be one of:
                              "cnn", "rf", or "rand".

    Returns:
        Callable: An instance of the requested classifier.

    Raises:
        ValueError: If the classifier_name is not recognized.
    """
    if classifier_name not in CLASSIFIER_REGISTRY:
        available_classifiers = ", ".join(CLASSIFIER_REGISTRY.keys())
        raise ValueError(
            f"Unknown classifier: '{classifier_name}'. Available options are: {available_classifiers}."
        )

    classifier_class = CLASSIFIER_REGISTRY[classifier_name]
    return classifier_class()
