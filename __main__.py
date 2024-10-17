import argparse

import numpy as np

from digit_classifier import DigitClassifier


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MNIST Digit Classifier")
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="cnn",
        choices=["cnn", "rf", "rand"],
        help="Algorithm to use for classification. Choose from 'cnn', 'rf', or 'rand'. Default is 'cnn'.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main execution function. Parses arguments, generates random input,
    initializes the classifier, and prints the prediction.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Generate random input representing a 28x28x1 grayscale image
    random_input = np.random.randint(0, 256, (28, 28, 1))

    # Initialize the classifier with the selected algorithm
    classifier = DigitClassifier(args.algorithm)

    # Predict the digit from the random input
    prediction = classifier.predict(random_input)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
