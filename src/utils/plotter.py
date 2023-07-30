import matplotlib.pyplot as plt
from typing import List


class Plotter:
    """
    Plotter class to visualize training history.

    """

    @staticmethod
    def plot_history(train_loss_history: List[float], train_acc_history: List[float], num_epochs: int) -> None:
        """
        Plot the training history for accuracy and loss.

        Args:
            train_loss_history (List[float]): List of training loss values for each epoch.
            train_acc_history (List[float]): List of training accuracy values for each epoch.
            num_epochs (int): The total number of epochs.

        Returns:
            None

        """
        plt.figure(figsize=(10, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_acc_history, label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
