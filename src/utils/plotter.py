import matplotlib.pyplot as plt


class Plotter:
    @staticmethod
    def plot_history(train_loss_history, train_acc_history, num_epochs):
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
