import torch
from sklearn.metrics import classification_report, confusion_matrix
from typing import Any


class Evaluator:
    """
    Evaluator class to evaluate a PyTorch model on a given dataloader.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the evaluation dataset.
        device (str, optional): The device on which to perform the evaluation. Default is 'cuda'.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the evaluation dataset.
        device (str): The device on which to perform the evaluation.

    """

    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)

    def evaluate_model(self) -> None:
        """
        Evaluate the model on the given dataloader and print the classification report and confusion matrix.

        Returns:
            None

        """
        self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_labels, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))


class BERTEvaluator:
    """
    BERTEvaluator class to evaluate a BERT-based model on a given dataloader.

    Args:
        model (torch.nn.Module): The BERT-based model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the evaluation dataset.
        device (str, optional): The device on which to perform the evaluation. Default is 'cuda'.

    Attributes:
        model (torch.nn.Module): The BERT-based model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the evaluation dataset.
        device (str): The device on which to perform the evaluation.

    """

    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)

    def evaluate_model(self) -> None:
        """
        Evaluate the BERT-based model on the given dataloader and print the classification report and confusion matrix.

        Returns:
            None

        """
        self.model.to(self.device)
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("Classification Report:")
        print(classification_report(all_labels, all_preds))

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
