import torch
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)

    def evaluate_model(self):
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
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)

    def evaluate_model(self):
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
