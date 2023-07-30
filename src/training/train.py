import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



class Trainer:
    def __init__(self, model, train_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device(device)

    def train(self, num_epochs, criterion, optimizer):
        self.model.to(self.device)
        self.model.train()

        train_loss_history = []
        train_acc_history = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0

            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct_preds / total_samples

            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Acc: {epoch_acc}")

        return train_loss_history, train_acc_history





class BERTTrainer:
    def __init__(self, model, train_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device(device)

    def train(self, num_epochs, criterion, optimizer):
        self.model.to(self.device)
        self.model.train()

        train_loss_history = []
        train_acc_history = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_preds = 0
            total_samples = 0

            for batch in tqdm(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct_preds / total_samples

            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Acc: {epoch_acc}")

        return train_loss_history, train_acc_history

