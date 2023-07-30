import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import BertForSequenceClassification


class EfficientNetClassifier(nn.Module):
    """
    EfficientNetClassifier class for image classification using EfficientNet.

    Args:
        num_classes (int): The number of output classes.

    Attributes:
        model (EfficientNet): The pre-trained EfficientNet model.

    """

    def __init__(self, num_classes: int):
        super(EfficientNetClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass of the EfficientNetClassifier.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.model(x)


class BertClassifier(nn.Module):
    """
    BertClassifier class for sequence classification using BERT.

    Args:
        num_labels (int): The number of output labels.

    Attributes:
        model (BertForSequenceClassification): The pre-trained BERT model for sequence classification.

    """

    def __init__(self, num_labels: int):
        super(BertClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the BertClassifier.

        Args:
            input_ids (torch.Tensor): The input tensor containing the tokenized input sequences.
            attention_mask (torch.Tensor): The attention mask tensor indicating which tokens to attend to.

        Returns:
            torch.Tensor: The output tensor containing the logits.

        """
        return self.model(input_ids, attention_mask=attention_mask)[0]
