import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import BertForSequenceClassification


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)[0]
