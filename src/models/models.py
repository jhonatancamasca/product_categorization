import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
