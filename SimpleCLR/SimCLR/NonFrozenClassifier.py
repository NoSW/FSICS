import torch.nn as nn
import torch

class NonFrozenClassifier(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, n_features, n_classes):
        super(NonFrozenClassifier, self).__init__()

        self.encoder = encoder
        self.linear = nn.Linear(n_features, n_classes)
        self.model.weight = nn.Parameter(torch.randn((n_classes, n_features)))
        self.model.bias = nn.Parameter(torch.zeros(n_classes))
        self.softmax = nn.Softmax()

    def forward(self, x_i):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
