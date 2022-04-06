import torch.nn as nn
import torch.nn.functional as F
class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes, weight=None, bias=None):
        super(LinearClassifier, self).__init__()
        self.model = nn.Linear(n_features, n_classes)
        if weight is not None:
            self.model.weight = nn.Parameter(weight)
        if bias is not None:
            self.model.bias = nn.Parameter(bias)
        self.softmax = nn.Softmax()
        

    def forward(self, x):
        # x =  F.normalize(x)
        # self.model.weight = nn.Parameter(F.normalize(self.model.weight))
        x = self.model(x) 
        return self.softmax(x)

