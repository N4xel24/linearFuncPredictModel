import torch.nn as nn

class SimpleFuncEvaluate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 67)
        self.hidden2 = nn.Linear(67, 67)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(67, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation1(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        x = self.output(x)
        return x