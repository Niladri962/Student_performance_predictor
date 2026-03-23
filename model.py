import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
#Why this file is created
#To define the neural network architecture
#Keeps model code separate and reusable
#Used by both:
#train.py (for training)
#predict.py (for inference)