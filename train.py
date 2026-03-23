import torch
import torch.nn as nn
import torch.optim as optim
from model import StudentModel
from utils import load_data
from sklearn.metrics import confusion_matrix

# Load data
X_train, X_test, y_train, y_test = load_data("data/student_data.csv")

# Model
model = StudentModel()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    preds = (model(X_test) >= 0.5).float()

print("Accuracy:", (preds == y_test).float().mean().item())
print("Confusion Matrix:\n", confusion_matrix(y_test.numpy(), preds.numpy()))

# Save model
torch.save(model.state_dict(), "model.pth")