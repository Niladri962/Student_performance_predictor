import torch
import numpy as np
from model import StudentModel

# Load model
model = StudentModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Example input: [studytime, failures, absences, G1, G2]
data = np.array([[3, 0, 4, 12, 13]])

# Normalize (same as training — adjust if needed)
mean = np.array([0, 0, 0, 0, 0])
std = np.array([1, 1, 1, 1, 1])
data = (data - mean) / std

data = torch.tensor(data, dtype=torch.float32)

with torch.no_grad():
    pred = model(data).item()

print("Probability:", pred)
print("Result:", "Pass" if pred > 0.5 else "Fail")