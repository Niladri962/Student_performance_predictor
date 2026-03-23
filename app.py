from flask import Flask, request, jsonify, render_template_string
import torch
import numpy as np
from model import StudentModel

app = Flask(__name__)

model = StudentModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

mean = np.array([0,0,0,0,0])
std = np.array([1,1,1,1,1])

# Simple UI
HTML = """
<h2>Student Predictor</h2>
<form method="post" action="/predict">
  Study Time: <input name="studytime"><br>
  Failures: <input name="failures"><br>
  Absences: <input name="absences"><br>
  G1: <input name="G1"><br>
  G2: <input name="G2"><br>
  <button type="submit">Predict</button>
</form>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    input_data = np.array([[
        float(data["studytime"]),
        float(data["failures"]),
        float(data["absences"]),
        float(data["G1"]),
        float(data["G2"])
    ]])

    input_data = (input_data - mean) / std
    input_data = torch.tensor(input_data, dtype=torch.float32)

    with torch.no_grad():
        pred = model(input_data).item()

    return f"Result: {'Pass' if pred > 0.5 else 'Fail'} | Probability: {round(pred,3)}"

if __name__ == "__main__":
    app.run(debug=True)