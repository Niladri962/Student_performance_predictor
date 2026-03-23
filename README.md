# 🎓 Student Performance Predictor

A complete **end-to-end Machine Learning project** that predicts whether a student will **pass or fail** using a neural network and provides predictions through a web interface.

---

## 🚀 Overview

This project demonstrates how to:

* Build a neural network from scratch using PyTorch
* Train it on real-world student data
* Evaluate performance using proper ML metrics
* Deploy it as a web application using Flask

---

## 🧠 Problem Statement

Educational institutions often need early insights into student performance.

This model predicts:
👉 **Pass (1)** or **Fail (0)**

Based on:

* Study time
* Number of past failures
* Absences
* First period grade (G1)
* Second period grade (G2)

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Flask

---

## 📂 Project Structure

```id="n1r2k5"
student-ml-project/
│── data/
│   └── student_data.csv
│── model.py
│── utils.py
│── train.py
│── predict.py
│── app.py
│── requirements.txt
```

---

## 🔄 Workflow Pipeline

```id="9c3w1m"
Data → Preprocessing → Training → Evaluation → Deployment → Prediction
```

---

## 📊 Data Preprocessing

* Selected relevant features from dataset
* Converted target into binary classification
* Applied normalization
* Split data into training and testing sets

---

## 🧠 Model Architecture

* Input Layer: 5 features
* Hidden Layer: 8 neurons (ReLU activation)
* Output Layer: 1 neuron (Sigmoid activation)

---

## 📈 Model Evaluation

* Accuracy score to measure performance
* Confusion matrix to analyze prediction errors

---

## 🌐 Web Application

A simple Flask-based web interface allows users to:

* Enter student details
* Get instant predictions
* View probability score

---

## ▶️ Setup Instructions

### 1. Clone repository

```id="j4t8x0"
git clone <your-repo-link>
cd student-ml-project
```

### 2. Install dependencies

```id="g8k2p9"
pip install -r requirements.txt
```

### 3. Train model

```id="x0z6m1"
python train.py
```

### 4. Run web app

```id="r2p8v4"
python app.py
```

---

## 🌐 Usage

* Open: `http://127.0.0.1:5000`
* Input values in form
* Click **Predict**
* Get result instantly

---

## 🔥 Key Features

* End-to-end ML pipeline
* Real dataset usage
* Clean modular code structure
* Web-based prediction system
* Easy to extend and deploy

---

## 🚀 Future Improvements

* Add advanced UI (React frontend)
* Deploy on cloud (Render/Railway)
* Add visualization dashboard
* Improve model with more features
* Add explainability (feature importance)

---

## 📌 Learning Outcomes

* Understanding of neural networks
* Data preprocessing techniques
* Model evaluation methods
* API and web deployment basics
* Real-world ML project structuring


---

## ⭐ Support

If you found this useful:

* ⭐ Star the repository
* 🍴 Fork and experiment
* 📢 Share with others

---
