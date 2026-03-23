import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    X = df[['studytime', 'failures', 'absences', 'G1', 'G2']].values
    y = (df['G3'] >= 10).astype(int).values.reshape(-1,1)

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )