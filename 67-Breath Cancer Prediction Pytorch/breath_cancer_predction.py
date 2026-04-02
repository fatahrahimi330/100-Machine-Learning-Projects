import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from tqdm import tqdm

bc = load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

print(f"Samples: {n_samples}. Features: {n_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression(n_features)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100

train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in tqdm(range(num_epochs)):
    # Training
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_acc = ((y_pred.round() == y_train).float().mean()) * 100

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_test)
        val_loss = criterion(y_val_pred, y_test)
        val_acc = ((y_val_pred.round() == y_test).float().mean()) * 100

    # Store metrics
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch: {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )


# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(train_losses, label="Train Loss", color="blue")
ax1.plot(val_losses, label="Val Loss", color="orange")
ax1.set_title("Loss over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(train_accs, label="Train Accuracy", color="blue")
ax2.plot(val_accs, label="Val Accuracy", color="green")
ax2.set_title("Train vs Val Accuracy over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# Test Evaluation
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = criterion(y_test_pred, y_test)
    test_acc = ((y_test_pred.round() == y_test).float().mean()) * 100

print(f"Test Loss: {test_loss.item():.4f} | Test Acc: {test_acc:.2f}%")
