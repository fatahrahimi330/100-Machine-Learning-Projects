# Part 0: Import Libraries
from random import shuffle
from turtle import forward
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch 
from torch import device, nn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Part 1: Get Data Ready (Turn into Tensor)
df = pd.read_csv('Social_Network_Ads.csv')

print(df.head())
print(df.info())
print(df.describe())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) # use torch.long for classification, float32

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = CSVDataset(X_train, y_train)
val_data = CSVDataset(X_val, y_val)
test_data = CSVDataset(X_test, y_test)

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size= 32, shuffle=False)

X_batch, y_batch = next(iter(train_dataloader))
print(X_batch.shape)
print(y_batch.shape)
print(X_batch.dtype)

# Part 2: Build or Pick a Pretrained Model (To fit into problem)
device = 'mps' if torch.accelerator.is_available() else 'cpu'
print(device)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.network(x)
        return out

model = ANN().to(device)
print(model)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Part 3: Fit Model to the Data
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch = len(dataloader)

    model.train()
    totla_loss, correct = 0, 0

    for x, y in tqdm(dataloader, desc='Training'):
        x, y = x.to(device), y.to(device)

        pred = model(x).squeeze(1)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totla_loss+= loss.item()
        preds = (torch.sigmoid(pred) >= 0.5).float()
        correct += (preds==y).sum().item()

    avg_loss = totla_loss/batch
    correct = correct/size

    print(f"Training Accuracy: {correct:.4f}   |   Train Loss: {avg_loss:.4f}")


def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    batch = len(dataloader)

    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Validation'):
            x, y = x.to(device), y.to(device)
            
            pred = model(x).squeeze(1)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            preds = (torch.sigmoid(pred) >= 0.5).float()
            correct += (preds == y).sum().item()

    avg_loss = total_loss/batch
    correct = correct/size

    print(f"Validation Accuracy: {correct:.4f}  |   Validation Loss: {avg_loss:.4f}")


# Part 4: Evaluate the Model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    batch = len(dataloader)

    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Test"):
            x, y = x.to(device), y.to(device)

            pred = model(x).squeeze(1)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            preds = (torch.sigmoid(pred) >= 0.5).float()
            correct += (preds == y).sum().item()

    avg_loss = total_loss/batch
    correct = correct/size

    print(f"Test Accuracy: {correct:.4f}    |   Test Loss: {avg_loss:.4f}")



# Part 5: Make Prediction

if __name__=="__main__":

    epochs = 100

    for epoch in range(1, epochs+1):
        print(f"Epoch: {epoch}--------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validate(val_dataloader, model, loss_fn)
    test(test_dataloader, model, loss_fn)