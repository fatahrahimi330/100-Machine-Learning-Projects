# Part 0: Importing Libraries
import torch
from torch import device, nn, softmax
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from tqdm.auto import tqdm
# Part 1: Get Data Ready (Turn into Tensor)
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Part 2: Build or Pick  Pretrained Model

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.flatten(x)
        out = self.network(out)
lt        return out


# Part 3: Fit Model to the Data and Make Prediction
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    total_loss = 0
    correct = 0

    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size
    print(f"Train Accuracy: {accuracy:.4f} | Train Loss: {avg_loss:.4f}")

def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()  # IMPORTANT
    total_loss = 0
    correct = 0

    with torch.no_grad():  # no gradient computation
        for X, y in tqdm(dataloader, desc="Validation"):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size

    print(f"Val Accuracy: {accuracy:.4f} | Val Loss: {avg_loss:.4f}")

# Part 4: Evaluate the Model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss_total += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / size
    print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {avg_loss:.4f}")

# Part 5: Save and Reload Trained Model
# torch.save(model.state_dict(), "model.pth")
# model = ANN().to(device)
# model.load_state_dict(torch.load("model.pth", weights_only=True))

if __name__=="__main__":

    print(f"Train Data Len: {len(train_data)}          Test Data Len: {len(test_data)}")

    train_dataLoader = DataLoader(dataset=train_data, batch_size=32)

    test_dataLoader = DataLoader(dataset=test_data, batch_size=32)

    for X, y in train_dataLoader:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        break

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = ANN().to(device)
    print(model)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= 1e-3)

    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n--------------------")
        train(train_dataLoader, model, loss, optimizer)
        validate(test_dataLoader, model, loss)
    test(test_dataLoader,model, loss)
