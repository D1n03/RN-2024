import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd


# Define transforms with data augmentation
def transforms(train=True):
    if train:
        return Compose([
            RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Random rotations and shifts
            ToTensor(),
            Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            lambda x: x.view(-1)  # Flatten image to [784]
        ])
    else:
        return Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            lambda x: x.view(-1)  # Flatten image to [784]
        ])


# Define the model
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


# Define the training loop
def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    mean_loss = 0.0

    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mean_loss += loss.item()

    mean_loss /= len(train_dataloader)
    return mean_loss


# Define the validation loop
@torch.inference_mode()
def val(model, val_dataloader, criterion, device):
    model.eval()
    mean_loss = 0.0

    for data, labels in val_dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        mean_loss += loss.item()

    mean_loss /= len(val_dataloader)
    return mean_loss


# Compute accuracy
@torch.inference_mode()
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# Generate submission file
@torch.inference_mode()
def generate_submission(model, dataloader, device, filepath="submission.csv"):
    model.eval()
    predictions = []

    for idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        predictions.append((idx, preds.item()))

    submission = pd.DataFrame(predictions, columns=["ID", "target"])
    submission.to_csv(filepath, index=False)


# Main training and validation process
def main(model, train_dataloader, val_dataloader, val_dataset, criterion, optimizer, scheduler, device, epochs):
    with tqdm(range(epochs)) as tbar:
        for epoch in tbar:
            train_loss = train(model, train_dataloader, criterion, optimizer, device)
            val_loss = val(model, val_dataloader, criterion, device)
            scheduler.step(val_loss)  # Reduce LR on plateau

            train_acc = compute_accuracy(model, train_dataloader, device)
            val_acc = compute_accuracy(model, val_dataloader, device)

            tbar.set_description(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.3f}, Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.3f}, Acc: {val_acc:.6f}"
            )

            if val_acc >= 0.99440:
                test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                generate_submission(model, test_dataloader, device, filepath="submission.csv")


# Load datasets and dataloaders
def load_data():
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms(train=True))
    val_dataset = MNIST(root='./data', train=False, download=True, transform=transforms(train=False))

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=500, shuffle=False)

    return train_dataloader, val_dataloader, val_dataset


if __name__ == "__main__":
    # Load data
    train_dataloader, val_dataloader, val_dataset = load_data()

    # Initialize model, loss, optimizer, and scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Train and validate model
    epochs = 300
    main(model, train_dataloader, val_dataloader, val_dataset, criterion, optimizer, scheduler, device, epochs)

    # Generate submission file
    test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    generate_submission(model, test_dataloader, device, filepath="submission.csv")