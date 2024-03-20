import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
    def __len__(self):
        return len(self.dl)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def main():
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = NeuralNetwork().to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    trainset = datasets.MNIST(root="./data", train=True,
                             download=True,transform=transform)
    train_loader = DataLoader(trainset, batch_size=10, shuffle=True)

    testset = datasets.MNIST(root="./data", train=False,
                             download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=10, shuffle=True)

    train_loader = DeviceDataLoader(train_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()

        for data in train_loader:
            X, y = data
            optimizer.zero_grad()
            output = model(X.view(-1, 28*28))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data in test_loader:
            X, y = data
            output = model(X.view(-1, 784))
            for index, i in enumerate(output):
                if torch.argmax(i) == y[index]:
                    correct_predictions += 1
                total_predictions += 1
    print(f"Accuracy: {round(correct_predictions/total_predictions, 3)}")




if __name__ == "__main__":
    main()