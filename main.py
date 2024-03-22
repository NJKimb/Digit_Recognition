import sys
import cv2
import torch
import numpy
from PIL import Image
from torch import nn
from torch.nn import MaxPool2d, LogSoftmax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparameters
LearningRate = 0.001
BatchSize = 10
Epochs = 25
DropoutRate = .3


# Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # First layer of convolution (Conv -> RelU -> Max Pool -> Dropout)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(DropoutRate)

        # Second layer of convolution (Conv -> RelU -> Max Pool -> Dropout)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(DropoutRate)

        # First layer of linear units ( Linear -> RelU -> Dropout)
        self.linear1 = nn.Linear(500, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(DropoutRate)

        # Output layer of NN ( Linear -> SoftMax)
        self.linear2 = nn.Linear(256, 10)
        self.softmax = LogSoftmax(dim=1)

    def forward(self, x):
        # First Layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        # Second Layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # Flatten then third layer
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Output Layer
        x = self.linear2(x)
        logits = self.softmax(x)

        return logits


# Dataloader to send data to GPU
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)


# Utility function for moving data to GPU
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def train_network():
    # Use GPU if available
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = NeuralNetwork().to(device)

    # Download and transform MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST(root="./data", train=True,
                              download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BatchSize, shuffle=True)

    testset = datasets.MNIST(root="./data", train=False,
                             download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=BatchSize, shuffle=True)

    train_loader = DeviceDataLoader(train_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate)

    # Training loop
    for epoch in range(Epochs):
        model.train()

        for (X, y) in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    correct_predictions = 0
    total_predictions = 0

    # Validation loop
    with torch.no_grad():
        model.eval()

        for data in test_loader:
            images, labels = data
            pred = model(images)
            predicted = torch.max(pred.data, 1)[1]
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    torch.save(model.state_dict(), "dict.pth")
    print(f"Accuracy: {accuracy}")


def detect():
    # Initialize and load neural network
    model = NeuralNetwork()
    model.load_state_dict(torch.load("dict.pth"))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Transform image to MNIST format
    image = Image.open(sys.argv[1]).convert('L')
    image = image.resize((28, 28))
    image = cv2.bitwise_not(numpy.asarray(image))

    # Transform to tensor
    image = transform(image)

    # Predict with neural network
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        predicted = torch.max(pred.data, 1)[1].item()
        class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        predicted_label = class_labels[predicted]
        print(f"Predicted label: {predicted_label}")


if __name__ == "__main__":
    if sys.argv[1] == '-t':
        train_network()
    else:
        detect()
