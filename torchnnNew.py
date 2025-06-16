# tutorial: https://github.com/nicknochnack/PyTorchin15/blob/main/torchnn.py

import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Number MNIST dataset
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# MNIST imgs are 28x28 px

# after each convolution, 2 px are removed from the original img?


# Image Classifier
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (22) * (22), 10),
        )

    def forward(self, x):
        return self.model(x)


classifier = ImageClassifier().to("mps")
opt = Adam(classifier.parameters(), lr=1e-2)
loss_func = nn.CrossEntropyLoss()

# Training
if __name__ == "__main__":
    for epoch in range(5):  # Train for 5 epochs
        for batch in dataset:
            X, y = batch
            X, y = X.to("mps"), y.to("mps")
            yPred = classifier(X)
            loss = loss_func(yPred, y)

            # Backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss: {loss.item()}")

    with open("model_state.pt", "wb") as f:
        save(classifier.state_dict(), f)

    with open("model_state.pt", "rb") as f:
        classifier.load_state_dict(load(f))

        img = Image.open("img_1.jpg")
        img_tensor = ToTensor()(img).unsqueeze(0).to("mps")

        print(torch.argmax(classifier(img_tensor)))
