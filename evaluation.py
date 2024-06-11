import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
import os

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-100 dataset
trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the Vision Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=100):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# Initialize model, loss function, and optimizer
net = VisionTransformer(num_classes=100)
net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
    return accuracy

# Load checkpoint function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {filepath}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch']

# Main function to load checkpoint and evaluate
if __name__ == '__main__':
    # Specify the checkpoint file path
    checkpoint_path = './checkpoints/ViT_b_32_checkpoint_epoch_50.pth'

    # Load the checkpoint
    start_epoch = load_checkpoint(checkpoint_path)

    # Evaluate the model
    evaluate_model(net, testloader)
