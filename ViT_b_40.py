import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
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

# Custom Vision Transformer model
from torchvision.models.vision_transformer import vit_b_16

class CustomVisionTransformer(nn.Module):
    def __init__(self, num_classes=100, patch_size=40):
        super(CustomVisionTransformer, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        
        # Modify patch embedding layer to match new patch size
        self.vit.patch_embed = nn.Conv2d(3, self.vit.hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # Adjust the positional embedding if necessary
        num_patches = (224 // patch_size) * (224 // patch_size)
        self.vit.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.vit.hidden_dim))
        
        # Replace the classification head
        self.vit.head = nn.Linear(self.vit.hidden_dim, num_classes)
    
    def forward(self, x):
        return self.vit(x)

# Initialize model, loss function, and optimizer
net = CustomVisionTransformer(num_classes=100, patch_size=40)
net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if exists
start_epoch = 0
# checkpoint_path = './checkpoints/ViT_b_24_checkpoint_epoch_50.pth'
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     print(f"Resuming from checkpoint at epoch {start_epoch}")

# Training the model
epochs = 100
print_freq = 100  # Print frequency
avg_losses = []

for epoch in range(start_epoch, epochs):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % print_freq == print_freq - 1:
            avg_loss = running_loss / print_freq
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] Avg Loss: {avg_loss:.3f}')
            avg_losses.append(avg_loss)
            running_loss = 0.0

    # Save checkpoint at the 50th epoch
    if epoch + 1 == 50:
        checkpoint_path = './checkpoints/ViT_b_40_checkpoint_epoch_50.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

print('Finished Training.')

# Save the trained model
save_directory = './saved_models'
os.makedirs(save_directory, exist_ok=True)
torch.save(net.state_dict(), os.path.join(save_directory, 'vit_cifar100.pth'))
print(f"Model saved to {os.path.join(save_directory, 'vit_cifar100.pth')}")

# Plot average losses
plt.plot(avg_losses)
plt.xlabel('Batch index')
plt.ylabel('Avg. batch loss')
plt.show()

# Evaluate the model
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
