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
    def __init__(self, num_classes=100, patch_size=8):
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
net = CustomVisionTransformer(num_classes=100, patch_size=8)
net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if exists
start_epoch = 0
checkpoint_path = './checkpoints/ViT_b_8_checkpoint_epoch_50.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from checkpoint at epoch {start_epoch}")

# Training the model
epochs = 1
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
        checkpoint_path = './checkpoints/ViT_b_8_checkpoint_epoch_50.pth'
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








# import torch
# import torch.nn as nn
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# import timm
# from torchvision.datasets import CIFAR100
# import matplotlib.pyplot as plt
# import os

# # Data transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize CIFAR-100 images to 224x224
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(224, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load CIFAR-100 dataset
# trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# testset = CIFAR100(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# # Set device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# # Custom Vision Transformer model
# class CustomVisionTransformer(nn.Module):
#     def __init__(self, num_classes=100, img_size=224, patch_size=24, pretrained_path=None):
#         super(CustomVisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.img_size = img_size
#         self.patch_size = patch_size

#         # Load pre-trained ViT model without loading weights
#         self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        
#         # Modify the patch embedding layer
#         self.vit.patch_embed.proj = nn.Conv2d(3, self.vit.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
#         # Modify the head
#         self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

#         # Adjust the positional embeddings
#         self.vit.pos_embed = nn.Parameter(self.resize_pos_embed(self.vit.pos_embed, self.vit.patch_embed, self.vit.embed_dim))

#         # Load pre-trained weights if provided
#         if pretrained_path:
#             state_dict = torch.load(pretrained_path)
#             self.vit.load_state_dict(state_dict, strict=False)

#     def forward(self, x):
#         return self.vit(x)

#     def resize_pos_embed(self, pos_embed, patch_embed, embed_dim):
#         num_patches = (self.img_size // self.patch_size) ** 2
#         cls_token = pos_embed[:, 0:1, :]
#         pos_tokens = pos_embed[:, 1:, :]

#         # Calculate new grid size
#         new_grid_size = int(num_patches ** 0.5)
#         old_grid_size = int(pos_tokens.shape[1] ** 0.5)

#         # Reshape pos_tokens to match old grid size
#         pos_tokens = pos_tokens.view(1, old_grid_size, old_grid_size, embed_dim)
        
#         # Interpolate
#         new_pos_tokens = nn.functional.interpolate(
#             pos_tokens.permute(0, 3, 1, 2),
#             size=(new_grid_size, new_grid_size),
#             mode='bilinear',
#             align_corners=False
#         )
        
#         # Reshape back to 2D
#         new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).view(1, new_grid_size * new_grid_size, embed_dim)
        
#         # Concatenate class token
#         new_pos_embed = torch.cat((cls_token, new_pos_tokens), dim=1)
#         return new_pos_embed

# # Path to the downloaded pre-trained weights
# pretrained_path = './weights/vit-base-patch16-224.bin'  # Update this path to your local file

# # Initialize model, loss function, and optimizer
# model = CustomVisionTransformer(num_classes=100, img_size=224, patch_size=8, pretrained_path=pretrained_path)
# model.to(device)

# loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training function
# def train_model():
#     epochs = 50
#     print_freq = 100  # Print frequency
#     avg_losses = []

#     for epoch in range(epochs):
#         running_loss = 0.0
#         model.train()
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = loss_func(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             if i % print_freq == print_freq - 1:
#                 avg_loss = running_loss / print_freq
#                 print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] Avg Loss: {avg_loss:.3f}')
#                 avg_losses.append(avg_loss)
#                 running_loss = 0.0

#         # Save checkpoint at the 50th epoch
#         if epoch + 1 == 10:
#             checkpoint_dir = './checkpoints'
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_vit_patch8_epoch10.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss.item(),
#             }, checkpoint_path)
#             print(f"Checkpoint saved at epoch {epoch + 1}")

#         if epoch + 1 == 20:
#             checkpoint_dir = './checkpoints'
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_vit_patch8_epoch20.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss.item(),
#             }, checkpoint_path)
#             print(f"Checkpoint saved at epoch {epoch + 1}")

#         if epoch + 1 == 30:
#             checkpoint_dir = './checkpoints'
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_vit_patch8_epoch30.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss.item(),
#             }, checkpoint_path)
#             print(f"Checkpoint saved at epoch {epoch + 1}")

#         if epoch + 1 == 40:
#             checkpoint_dir = './checkpoints'
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_vit_patch8_epoch40.pth')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss.item(),
#             }, checkpoint_path)
#             print(f"Checkpoint saved at epoch {epoch + 1}")

#     print('Finished Training.')

#     # Save the trained model
#     save_directory = './saved_models'
#     os.makedirs(save_directory, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(save_directory, 'vit_cifar100_patch8.pth'))
#     print(f"Model saved to {os.path.join(save_directory, 'vit_cifar100_patch8.pth')}")

#     # Plot average losses
#     plt.plot(avg_losses)
#     plt.xlabel('Batch index')
#     plt.ylabel('Avg. batch loss')
#     plt.title('Training Loss for ViT with Patch Size 8')
#     plt.show()

# # Evaluation function
# def evaluate_model(model, dataloader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in dataloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
#     return accuracy

# # Main function to train and evaluate
# if __name__ == '__main__':
#     train_model()
#     evaluate_model(model, testloader)

# If you need to evaluate a saved checkpoint later:
# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     return checkpoint['epoch']

# load_checkpoint('./checkpoints/checkpoint_vit_patch8_epoch50.pth')
# evaluate_model(model, testloader)
