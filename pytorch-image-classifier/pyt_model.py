import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
import matplotlib.pyplot as plt

# Shows window, which halts execution until closed, set to False to hide pop-ups
SHOW_PLOTS = True

# Reproducable results
torch.manual_seed(42) 

device = torch.device("cpu")  # Fallback to regular processor if torch can't detect other chip

# I am running on apple silicon, so I will have torch detect just that chip for this program
if torch.backends.mps.is_built() and torch.backends.mps.is_available():
	device = torch.device("mps")  


def finish_plot(filename):
    """Save the current figure, then show or close it depending on SHOW_PLOTS."""
    plt.savefig(filename)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

fig, axes = plt.subplots(1, 10, figsize=(12, 3))
for i in range(10):
    image = train_loader.dataset[i][0].permute(1, 2, 0)
    denormalized_image = image / 2 + 0.5  # matplotlib requires pixel values to be 0-1
    axes[i].imshow(denormalized_image)
    axes[i].set_title(classes[train_loader.dataset[i][1]])
    axes[i].axis('off')
plt.show()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    # Connect all layers
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten feature map, to be fed into linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


net = CNN()
summary(net, input_size=(1, 3, 32, 32)) 
net.to(device) # Send CNN to GPU



print_every = 500  # 3 updates per epoch for improved readability

loss_function = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 10
epoch_losses = [] # used for loss curve
net.train()
start_time = time.time()

for epoch in range(epochs):

    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % print_every == print_every - 1:
            print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] loss: {running_loss / print_every:.3f}')
            running_loss = 0.0

    epoch_losses.append(epoch_loss / len(train_loader))

training_time = time.time() - start_time
print(f'Finished Training in {training_time:.1f}s')

# Save the trained weights so the model can be reused without retraining
torch.save(net.state_dict(), 'cifar_cnn.pth')
 
# Plot the training loss curve
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average training loss')
plt.title('PyTorch CNN Training Loss')
plt.grid(True)
plt.tight_layout()
finish_plot('train_loss_curve_pyt.png')

def view_result(image, probabilities, actual_label):
    probabilities = probabilities.detach().cpu().numpy().squeeze() 
    predicted_label = int(probabilities.argmax())

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)

    image = image.permute(1, 2, 0)
    denormalized_image = image / 2 + 0.5
    ax1.imshow(denormalized_image)
    ax1.set_title(f'True: {classes[actual_label]}\nPredicted: {classes[predicted_label]}')
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

# Eval mode for predictions + testing
net.eval()

sample_images, sample_labels = next(iter(test_loader))
sample_image = sample_images[0]
sample_label = sample_labels[0].item()

with torch.no_grad():
    output = net(sample_image.unsqueeze(0).to(device))
    probabilities = torch.exp(output)
    
view_result(sample_image, probabilities, sample_label)
finish_plot('prediction_pyt.png')


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
 
        outputs = net(images)
 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print(f'Test accuracy on the {total} test images: {100 * correct / total:.2f}%')
print(f'Training time: {training_time:.1f}s on {device}')