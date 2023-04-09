import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def forward_with_activations(model, x):
    activations = {'input': x}
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            x_pre = x
            x = layer(x)
            activations[name + '_pre'] = x_pre
            activations[name] = x
        elif isinstance(layer, nn.ReLU):
            x = layer(x)
            activations[name] = x
    return x, activations

def hebbian_update(model, labels, activations, learning_rate, clip_range=(-1, 1)):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            post_activation = activations[name]
            if name == 'fc1':
                pre_activation = activations['input']
            else:
                pre_activation = activations['relu']
            if name == 'fc2':
                labels = labels.float()
                predicted = torch.argmax(post_activation, dim=1)
                max_diff = activations['fc2'].size(1) - 1
                reward = (max_diff - torch.abs(predicted - labels)) / max_diff  # Use a non-negative reward
                reward = reward.unsqueeze(1)
            else:
                reward = activations['fc2_pre']
            outer_product = torch.bmm(pre_activation.unsqueeze(2), post_activation.unsqueeze(1))
            reward_expanded = reward.view(reward.shape[0], 1, -1).expand_as(outer_product)
            weight_update = torch.mean(reward_expanded * outer_product, dim=0)
            weight_update = torch.clamp(weight_update.t(), clip_range[0], clip_range[1])
            layer.weight.data += learning_rate * weight_update

# Load and preprocess the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
model = SimpleNN().to(device)

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
hebbian_learning_rate = 0.0005
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Flatten the input images
        inputs = inputs.view(inputs.shape[0], -1)

        optimizer.zero_grad()

        outputs, activations = forward_with_activations(model, inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Hebbian update
        hebbian_update(model, labels, activations, hebbian_learning_rate)

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

print("Finished Training")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # Flatten the input images
        images = images.view(images.shape[0], -1)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))