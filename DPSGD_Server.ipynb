import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Moments Accountant class
class MomentsAccountant:
    def __init__(self, delta=1e-5):
        self.epsilon = 0.0
        self.delta = delta
        self.noise_multiplier = 0.1
        self.total_gradients = 0

    def update(self, noise_multiplier, batch_size, steps_per_round):
        """
        Updates the privacy budget epsilon based on the noise multiplier
        and number of steps.
        """
        sensitivity = 1.0  # sensitivity of gradients
        epsilon_increase = (noise_multiplier ** 2) * (steps_per_round / batch_size) * sensitivity
        self.epsilon += epsilon_increase

    def get_privacy_budget(self):
        return self.epsilon, self.delta

# Define a larger CNN Model to increase capacity
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Additional layer
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Increased fully connected layer size
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Non-IID data split function
def non_iid_split(dataset, num_clients):
    indices_by_class = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(dataset):
        indices_by_class[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]
    for i in range(10):  # Each class
        np.random.shuffle(indices_by_class[i])
        for j in range(num_clients):
            class_part = len(indices_by_class[i]) // num_clients
            client_indices[j].extend(indices_by_class[i][j * class_part: (j + 1) * class_part])

    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    return client_datasets

# DP-SGD Training Function with improved privacy budget tracking
def dp_sgd_train(model, data_loader, lr=0.01, noise_multiplier=0.1, max_grad_norm=1.0, epochs=2, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train()
    epoch_train_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # DP-SGD with gradient clipping and noise addition
            for param in model.parameters():
                if param.grad is not None:
                    # Clip gradients (can be per-layer)
                    grad_norm = param.grad.norm(2)
                    if grad_norm > max_grad_norm:
                        param.grad.mul_(max_grad_norm / (grad_norm + 1e-6))

                    # Add noise (with smaller noise multiplier helps reduce instability)
                    noise = torch.normal(0, noise_multiplier * max_grad_norm, size=param.grad.shape).to(device)
                    param.grad.add_(noise)

            optimizer.step()

            # Calculate the loss and accuracy
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_train_loss = epoch_loss / len(data_loader)  # Average loss over all batches
        epoch_train_accuracy = 100 * correct / total  # Accuracy over the epoch

    return model, epoch_train_loss, epoch_train_accuracy

# Federated Training Loop with improved privacy budget tracking
def train_federated(num_clients=10, num_rounds=15, lr=0.01, noise_multiplier=0.1, max_grad_norm=1.0, device='cpu'):
    # Load MNIST data and split into non-IID datasets for clients
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    client_datasets = non_iid_split(mnist_data, num_clients)

    # Initialize global model
    global_model = CNNModel().to(device)

    # Initialize the moments accountant for privacy budget tracking
    privacy_accountant = MomentsAccountant(delta=1e-5)

    all_train_loss = []
    all_train_accuracy = []
    all_test_loss = []
    all_test_accuracy = []

    for round in range(num_rounds):
        print(f"--- Round {round+1} ---")
        client_models = []

        # Dynamic noise multiplier reduction (optional, you can tune this)
        noise_multiplier = max(0.05, noise_multiplier - 0.05)  # Reduce noise decay

        round_train_loss = 0.0
        round_train_accuracy = 0.0

        for client_idx in range(num_clients):
            # Load data for this client
            client_loader = DataLoader(client_datasets[client_idx], batch_size=64, shuffle=True)
            local_model = copy.deepcopy(global_model)

            # Perform DP-SGD training on each client's data
            local_model, epoch_train_loss, epoch_train_accuracy = dp_sgd_train(local_model, client_loader,
                                                                              lr=lr, noise_multiplier=noise_multiplier,
                                                                              max_grad_norm=max_grad_norm, device=device)

            # Collect client models after local updates
            client_models.append(local_model)
            round_train_loss += epoch_train_loss
            round_train_accuracy += epoch_train_accuracy

        # Federated Averaging with optional weights based on client data size
        global_model = federated_average(client_models).to(device)

        # Update the privacy budget after each round (based on moments accountant)
        privacy_accountant.update(noise_multiplier, batch_size=64, steps_per_round=len(client_datasets[0]) // 64)

        # Get current privacy budget
        epsilon, delta = privacy_accountant.get_privacy_budget()
        print(f"Round {round+1} complete. Privacy Budget - ε: {epsilon:.4f}, δ: {delta}")

        # Calculate training loss and accuracy on the global model
        global_train_loader = DataLoader(mnist_data, batch_size=64, shuffle=True)
        global_train_loss, global_train_accuracy = calculate_metrics(global_model, global_train_loader, device)

        # Test accuracy after federated averaging
        test_loss, test_accuracy, precision, recall, f1, all_preds, all_targets = test(global_model, device)

        # Print round metrics (server side)
        print(f"Round {round+1} - Train Loss: {global_train_loss:.4f}, Train Accuracy: {global_train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        # Store round metrics for later plotting
        all_train_loss.append(round_train_loss / num_clients)
        all_train_accuracy.append(round_train_accuracy / num_clients)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)

    return global_model, all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_preds, all_targets

# Federated Averaging with optional weights based on client data size
def federated_average(models):
    global_model = copy.deepcopy(models[0])
    for k in global_model.state_dict().keys():
        global_model.state_dict()[k] = torch.stack([m.state_dict()[k] for m in models], 0).mean(0)
    return global_model

# Testing the Global Model
def test(model, device='cpu'):
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')

    return test_loss / len(test_loader.dataset), accuracy, precision, recall, f1, all_preds, all_targets

# Calculate metrics for training data (e.g., for the global model)
def calculate_metrics(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += F.cross_entropy(output, target, reduction='sum').item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    loss /= len(data_loader.dataset)
    return loss, accuracy

# Plot metrics function
def plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses):
    rounds = len(train_accuracies)

    # Plot Training Accuracy vs Rounds
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(range(1, rounds + 1), train_accuracies, label='Training Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs Rounds')

    # Plot Training Loss vs Rounds
    plt.subplot(2, 2, 2)
    plt.plot(range(1, rounds + 1), train_losses, label='Training Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Rounds')

    # Plot Testing Accuracy vs Rounds
    plt.subplot(2, 2, 3)
    plt.plot(range(1, rounds + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy vs Rounds')

    # Plot Testing Loss vs Rounds
    plt.subplot(2, 2, 4)
    plt.plot(range(1, rounds + 1), test_losses, label='Testing Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Testing Loss vs Rounds')

    plt.tight_layout()
    plt.show()

# Confusion Matrix Plot
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Run federated training with DP-SGD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model, all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_preds, all_targets = train_federated(
    num_clients=10, num_rounds=20, lr=0.01, noise_multiplier=0.1, max_grad_norm=1.0, device=device
)

# Final test
test_loss, test_accuracy, precision, recall, f1, all_preds, all_targets = test(global_model, device)

# Print final test metrics
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Plot confusion matrix
cm = confusion_matrix(all_targets, all_preds)
plot_confusion_matrix(cm)

# Plot metrics
plot_metrics(all_train_accuracy, all_train_loss, all_test_accuracy, all_test_loss)
