import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Meta-Learning Inner Loop (Local Update on Client)
def maml_inner_loop(model, data, target, inner_lr=0.01, steps=1, device='cpu'):
    temp_model = copy.deepcopy(model).to(device)
    optimizer = optim.SGD(temp_model.parameters(), lr=inner_lr, momentum=0.9)

    for _ in range(steps):
        optimizer.zero_grad()
        output = temp_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    return temp_model.cpu()

# Federated Averaging to Aggregate Models
def federated_average(models):
    global_model = copy.deepcopy(models[0])
    for k in global_model.state_dict().keys():
        global_model.state_dict()[k] = torch.stack([m.state_dict()[k] for m in models], 0).mean(0)
    return global_model

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

# Main Training Loop
def train_federated(num_clients=10, num_rounds=20, inner_steps=1, inner_lr=0.01, device='cpu'):
    # Load MNIST data and split into non-IID datasets for clients
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    client_datasets = non_iid_split(mnist_data, num_clients)

    # Initialize global model
    global_model = CNNModel().to(device)

    # Store metrics
    all_train_accuracy = []
    all_train_loss = []
    all_test_accuracy = []
    all_test_loss = []
    all_preds = []
    all_targets = []

    for round in range(num_rounds):
        print(f"--- Round {round+1} ---")
        client_models = []

        round_train_loss = 0.0
        round_train_accuracy = 0.0

        for client_idx in range(num_clients):
            # Load data for this client
            client_loader = DataLoader(client_datasets[client_idx], batch_size=64, shuffle=True)
            local_model = copy.deepcopy(global_model)

            # Perform inner-loop (meta-learning) on each client's data
            for data, target in client_loader:
                data, target = data.to(device), target.to(device)
                local_model = maml_inner_loop(local_model, data, target, inner_lr=inner_lr, steps=inner_steps, device=device)

            # Collect client models after local updates
            client_models.append(local_model)

        # Aggregate client models to update global model
        global_model = federated_average(client_models).to(device)

        # Calculate training loss and accuracy on the global model (not used in MAML but tracked for visualization)
        global_train_loader = DataLoader(mnist_data, batch_size=64, shuffle=True)
        train_loss, train_accuracy = calculate_metrics(global_model, global_train_loader, device)
        all_train_loss.append(train_loss)
        all_train_accuracy.append(train_accuracy)

        # Test the model
        test_loss, test_accuracy, precision, recall, f1, preds, targets = test(global_model, device)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_preds.extend(preds)
        all_targets.extend(targets)

        print(f"Round {round+1} complete.")

    # Return metrics and predictions for confusion matrix
    return global_model, all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_preds, all_targets

# Calculate metrics (loss, accuracy) on a given data loader
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

# Testing the Global Model with Precision, Recall, F1 Score, Confusion Matrix
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

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return test_loss / len(test_loader.dataset), accuracy, precision, recall, f1, all_preds, all_targets

# Confusion Matrix Plot
def plot_confusion_matrix(predictions, targets):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Separate Plotting functions for each metric
def plot_train_loss(train_loss):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_loss) + 1), train_loss)
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Rounds')
    plt.show()

def plot_test_loss(test_loss):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(test_loss) + 1), test_loss)
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Rounds')
    plt.show()

def plot_train_accuracy(train_accuracy):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Rounds')
    plt.show()

def plot_test_accuracy(test_accuracy):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(test_accuracy) + 1), test_accuracy)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Rounds')
    plt.show()

# Run federated training with meta-learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model, all_train_loss, all_train_accuracy, all_test_loss, all_test_accuracy, all_preds, all_targets = train_federated(num_clients=10, num_rounds=20, inner_steps=1, inner_lr=0.01, device=device)

# Final test and display results
final_test_loss = all_test_loss[-1]
final_test_accuracy = all_test_accuracy[-1]
final_precision = precision_score(all_targets, all_preds, average='weighted')
final_recall = recall_score(all_targets, all_preds, average='weighted')
final_f1 = f1_score(all_targets, all_preds, average='weighted')

print(f"Final Test Loss: {final_test_loss:.4f}")
print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
print(f"Final Test Precision: {final_precision:.4f}")
print(f"Final Test Recall: {final_recall:.4f}")
print(f"Final Test F1-Score: {final_f1:.4f}")

# Plot metrics
plot_train_loss(all_train_loss)
plot_test_loss(all_test_loss)
plot_train_accuracy(all_train_accuracy)
plot_test_accuracy(all_test_accuracy)

# Plot confusion matrix at the end of training
plot_confusion_matrix(all_preds, all_targets)
