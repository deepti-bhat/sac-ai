import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Moments Accountant class for privacy budget tracking
class MomentsAccountant:
    def __init__(self, delta=1e-5, max_epsilon=1.0):
        self.epsilon = 0.0
        self.delta = delta
        self.noise_multiplier = 0.1
        self.max_epsilon = max_epsilon  # Set the maximum epsilon value

    def update(self, noise_multiplier, batch_size, steps_per_round):
        sensitivity = 1.0  # sensitivity of gradients
        epsilon_increase = (noise_multiplier ** 2) * (steps_per_round / batch_size) * sensitivity
        self.epsilon += epsilon_increase

        # Ensure epsilon doesn't exceed the maximum epsilon value
        if self.epsilon > self.max_epsilon:
            self.epsilon = self.max_epsilon

    def get_privacy_budget(self):
        return self.epsilon, self.delta

# CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 1024)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Non-IID Data Loader for MNIST
def get_non_iid_data_loader(batch_size=32, num_clients=10):
    transform = transforms.Compose([transforms.ToTensor()])
    fashion_mnist_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    data_per_client = len(fashion_mnist_data) // num_clients
    client_loaders = []

    for client_id in range(num_clients):
        indices = list(range(client_id * data_per_client, (client_id + 1) * data_per_client))
        random.shuffle(indices)
        client_data = torch.utils.data.Subset(fashion_mnist_data, indices)
        client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_loaders.append(client_loader)

    return client_loaders

# Function to shuffle model parameters
def shuffle_model_parameters(client_models):
    """
    Shuffle the parameters of client models before aggregating.
    We shuffle the model weights across clients to anonymize updates.
    """
    shuffled_models = []
    num_params = len(client_models[0].state_dict())

    # Shuffle the state dictionaries of the clients
    for param_idx in range(num_params):
        # Gather the same parameter from each model
        client_param_values = [model.state_dict()[list(model.state_dict().keys())[param_idx]].data for model in client_models]
        random.shuffle(client_param_values)  # Shuffle the values

        # Assign the shuffled parameters back to a new model
        shuffled_model = copy.deepcopy(client_models[0])
        for i, client_model in enumerate(client_models):
            param_name = list(client_model.state_dict().keys())[param_idx]
            shuffled_model.state_dict()[param_name].data = client_param_values[i]

        shuffled_models.append(shuffled_model)

    return shuffled_models

# DP-SGD Training for each client
def dp_sgd_train(model, data_loader, lr=0.01, noise_multiplier=0.1, max_grad_norm=1.0):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # DP-SGD with gradient clipping and noise addition
        for param in model.parameters():
            if param.grad is not None:
                # Clip gradients
                grad_norm = param.grad.norm(2)
                if grad_norm > max_grad_norm:
                    param.grad.mul_(max_grad_norm / (grad_norm + 1e-6))

                # Add noise
                noise = torch.normal(0, noise_multiplier * max_grad_norm, size=param.grad.shape).to(device)
                param.grad.add_(noise)

        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Reptile Meta-learning algorithm on the server
def reptile_on_server(global_model, client_models, client_loaders, inner_steps=5, inner_lr=0.01):
    initial_weights = {name: param.clone() for name, param in global_model.named_parameters()}

    for _ in range(inner_steps):
        for i, client_model in enumerate(client_models):
            # Use the correct data loader for each client
            local_train(client_model, client_loaders[i], inner_lr)

        # Use FedMedian instead of FedAvg
        global_model = federated_median(client_models)

        for name, param in global_model.named_parameters():
            param.data = initial_weights[name] + 0.1 * (param.data - initial_weights[name])

    return global_model


# Local training (without DP) for Reptile
def local_train(model, data_loader, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Federated Averaging (FedAvg) function
def federated_median(client_models):
    global_model = copy.deepcopy(client_models[0])
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            # Get all parameter values from the client models for the current parameter name
            client_param_values = [client_model.state_dict()[name].data for client_model in client_models]

            # Calculate the median of the parameters for the current layer
            median_param = torch.median(torch.stack(client_param_values), dim=0)[0]

            # Set the global model parameter to the median value
            param.data = median_param

    return global_model


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    total_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            loss = nn.CrossEntropyLoss()(outputs, target)

            total_loss += loss.item()
            correct += (predicted == target).sum().item()
            total += target.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)

    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, avg_loss, f1, recall, precision, conf_matrix

# Function to plot all the graphs and print metrics
def plot_and_print_metrics(train_accuracies, train_losses, test_accuracies, test_losses, conf_matrix, final_metrics):
    plt.figure(figsize=(12, 6))

    # Plot training accuracy vs rounds
    plt.subplot(2, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Rounds')
    plt.legend()

    # Plot training loss vs rounds
    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Rounds')
    plt.legend()

    # Plot testing accuracy vs rounds
    plt.subplot(2, 2, 3)
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy vs Rounds')
    plt.legend()

    # Plot testing loss vs rounds
    plt.subplot(2, 2, 4)
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Testing Loss vs Rounds')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Print final metrics
    print(f"Final Test Accuracy: {final_metrics[0]:.2f}%")
    print(f"Final Test Loss: {final_metrics[1]:.4f}")
    print(f"Final F1 Score: {final_metrics[2]:.4f}")
    print(f"Final Recall: {final_metrics[3]:.4f}")
    print(f"Final Precision: {final_metrics[4]:.4f}")

# Main training loop
def main():
    num_rounds = 50  # Number of rounds (previously epochs)
    num_clients = 10  # Number of clients
    inner_steps = 5
    batch_size = 32

    # Generate client loaders
    client_loaders = get_non_iid_data_loader(batch_size=batch_size, num_clients=num_clients)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False)

    global_model = CNNModel().to(device)

    # Initialize the moments accountant for privacy budget tracking
    privacy_accountant = MomentsAccountant(delta=1e-5)

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for round_idx in range(num_rounds):
        client_models = []
        round_train_accuracies = []
        round_train_losses = []

        # DP-SGD training for clients
        for client_loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            # Perform DP-SGD training on each client's data
            train_loss, train_accuracy = dp_sgd_train(local_model, client_loader, lr=0.01, noise_multiplier=0.1, max_grad_norm=1.0)
            round_train_accuracies.append(train_accuracy)
            round_train_losses.append(train_loss)
            client_models.append(local_model)

        avg_train_accuracy = sum(round_train_accuracies) / len(round_train_accuracies)
        avg_train_loss = sum(round_train_losses) / len(round_train_losses)

        train_accuracies.append(avg_train_accuracy)
        train_losses.append(avg_train_loss)

        # Shuffle model parameters before server update
        shuffled_client_models = shuffle_model_parameters(client_models)

        # Reptile update on the server
        global_model = reptile_on_server(global_model, shuffled_client_models, client_loaders, inner_steps=inner_steps)

        # Evaluate the global model
        test_accuracy, test_loss, f1, recall, precision, conf_matrix = evaluate_model(global_model, test_loader)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

        # Print metrics for this round
        print(f'Round {round_idx + 1}')
        print(f'Training Accuracy: {avg_train_accuracy:.2f}% | Training Loss: {avg_train_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}')
        print(f'F1 Score: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}')
        print('-' * 50)

        # Update the privacy budget after each round
        privacy_accountant.update(noise_multiplier=0.1, batch_size=batch_size, steps_per_round=len(client_loaders[0]))

        # Get current privacy budget
        epsilon, delta = privacy_accountant.get_privacy_budget()
        print(f"Privacy Budget - ε: {epsilon:.4f}, δ: {delta}")

    # Plot and print metrics
    plot_and_print_metrics(train_accuracies, train_losses, test_accuracies, test_losses, conf_matrix, (test_accuracy, test_loss, f1, recall, precision))


if __name__ == "__main__":
    main()
