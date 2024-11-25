import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import seaborn as sns

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create non-IID data distribution
def create_noniid_data(dataset, num_clients, num_classes):
    client_data = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        class_data = [j for j in range(len(dataset)) if dataset.targets[j] == i]
        np.random.shuffle(class_data)
        for j in range(num_clients):
            client_data[j].extend(class_data[j::num_clients])
    return [Subset(dataset, client) for client in client_data]

# Inner loop: Adaptation phase for FedMeta
def train_inner(model, data_loader, optimizer, inner_steps=5):
    model.train()
    for _ in range(inner_steps):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

# Meta-Update Function (Outer loop)
def meta_update(global_model, client_models, client_weights):
    with torch.no_grad():
        for global_param, *client_params in zip(global_model.parameters(), *(m.parameters() for m in client_models)):
            update = torch.zeros_like(global_param.data)
            for client_param, weight in zip(client_params, client_weights):
                update += weight * (client_param.data - global_param.data)
            global_param.data += update

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = 100 * correct / total
    average_loss = loss / len(data_loader)

    # Calculate F1 score, recall, and precision
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, average_loss, f1, recall, precision, conf_matrix

# Plot function
def plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses, conf_matrix, final_metrics):
    # Plot training accuracy vs epochs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs Rounds')
    plt.legend()

    # Plot training loss vs epochs
    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Rounds')
    plt.legend()

    # Plot testing accuracy vs rounds
    plt.subplot(2, 2, 3)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Rounds')
    plt.legend()

    # Plot testing loss vs rounds
    plt.subplot(2, 2, 4)
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Test Loss vs Rounds')
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

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create non-IID data for clients
num_clients = 10  # Set number of clients to 10
client_datasets = create_noniid_data(mnist_dataset, num_clients, 10)

# Initialize global model
global_model = SimpleNN()
global_model_params = [param.data.clone() for param in global_model.parameters()]

# Training loop
num_rounds = 20  # Set number of epochs (rounds) to 20
inner_steps = 5       # Steps for local adaptation
batch_size = 32

# Create DataLoader for test dataset
test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)

# Metrics tracking lists
train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

# Final test metrics for printing at the end
final_metrics = None

for round_num in range(num_rounds):
    print(f"Round {round_num + 1}/{num_rounds}")

    client_data_sizes = [len(client_data) for client_data in client_datasets]
    total_data_size = sum(client_data_sizes)
    client_weights = [size / total_data_size for size in client_data_sizes]

    # Store client models after adaptation
    adapted_client_models = []

    # Track training accuracy and loss
    round_train_accuracies = []
    round_train_losses = []

    # Clients perform local training
    for client_idx, client_data in enumerate(client_datasets):
        # Initialize client model and optimizer
        client_model = SimpleNN()
        client_model.load_state_dict(global_model.state_dict())
        optimizer = optim.SGD(client_model.parameters(), lr=0.01)
        client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)

        # Inner loop: Client adapts the model locally
        train_inner(client_model, client_loader, optimizer, inner_steps=inner_steps)

        # Store adapted client model
        adapted_client_models.append(client_model)

        # Evaluate client model on its local data (optional, not strictly necessary)
        train_accuracy, train_loss = evaluate_model(client_model, client_loader)[0], evaluate_model(client_model, client_loader)[1]
        round_train_accuracies.append(train_accuracy)
        round_train_losses.append(train_loss)

    # Meta-update the global model using adapted client models (aggregation step happens here)
    meta_update(global_model, adapted_client_models, client_weights)

    # Update global model parameters for the next round
    global_model_params = [param.data.clone() for param in global_model.parameters()]

    # Evaluate the global model on the test dataset
    test_accuracy, test_loss, f1, recall, precision, conf_matrix = evaluate_model(global_model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
    print(f"F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Store metrics for later plotting
    train_accuracies.append(np.mean(round_train_accuracies))
    train_losses.append(np.mean(round_train_losses))
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)

    # Save the final metrics for printing after training is complete
    final_metrics = (test_accuracy, test_loss, f1, recall, precision)

# Plot training and test accuracy
plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses, conf_matrix, final_metrics)

print("Federated Meta-Learning with FedAvg completed.")
