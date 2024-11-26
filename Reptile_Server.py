import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def get_non_iid_data_loader(batch_size=32, num_clients=10, num_classes=10):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    data_per_client = int(len(mnist_data) / num_clients)
    client_loaders = []

    for client_id in range(num_clients):
        indices = [i for i in range(client_id * data_per_client, (client_id + 1) * data_per_client)]
        random.shuffle(indices)
        client_data = torch.utils.data.Subset(mnist_data, indices)
        client_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_loaders.append(client_loader)

    return client_loaders

# Local training (no Reptile on clients)
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

# Reptile Meta-learning algorithm on the server
def reptile_on_server(global_model, client_models, inner_steps=5, inner_lr=0.01):
    initial_weights = {name: param.clone() for name, param in global_model.named_parameters()}

    for _ in range(inner_steps):
        for client_model in client_models:
            local_train(client_model, client_model.data_loader, inner_lr)

        global_model = federated_averaging(client_models)

        for name, param in global_model.named_parameters():
            param.data = initial_weights[name] + 0.1 * (param.data - initial_weights[name])

    return global_model

# Federated Averaging (FedAvg) function
def federated_averaging(client_models):
    global_model = copy.deepcopy(client_models[0])
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            param.data = torch.mean(torch.stack([client_model.state_dict()[name].data for client_model in client_models]), dim=0)
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
    num_rounds = 20  # Number of rounds (previously epochs)
    num_clients = 10  # Number of clients
    inner_steps = 5
    batch_size = 32

    client_loaders = get_non_iid_data_loader(batch_size=batch_size, num_clients=num_clients)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False)

    global_model = CNNModel().to(device)

    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for round_idx in range(num_rounds):
        client_models = []
        round_train_accuracies = []
        round_train_losses = []

        for client_loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_model.data_loader = client_loader
            train_loss, train_accuracy = local_train(local_model, client_loader)
            round_train_accuracies.append(train_accuracy)
            round_train_losses.append(train_loss)
            client_models.append(local_model)

        avg_train_accuracy = sum(round_train_accuracies) / len(round_train_accuracies)
        avg_train_loss = sum(round_train_losses) / len(round_train_losses)

        train_accuracies.append(avg_train_accuracy)
        train_losses.append(avg_train_loss)

        global_model = reptile_on_server(global_model, client_models, inner_steps=inner_steps)

        test_accuracy, test_loss, f1, recall, precision, conf_matrix = evaluate_model(global_model, test_loader)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)

        print(f'Round {round_idx + 1}')
        print(f'Training Accuracy: {avg_train_accuracy:.2f}% | Training Loss: {avg_train_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}')
        print(f'F1 Score: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}')
        print('-' * 50)

    plot_and_print_metrics(train_accuracies, train_losses, test_accuracies, test_losses, conf_matrix, (test_accuracy, test_loss, f1, recall, precision))

if __name__ == "__main__":
    main()
