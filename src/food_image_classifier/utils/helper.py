import matplotlib.pyplot as plt

import torch

def plot_training_curves(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    """
    Plots training and validation accuracy and loss curves.

    Args:
        train_acc_history (list): A list of training accuracy values per epoch.
        val_acc_history (list): A list of validation accuracy values per epoch.
        train_loss_history (list): A list of training loss values per epoch.
        val_loss_history (list): A list of validation loss values per epoch.
    """

    num_epochs = len(train_acc_history)

    plt.figure(figsize=(10, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training')
    plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training')
    plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_trained_model(model, save_path):
    """
    Saves a trained PyTorch model to the specified location.

    Args:
        model (nn.Module): The trained ResNet18 model.
        save_path (str): The path where the model should be saved, 
                            including the desired filename (e.g., 'trained_resnet18.pth')
    """

    # Important: Ensure the parent directory in the save_path exists

    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to: {save_path}")

import torch

def load_model(model_path):
    """Loads a trained PyTorch model from a saved file.

    Args:
        model_path (str): Path to the saved model file (.pth).

    Returns:
        nn.Module: The loaded PyTorch model.
    """

    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

