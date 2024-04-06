import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms 
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

def load_model():
    # Load the model
    model =  models.resnet18()  # Don't use pre-trained weights
    num_ftrs = model.fc.in_features  # Adapt the last layer based on your classes
    model.fc = torch.nn.Linear(num_ftrs, out_features=3)
    pretrained_state_dict = torch.load('C:/datascienceprojects/food_image_classification/models/model_0_resnet.pth')
    new_state_dict = {}
    for key, param in pretrained_state_dict.items():
        if key in model.state_dict():  # Filter for matching keys
            new_state_dict[key] = param
    model.load_state_dict(new_state_dict, strict=False)
    model.eval() 
    return model

def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image_tensor = transform(image)  # Use the 'transform' you defined earlier
    image_tensor = image_tensor.unsqueeze_(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

class_mapping = {0: 'pizza', 1: 'steak', 2: 'sushi'}  # Your dictionary

def get_class_label(predicted_index):
    return class_mapping.get(predicted_index, "Unknown")

