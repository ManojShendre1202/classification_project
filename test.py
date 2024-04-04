import torch
from torch.optim import lr_scheduler
from torch import nn
from food_image_classifier.components.data_transformation import DataTransformations
from food_image_classifier.components.model_training import get_pretrained_model, train_model
from food_image_classifier.utils.helper import plot_training_curves, save_trained_model

# Hyperparameters
data_dir = './data/sorted_food_images' 
batch_size = 32
num_epochs = 20
model_name = 'resnet18'

# Data Transformations
data_transform = DataTransformations(batch_size)
dataloaders, dataset_sizes, class_names = data_transform.get_data_loaders(data_dir)

# Get model, loss, optimizer 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_pretrained_model(model_name, len(class_names))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 

# Train!
model, train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(model=model, 
                                                                                        criterion=criterion, 
                                                                                        device=device, 
                                                                                        dataloaders=dataloaders, 
                                                                                        dataset_sizes=dataset_sizes, 
                                                                                        optimizer=optimizer, 
                                                                                        scheduler=scheduler, 
                                                                                        num_epochs=num_epochs) 

# plot curves
plot_training_curves(train_loss_history, train_acc_history, val_loss_history, val_acc_history)

# saving the model
save_path = 'C:/datascienceprojects/food_image_classification/models/model_2_resnet18.pth'  # Replace with your desired path
save_trained_model(model, save_path)    

