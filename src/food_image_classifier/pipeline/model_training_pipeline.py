from src.food_image_classifier.components.model_training import ModelTraining
from src.food_image_classifier.utils.helper import plot_training_curves

class PreTraininedModel:
    def __init__(self, model_name, num_classes):
        self.model = model_name
        self.num_classes = num_classes

    def TrainModel(self, model, criterion, device, dataloaders, dataset_sizes, 
                optimizer, scheduler, num_epochs):
        trainer = ModelTraining(self.model, self.num_classes)
        model, train_loss_history, train_acc_history, val_loss_history, val_acc_history = trainer.train_model(model=model, 
                                                                                        criterion=criterion, 
                                                                                        device=device, 
                                                                                        dataloaders=dataloaders, 
                                                                                        dataset_sizes=dataset_sizes, 
                                                                                        optimizer=optimizer, 
                                                                                        scheduler=scheduler, 
                                                                                        num_epochs=num_epochs)
        return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history
    
    def PlotCurves(self, train_loss_history, train_acc_history, val_loss_history, val_acc_history):
        plot_training_curves(train_loss_history, train_acc_history, val_loss_history, val_acc_history)

