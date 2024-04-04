from food_image_classifier.components.data_transformation import DataTransformations


def DataTransformationsPipeline(batch_size, data_dir):
    transformers = DataTransformations(batch_size)
    dataloaders, dataset_sizes, class_names = transformers.get_data_loaders(data_dir)
    return dataloaders, dataset_sizes, class_names