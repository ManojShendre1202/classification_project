from food_image_classifier.components.data_ingestion import FoodDataOrganizer, DownloadData

if __name__ == "__main__":
    choice = input("Do you want to download data? (y/n) ")
    if choice == "y":
        download = DownloadData()
    elif choice == "n":
        data_dir = "./food_data" 
        final_list = ['pizza', 'steak', 'sushi'] 
        amount_to_get  = float(input("Enter the percentage of data to get from the main dataset")) 
        target_dir_name = "./data/sorted_food_images"

        organizer = FoodDataOrganizer(data_dir, final_list)
        label_splits = organizer.get_random_subset(amount=amount_to_get) 
        organizer.organize_subset(label_splits, target_dir_name)