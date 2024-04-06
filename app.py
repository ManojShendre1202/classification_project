from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from food_image_classifier.utils.helper import predict_image as predict_food_image 
from food_image_classifier.utils.helper import load_model

model = load_model()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image display and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("")  # Space
    st.write("Classifying...")
    class_index = predict_food_image(uploaded_file, model, transform=transform)
    class_labels = ['pizza', 'steak', 'sushi']  
    predicted_label = class_labels[class_index]
    st.write(f"Predicted class: {predicted_label}")
