
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from food_image_classifier.utils.helper import load_model

# Load your model
model_path = 'C:/datascienceprojects/food_image_classification/research/model_1_resnet18.png'
model = load_model(model_path) 

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Streamlit App
st.title("Image Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')

    if st.button('Predict'):
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)
        _, predicted = torch.max(output.data, 1)

        # Get class label (assuming you have a mapping from indices to labels)
        your_labels = {0: 'pizza', 1:'steak',3:'sushi'}
        class_label = your_labels[predicted.item()] 

        st.write(f"Predicted Class: {class_label}")
