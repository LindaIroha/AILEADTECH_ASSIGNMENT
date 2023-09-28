import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

#Create the title for the App
st.title ('Covid_19 Radiographic Image Classification')
st.write('Upload a Radiographic Image and we will predict whether it is Normal, Covid_19 or Viral Pneumonia')

#Create a File Uploader
uploaded_file =st.file_uploader('Upload an image..', type= ['jpg', 'jpeg', 'png'])

#Check if the image is uploaded
if uploaded_file is not None:
    #display the image
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded Image')
    st.write('')

    #Preprocess the image
    img = np.array(image)

    # Resize the image using PIL (Pillow)
    img = Image.fromarray(img).resize((128, 128))
    img = np.array(img)

     # Ensure the image has 3 color channels (RGB)
    if img.shape[-1] != 3:
        # If the image doesn't have 3 channels, convert it to RGB
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img)

    #Define the dimension
    img =  np.expand_dims(img, axis = 0)
    #st.write(f'{img.shape}')

    #Load the trained  model
    model = load_model('C:/Users/44773/OneDrive/Desktop/Radio_images/radio_images4.h5')

    #Make Predictions
    predictions = model.predict(img)
    # Define class labels
    class_labels = ['Viral Pneumonia', 'Covid', 'Normal']

# Check if the predicted class probabilities are above a certain threshold
    threshold = 0.5  # Adjust the threshold as needed

# Create a list to store labels with probabilities above the threshold
    selected_labels = [class_labels[i] for i, prob in enumerate(predictions[0]) if prob > threshold]

# Check if there are selected labels
    if selected_labels:
    # If there are selected labels, join them into a single string
        label = ', '.join(selected_labels)
    else:
        label = 'Uncertain'

# Display the prediction
    st.write(f'## Predicted Image: {label}')