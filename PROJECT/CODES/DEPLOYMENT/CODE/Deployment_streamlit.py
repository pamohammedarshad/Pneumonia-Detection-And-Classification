import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Define the model loading and prediction function
def predict_class(img):
    # Load the model
    model = tf.keras.models.load_model('model.h5')
    # Preprocess the input image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB format
    img = cv2.resize(img, (224, 224)) # resize to match the input size of the model
    img = img / 255.0 # normalize the pixel values
    # Make a prediction
    pred = model.predict(np.array([img]))[0]
    # Get the class names
    class_names = ['Bacterial', 'Normal', 'Viral']
    # Create the output dictionary
    output = {}
    for i, class_name in enumerate(class_names):
        output[class_name] = float(pred[i])
    return output

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Chest X-ray Classifier")
    # Add an image uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        # Read the image file and show it
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make a prediction on the image
        pred = predict_class(image)
        # Show the prediction results
        st.write("Prediction Results:")
        # Sort the predictions in descending order
        sorted_pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        for class_name, prob in sorted_pred:
            st.write(f"{class_name}: {prob*100:.2f}%")
            st.progress(prob)
            st.text('\n') # Add some spacing between the progress bars
    else:
        # Load the default image
        default_image = cv2.imread('IMG/image.jpg')
        st.image(default_image, caption='Default Image', use_column_width=True)

# Run the app
if __name__ == '__main__':
    app()
