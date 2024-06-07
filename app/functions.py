from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Define the mapping from model outputs to disease names
# Replace this with your actual mapping
output_to_disease = {
    0: "Disease A",
    1: "Disease B",
    2: "Disease C",
    # ...
}

def predict_disease(image_path):
    # Load the pre-trained model
    model = load_model('path_to_your_model.h5')

    # Load and process the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    # Make a prediction
    prediction = model.predict(img)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])

    # Return the corresponding disease name
    return output_to_disease[predicted_index]