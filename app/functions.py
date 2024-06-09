from keras.models import load_model
from keras.preprocessing import image
import numpy as np

output_to_disease = {
    0: "Bacterial Pneumonia",
    1: "Corona Virus Disease",
    2: "Normal",
    3: "Tuberculosis",
    4: "Viral Pneumonia"
}

def predict_disease(image_path):
    # Load the pre-trained model
    model = load_model('../model/lung_disease_detection_model.h5')

    # Load and process the image
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255

    # Make a prediction
    prediction = model.predict(img)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction[0])

    # Return the corresponding disease name
    return output_to_disease[predicted_index]