import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the GRU model
gru_model = load_model('diagnosis_GRU_CNN_1.h5')  # Replace with the actual path

# Function to predict class
def predict_class(audio_file_path, gru_model, features=52, soundDir=''):
    val = []
    classes = ["COPD", "Bronchiolitis", "Pneumonia", "URTI", "Healthy"]
    data_x, sampling_rate = librosa.load(audio_file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=features).T, axis=0)
    val.append(mfccs)
    val = np.expand_dims(val, axis=1)
    prediction = classes[np.argmax(gru_model.predict(val))]
    return prediction

# Streamlit app
def main():
    st.title("Respiratory Diseases Classification App")

    # File upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Make prediction when the user clicks the button
        if st.button("Predict"):
            # Save the uploaded file to a temporary location
            with open("temp_file.wav", "wb") as f:
                f.write(uploaded_file.read())

            # Get the prediction using the temporary file
            prediction = predict_class("temp_file.wav", gru_model)
            st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
