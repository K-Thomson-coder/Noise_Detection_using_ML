import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from scripts import file
import tempfile
import scipy.io.wavfile

model = file.load_file("models/model.pkl")
scaler = file.load_file("models/scaler.pkl")

label_map = {0 : 'Silence', 1 : 'Noise'}

st.title("Real-Time Noise Detector")

duration = st.slider("Recording duration (seconds)", 1, 5, 3)

if st.button("Record & Predict") :
    st.write("🎤 Recording...")
    fs = 22050 #sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f :
        scipy.io.wavfile.write(f.name, fs, recording)
        file_path = f.name

    st.audio(file_path, format="audio/wav")

    try :
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

        scaled_mfccs = scaler.transform(mfccs_mean)

        probs = model.predict_proba(scaled_mfccs)[0]
        prediction = np.argmax(probs)


        st.write(f"## 🎧 Prediction : `{label_map[prediction]}`")
        st.markdown("### 📊 Class Probabilities ")

        for i, prob in enumerate(probs) :
            st.write(f"**{label_map[i]}**")
            st.progress(min(max(prob, 0.0), 1.0))

    except Exception as e :
        st.error(f"Error processing audio : {e}")














