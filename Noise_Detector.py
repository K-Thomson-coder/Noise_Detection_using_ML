import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
from scripts import file, add_bg_image
import tempfile
import scipy.io.wavfile
import matplotlib.pyplot as plt

#config
st.set_page_config(page_title="Noise Detector", page_icon="🔊", layout="centered")

model = file.load_file("models/model.pkl")
scaler = file.load_file("models/scaler.pkl")
label_map = {0 : 'Silence', 1 : 'Noise'}
    
# Setting bar
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b :
    with st.expander("⚙ Settings", expanded=False) :
        col1, col2 = st.columns([2, 2])

        with col1 :
            show_waveform = st.checkbox("Waveform", value=False)
        with col2 :
            show_spectrogram = st.checkbox("Spectrogram", value=False)
        
        col3 = st.columns(2)
        with col3[0] :
            bg_choice = st.selectbox("Background", ["None", "BG-1", "BG-2"])

        duration = st.slider("Recording Duration (sec)", 1, 10, 5)

#Header
st.markdown("<h1 style='text-align:center; color:#4A90E2;'>🔊 Real-Time Noise Detector</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>This app records 5 seconds of audio and classifies it as <b>Noise</b> or <b>Silence</b>.<br>"
    "Powered by a custom-trained <i>RandomForestClassifier</i>.</p>", unsafe_allow_html=True)

st.markdown("---")

if bg_choice == "BG-1" :
    add_bg_image.set_bg_img("assets/bg_3.png") 
elif bg_choice == "BG-2" :
    add_bg_image.set_bg_img("assets/bg_2.png")

#Button
if st.button("🎙 Record & Predict") :
    with st.spinner("🎤 Recording 5 seconds of audio...") :
        fs = 22050 #sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

    #Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f :
        scipy.io.wavfile.write(f.name, fs, recording)
        file_path = f.name

    st.audio(file_path, format="audio/wav")
    
    #Process audio
    try :
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

        scaled_mfccs = scaler.transform(mfccs_mean)

        probs = model.predict_proba(scaled_mfccs)[0]
        prediction = np.argmax(probs)
        predicted_label = label_map[prediction]

        #Show Prediction
        st.markdown(
            f"""
            <div style='
                padding: 1rem;
                background-color: #222;
                border-radius: 0.5rem;
                color: #0f0;
                font-size: 1.3rem;
                text-align: center;
                border: 2px solid #0f0;
                margin-top: 20px;
                width: fit-content;
                margin-left: auto;
                margin-right: auto;
            '>
            🎧 Detected Sound : <strong>{label_map[prediction]}</strong>
            </div><br>
            """,
            unsafe_allow_html=True
        )

        #Show probability
        st.markdown("### 📊 Class Probabilities ")
        for i, prob in enumerate(probs) :
            percent = prob * 100
            label = label_map[i]
            st.write(f"{label}")
            bar_col, val_col = st.columns([5, 1])
            with bar_col :
                st.progress(min(int(percent), 100))
            with val_col :
                st.write(f"<b>{percent:.2f}%</b>", unsafe_allow_html=True)

        if show_waveform :
            st.markdown("#### Waveform")
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax, color='cornflowerblue')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig, use_container_width=True)

        if show_spectrogram :
            st.markdown("#### Spectogram")
            fig, ax = plt.subplots()
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            img = librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
            fig.colorbar(img, ax=ax, format='%+2.0fdB')
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig, use_container_width=True)

    except Exception as e :
        st.error(f"⚠ Error processing audio : {e}")

#Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ♥ using Streamlit<br>"
    "<a href='https://github.com/K-Thomson-coder' target='_blank'>View on GitHub</a></p>", unsafe_allow_html=True
    )










