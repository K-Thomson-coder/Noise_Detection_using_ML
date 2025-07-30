# 🔊 Noise vs Silence Classification

A real-time audio classification app that distinguishes between noise and silence using a custom-trained RandomForestClassifier. Built with Python, Librosa, Scikit-learn, and Streamlit, the app features interactive audio recording, visual feedback (waveform and spectrogram), and a clean user interface.


---

## 📁 Project Structure
```
├── data/
│   ├── converted/                   # Converted .wav files from raw sources
│   ├── raw/                         # Original audio files (e.g., .webm)
│   ├── wav/                         # Labeled audio files (noise/silence)
│   └── features.csv                 # Extracted MFCC features

├── models/
│   ├── model.pkl                    # Trained ML model
│   └── scaler.pkl                   # Scaler for feature normalization

├── notebooks/
│   └── RandomForest.ipynb           # Notebook for training/testing

├── scripts/
│   ├── convert_webm_to_wav_ffmpeg.py   # Convert .webm to .wav
│   ├── featureExtraction.py            # Extract MFCCs from audio
│   ├── file.py                         # Save/load functions
│   └── add_bg_image.py                 # Custom Streamlit background loader

├── Noise_Detector.py               # Main Streamlit app
├── README.md                       # Project documentation [You are here]
├── requirements.txt                # Required Python packages

```
---

## 🚀 Features

🎙 Real-time audio recording with microphone<br>
📈 Waveform and spectrogram visualization<br>
🧠 RandomForestClassifier trained on MFCC features<br>
🎛 Duration control and background customization<br>
🧪 Live predictions and class probability output<br>
🧩 Modular and user-friendly interface with collapsible settings


---

## ⚙ Requirements

Install dependencies:
```
pip install -r requirements.txt
```
Make sure ffmpeg is installed and your Python version is compatible.


---

## 🛠 How to Use

1. Prepare dataset

    Place .wav files into:
```
    data/wav/noise/
    data/wav/silence/
```

2. Extract features

```
    python scripts/featureExtraction.py
```

3. Train the model

    Open the Jupyter notebook:
```
    jupyter notebook notebooks/RandomForest.ipynb
```
4. Run the app

```
    streamlit run Noise_Detector.py
```

---

## 📚 Learning Goals
<ul>
    <li>Audio feature extraction using MFCC (Librosa)</li>
    <li>Model training with scikit-learn</li>
    <li>Live audio input handling</li>
    <li>Building a reactive UI with Streamlit</li>
    <li>Custom UI styling with CSS and layout tweaks</li>
</ul>

---

## 📦 Dependencies
```
librosa

numpy

pandas

scikit-learn

sounddevice

streamlit

moviepy

ffmpeg
```

(See requirements.txt for full list)

---

## 🎨 Custom Background (Optional)

This app includes an optional custom background feature using an image stored in an assets/ folder.

To enable the background styling:

1. Create a folder named assets/ in the project root.  
2. Add your background image inside it, e.g., assets/background.png.  
3. The app will automatically load the image if it exists.

> 🔒 *Note:* The assets/ folder is excluded from the repository to keep it lightweight. You can customize your background by adding your own image.


---

✍ Credits

Built with curiosity, sound, and Python
by ***Kangujam Thomson Singh***


---