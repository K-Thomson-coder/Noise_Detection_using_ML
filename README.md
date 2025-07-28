# 🔊 Noise vs Silence Classification



A simple machine learning project to classify noise and silence audio using Python, Librosa, and Scikit-learn, with a real-time audio testing interface built using Streamlit.





---



## 📁 Project Structure


```
├── data/
│   ├── converted/                   # Converted .wav files from raw sources
│   ├── raw/                         # Original downloaded audio files (e.g., .webm)
│   ├── wav/                         # Cleaned and labeled audio files (noise/silence)
│   └── features.csv                 # Extracted MFCC features

├── models/
│   ├── model.pkl                    # Trained ML model
│   └── scaler.pkl                   # Scaler used to normalize features

├── notebooks/
│   └── RandomForest.ipynb           # Experimentation & model training notebook

├── scripts/
│   ├── convert_webm_to_wav_ffmpeg.py   # Script to convert .webm files to .wav using ffmpeg
│   ├── featureExtraction.py            # Script to extract MFCC features and save to CSV
│   └── file.py                         # Contains helper functions to save/load joblib files

├── Noise_Detector.py               # Streamlit app for real-time audio classification
├── README.md                       # Project documentation
├── requirements.txt                # List of required Python packages (You are here)

```



---



## 🚀 Features



🔉 Real-time audio recording with microphone



🧠 ML model trained on MFCC features using RandomForestClassifier



📊 Streamlit UI for testing with live mic input



📁 Clean folder structure for dataset, scripts, and models







---



## ⚙ Requirement



Install dependencies:



pip install -r requirement.txt



Make sure ffmpeg is installed for audio conversion, and your Python version is compatible.





---



## 🛠 How to Use



1. Prepare dataset

Place .wav files into data/wav/noise/ and data/wav/silence/.





2. Extract features



python scripts/featureExtraction.py





3. Train the model



jupyter notebook notebooks/RandomForest.ipynb





4. Run Streamlit app



streamlit run Noise_Detector.py









---



## 📚 Learning Goals



Feature extraction using MFCC (Librosa)



Building and training ML model on audio data



Real-time testing with microphone input



Building a UI with Streamlit







---



## 📦 Dependencies



librosa



numpy



pandas



scikit-learn



sounddevice



streamlit



moviepy



ffmpeg





(See requirements.txt for full list)





---



## ✍ Credits



Built with curiosity and Python by Kangujam Thomson Singh.





---

