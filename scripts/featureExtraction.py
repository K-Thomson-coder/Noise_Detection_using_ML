import os
import librosa
import numpy as np
import pandas as pd

def extract_from_dir(parent_dir, label) :
    features = []
    for filename in os.listdir(parent_dir) :
        if filename.endswith(".wav") :
            file_path = os.path.join(parent_dir, filename)

            try :
                y, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs.T, axis=0)

                features.append(np.append(mfccs_mean, label))

            except Exception as e :
                print(f"Error processing {filename} : {e}")

    return features

def build_feature_dataframe(noise_dir, silence_dir) :
    noise_features = extract_from_dir(noise_dir, label=1)
    silence_features = extract_from_dir(silence_dir, label=0)

    all_features = noise_features + silence_features
    df = pd.DataFrame(all_features)

    col_names = [f"mfcc_{i}" for i in range(13)] + ['label']
    df.columns = col_names

    return df

def feature_extraction(file_path) :
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs.T, axis=0)
    return mean_mfccs.reshape(1, -1)

if __name__ == "__main__" :
    noise_dir = "data/wav/noise"
    silence_dir = "data/wav/silence"

    df = build_feature_dataframe(noise_dir, silence_dir)
    df.to_csv("data/features.csv", index=False)
    print("Features saved to data/features.csv")
