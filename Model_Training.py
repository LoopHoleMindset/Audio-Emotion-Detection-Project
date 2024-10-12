import numpy as np
import pandas as pd
import librosa
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
SAMPLE_RATE = 22050
DURATION = 5  # seconds
NUM_MFCC = 13
file_path = r'C:\Users\Home-PC\Downloads\Kevin\Dataset\archive'

# Function to extract features from audio
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    return np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean))

# Load dataset
def load_dataset(dataset_path):
    features = []
    labels = []
    
    for foldername in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, foldername)):
            for filename in os.listdir(os.path.join(dataset_path, foldername)):
                if filename.endswith('.wav'):
                    file_path = os.path.join(dataset_path, foldername, filename)
                    emotion = foldername  # Assuming folder names are emotion labels
                    mfccs = extract_features(file_path)
                    features.append(mfccs)
                    labels.append(emotion)

    return np.array(features), np.array(labels)

# Build and train the model using XGBoost
def build_and_train_model(dataset_path):
    X, y = load_dataset(dataset_path)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    # After training and evaluating your model
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Save the model and label encoder
    joblib.dump(model, 'emotion_detection_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    return model, X_test, y_test, label_encoder

# Evaluate the model
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    dataset_path = input("Enter the dataset folder path: ")
    model, X_test, y_test, label_encoder = build_and_train_model(dataset_path)
    evaluate_model(model, X_test, y_test, label_encoder)
