import numpy as np
import librosa
import joblib
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox

# Constants
SAMPLE_RATE = 22050
DURATION = 5  # seconds
NUM_MFCC = 13

# Function to extract features from audio data
def extract_features_from_audio(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLE_RATE)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=SAMPLE_RATE)
    
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    return np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean)).reshape(1, -1)

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")

        self.record_button = tk.Button(root, text="Record Audio", command=self.record_audio)
        self.record_button.pack(pady=20)

        self.file_button = tk.Button(root, text="Select Audio File", command=self.select_audio_file)
        self.file_button.pack(pady=20)

        self.result_label = tk.Label(root, text="", font=('Helvetica', 16))
        self.result_label.pack(pady=20)

        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.pack(pady=20)

        # Load the trained model and label encoder
        self.model = joblib.load('emotion_detection_model.pkl')
        self.label_encoder = joblib.load('label_encoder.pkl')

    def record_audio(self):
        messagebox.showinfo("Recording", "Recording will start now.")
        audio_data = self.record(DURATION)
        emotion = self.predict_emotion(audio_data)
        self.result_label.config(text=f'Detected Emotion: {emotion}')

    def record(self, duration):
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
        sd.wait()
        return recording.flatten()

    def select_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            emotion = self.predict_emotion(audio_data)
            self.result_label.config(text=f'Detected Emotion from File: {emotion}')

    def predict_emotion(self, audio_data):
        features = extract_features_from_audio(audio_data)
        prediction = self.model.predict(features)
        emotion_index = prediction[0]
        return self.label_encoder.inverse_transform([emotion_index])[0]

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
