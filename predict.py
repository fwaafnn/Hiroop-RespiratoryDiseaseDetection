import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class AudioPredictor:
    def __init__(self, model_path, scaler=None):
        self.model = load_model(model_path)
        self.scaler = scaler if scaler else StandardScaler()

    def preprocess_audio(self, audio_path, duration=42, offset=0.6):
        audio, sample_rate = librosa.load(audio_path, duration=duration, offset=offset, res_type='kaiser_fast')
        features = self.extract_features(audio, sample_rate)
        features = features.reshape(-1, 182)
        
        # Scale and reshape for 1D CNN input
        features = self.scaler.transform(features)
        features = np.expand_dims(features, axis=2)
        return features

    def extract_features(self, audio, sample_rate):
        # Feature extraction functions
        def add_noise(data):
            noise = 0.015 * np.random.uniform() * np.amax(data)
            return data + noise * np.random.normal(size=data.shape[0])

        def stretch_process(data, rate=0.8):
            return librosa.effects.time_stretch(y=data, rate=rate)

        def pitch_process(data, sr, pitch_factor=0.7):
            return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

        # Extract original and augmented features
        features = self.calculate_features(audio, sample_rate)
        features = np.vstack([
            features,
            self.calculate_features(add_noise(audio), sample_rate),
            self.calculate_features(pitch_process(stretch_process(audio), sample_rate), sample_rate)
        ])
        return features

    def calculate_features(self, audio, sample_rate):
        # Calculate various audio features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate))
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40))
        rms = np.mean(librosa.feature.rms(y=audio))
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate))
        return np.hstack([zcr, chroma, mfcc, rms, mel])

    def make_prediction(self, processed_audio):
        # Predict class probabilities and return label with score
        predictions = self.model.predict(processed_audio)
        predicted_class = np.argmax(predictions, axis=1)
        mapping = {0: 'Bronchiectasis', 1: 'Bronchiolitis', 2: 'COPD', 3: 'Healthy', 4: 'Pneumonia', 5: 'URTI'}
        labels = [mapping.get(label) for label in predicted_class]
        return labels, predictions

    def predict_audio(self, audio_path):
        # Load, process, and predict the audio file
        processed_audio = self.preprocess_audio(audio_path)
        labels, predictions = self.make_prediction(processed_audio)
        return {"audio_label": labels, "prediction_score": predictions}
