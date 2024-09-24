import os
import torch
import torchaudio
from transformers import Wav2Vec2Model
import numpy as np
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set device to CPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Load the pretrained Wav2Vec2 model for feature extraction
MODEL_NAME = "facebook/wav2vec2-base-960h"
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)

# Load the trained audio classifier model
class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def load_model(model_path, input_dim, num_classes):
    model = AudioClassifier(input_dim, num_classes).to(device)  # Ensure model is on CPU
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model state on CPU
    model.eval()  # Set to evaluation mode
    return model

def trim_silence(waveform, threshold=0.01, padding=0.1):
    """Trims leading and trailing silence from the waveform."""
    waveform_np = waveform.cpu().numpy()[0]  # Take the first channel
    non_silent_indices = np.where(np.abs(waveform_np) > threshold)[0]

    if len(non_silent_indices) == 0:  # All silent
        return waveform

    start = max(0, non_silent_indices[0] - int(padding * 16000))
    end = min(len(waveform_np), non_silent_indices[-1] + int(padding * 16000))

    trimmed_waveform = torch.tensor(waveform_np[start:end]).unsqueeze(0)
    return trimmed_waveform

def extract_features(audio_path):
    waveform, _ = torchaudio.load(audio_path)  # Load the waveform
    waveform = trim_silence(waveform)  # Trim silence
    waveform = waveform.to(device)  # Move waveform to CPU
    with torch.no_grad():
        features = wav2vec_model(waveform).last_hidden_state  # Extract features
    return features.mean(dim=1)  # Aggregate features across time steps

def predict(model, audio_path, label_mapping):
    features = extract_features(audio_path)  # Extract features from the audio
    features = features.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(features.to(device))  # Ensure features are on CPU
        _, predicted = torch.max(outputs, 1)
    return label_mapping[predicted.item()]

# Load the model
model_path = 'audio_classifier.pth'
input_dim = 768  # Feature dimension size (based on Wav2Vec2 model)
num_classes = 5  # Number of labels
label_mapping = ['Bomb', 'Kidnapping', 'Neutral', 'OtherKeywords', 'Threat']
model = load_model(model_path, input_dim, num_classes)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}}) 

@app.route('/detect', methods=['POST'])
def transcribe_audio():
    try:
        data = request.get_json()
        
        file_path = data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400
        
        result = predict(model,file_path,label_mapping)
        
        return jsonify({'transcription': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)