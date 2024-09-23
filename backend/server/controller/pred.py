import os
import torch
import torchaudio
from transformers import Wav2Vec2Model
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

LABELS = ['Bomb', 'Kidnapping', 'Neutral', 'OtherKeywords', 'Threat']

# Initialize the pretrained Wav2Vec2 model for feature extraction
MODEL_NAME = "facebook/wav2vec2-base-960h"
model_wav2vec = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)

def extract_features(audio_path):
    waveform, _ = torchaudio.load(audio_path)  # Load the waveform
    waveform = waveform.to(device)  # Move waveform to GPU
    with torch.no_grad():
        features = model_wav2vec(waveform).last_hidden_state  # Extract features
    return features.mean(dim=1)  # Aggregate features across time steps

# Define the classifier model structure (must match the trained model)
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

# Initialize the classifier model and load the trained weights
input_dim = 768  # Feature dimension size (adjust according to your feature extraction)
num_classes = len(LABELS)
model = AudioClassifier(input_dim, num_classes).to(device)

# Load the saved model weights
# Load the saved model weights and map to CPU if necessary
model.load_state_dict(torch.load('C:/Users/mccra/OneDrive/Desktop/voicefrontend/backend/server/controller/audio_classifier.pth', map_location=torch.device('cpu')))

model.eval()  # Set the model to evaluation mode

def predict(audio_path):
    # Extract features from the audio file
    features = extract_features(audio_path).unsqueeze(0).to(device)  # Add batch dimension
    # Make predictions
    with torch.no_grad():
        outputs = model(features)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    # Return the predicted label
    return LABELS[predicted_class]

# Example usage: Predict the label of a new audio file
#audio_file_path = 'C:/Users/mccra/Downloads/Sahil-SIH/audios/Neutral/Neutral_1_1.wav'
#predicted_label = predict(audio_file_path)
#if predicted_label:
    print(f"The predicted label for the audio file is: {predicted_label}")
#else:
#    print("Prediction failed.")

# Print available audio backends
#print("Available audio backends:")
#print(torchaudio.list_audio_backends())


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}}) 

@app.route('/detect', methods=['POST'])
def transcribe_audio():
    try:
        data = request.get_json()
        
        file_path = data.get('file_path')
        print(file_path)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400
        
        result = predict(file_path)
        
        return jsonify({'transcription': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)