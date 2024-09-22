from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5000"]}})

model = whisper.load_model("base")  

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        data = request.get_json()
        
        file_path = data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400
        
        result = model.transcribe(file_path)
        
        return jsonify({'transcription': result['text']}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start Flask server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
