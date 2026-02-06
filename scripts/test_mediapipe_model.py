import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import base64
from mediapipe_audio_classifier import classify_audio_with_mediapipe

# Path to a small WAV file for testing
test_audio_path = "scripts/test_audio.wav"

# Read and encode as base64
with open(test_audio_path, "rb") as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

# Run the model
print("🚀 Running YAMNet classification in terminal...")
result = classify_audio_with_mediapipe(audio_base64, filename="test_audio.wav")

print("\n📊 Final JSON Output:\n", result)
 