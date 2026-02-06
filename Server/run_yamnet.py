import numpy as np
import tensorflow as tf
import librosa  # For audio loading
import csv
import os

# --- Configuration ---
TFLITE_MODEL_PATH = 'scripts/yamnet.tflite'
CLASS_MAP_PATH = 'scripts/yamnet_class_map.csv'
SAMPLE_RATE = 16000  # YAMNet requires mono audio at 16kHz

# --- Helper functions ---
def load_labels(class_map_csv_file):
    """Load YAMNet class names from a CSV file."""
    if not os.path.exists(class_map_csv_file):
        raise FileNotFoundError(f"Class map file not found at: {class_map_csv_file}")
    with open(class_map_csv_file, 'r') as f:
        reader = csv.reader(f)
        class_names = [row[2] for row in reader][1:]  # Skip header row
    return class_names

def run_tflite_inference(model_path, waveform):
    """Run inference on a waveform using the TFLite model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    waveform_input_index = input_details[0]['index']
    scores_output_index = output_details[0]['index']

    # Resize input tensor for audio length
    input_size = len(waveform)
    input_tensor = np.asarray(waveform, dtype=np.float32)
    interpreter.resize_tensor_input(waveform_input_index, [input_size], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, input_tensor)

    # Run inference
    interpreter.invoke()
    scores = interpreter.get_tensor(scores_output_index)
    return scores

# --- Main function to be imported ---
def run_yamnet(input_wav_path):
    """
    Run YAMNet analysis on a given audio file.
    Returns top 5 predicted classes with scores.
    """
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"Audio file not found at: {input_wav_path}")

    # Load audio
    waveform, sr = librosa.load(input_wav_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)

    # Run inference
    scores = run_tflite_inference(TFLITE_MODEL_PATH, waveform)

    # Load class labels
    class_names = load_labels(CLASS_MAP_PATH)

    # Average scores over time frames
    mean_scores = np.mean(scores, axis=0)

    # Get top 5 predictions
    top_indices = np.argsort(mean_scores)[::-1][:5]
    results = []
    for i in top_indices:
        results.append({
            "class": class_names[i],
            "score": float(mean_scores[i])
        })
    return results

# --- Optional: run script directly ---
if __name__ == "__main__":
    TEST_WAV_PATH = 'scripts/test_audio.wav'
    try:
        results = run_yamnet(TEST_WAV_PATH)
        print(f"\n✅ Analysis of {TEST_WAV_PATH} Complete.")
        print("--- Top 5 Predicted Audio Events ---")
        for r in results:
            print(f"  - {r['class']}: {r['score']:.4f}")
    except FileNotFoundError as e:
        print(f"\n🛑 Error: {e}")
    except Exception as e:
        print(f"\n🛑 Unexpected error: {e}")
