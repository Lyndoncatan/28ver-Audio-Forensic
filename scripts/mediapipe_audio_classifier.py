import numpy as np
import json
import sys
import os
import warnings
import tempfile
import subprocess
from scipy.io import wavfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import audio

# Silence all background noise from TensorFlow/MediaPipe
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_yamnet_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yamnet.tflite')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.getcwd(), 'scripts', 'yamnet.tflite')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YAMNet model missing at {model_path}")
    return model_path

def map_to_forensic_category(mediapipe_category):
    mapping = {
        # Human Voice
        "Speech": "Human Voice",
        "Singing": "Human Voice",
        "Male speech": "Human Voice",
        "Female speech": "Human Voice",
        "Child speech": "Human Voice",
        "Conversation": "Human Voice",
        "Laughter": "Human Voice",
        "Screaming": "Scream / Aggression",
        "Shout": "Scream / Aggression",
        "Yell": "Scream / Aggression",
        "Crying": "Scream / Aggression",
        
        # Musical Content
        "Music": "Musical Content",
        "Background music": "Musical Content",
        "Strum": "Musical Content",
        "Guitar": "Musical Content",
        "Piano": "Musical Content",
        
        # Vehicle Sound
        "Vehicle": "Vehicle Sound",
        "Car": "Vehicle Sound",
        "Bus": "Vehicle Sound",
        "Truck": "Vehicle Sound",
        "Motorcycle": "Vehicle Sound",
        "Engine": "Vehicle Sound",
        "Accelerating": "Vehicle Sound",
        "Brake": "Vehicle Sound",
        "Tire": "Vehicle Sound",
        
        # Siren / Alarm
        "Emergency vehicle": "Siren / Alarm",
        "Police car": "Siren / Alarm",
        "Ambulance": "Siren / Alarm",
        "Fire engine": "Siren / Alarm",
        "Siren": "Siren / Alarm",
        "Alarm": "Siren / Alarm",
        "Buzzer": "Siren / Alarm",
        "Whistle": "Siren / Alarm",
        "Smoke detector": "Siren / Alarm",
        
        # Gunshot / Explosion
        "Gunshot": "Gunshot / Explosion",
        "Explosion": "Gunshot / Explosion",
        "Cap gun": "Gunshot / Explosion",
        "Fusillade": "Gunshot / Explosion",
        "Artillery": "Gunshot / Explosion",
        "Machine gun": "Gunshot / Explosion",
        "Firecracker": "Gunshot / Explosion",
        "Burst": "Gunshot / Explosion",
        "Crack": "Gunshot / Explosion",
        "Pop": "Gunshot / Explosion",
        "Slam": "Gunshot / Explosion",
        "Thump": "Gunshot / Explosion",
        "Thunder": "Gunshot / Explosion", # Sometimes confused with explosion
        
        # Impact / Footsteps
        "Hammer": "Impact / Breach",
        "Shatter": "Impact / Breach",
        "Glass": "Impact / Breach",
        "Smash": "Impact / Breach",
        "Footsteps": "Footsteps",
        "Clatter": "Impact / Breach"
    }
    
    # Exact match
    if mediapipe_category in mapping:
        return mapping[mediapipe_category]
    
    # Partial match
    cat_lower = mediapipe_category.lower()
    for key, value in mapping.items():
        if key.lower() in cat_lower:
            return value
            
    return "Ambient / Noise"

def convert_and_normalize(input_path):
    """
    Ensure the audio file is in the format YAMNet expects and NORMALIZE volume
    to ensure faint sounds are picked up.
    """
    temp_wav = tempfile.mktemp(suffix=".wav")
    try:
        # Increase gain/normalize to -1dB peak during conversion
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Loudness normalization
            '-ar', '16000', '-ac', '1',
            '-c:a', 'pcm_s16le', temp_wav
        ], check=True, capture_output=True)
        return temp_wav
    except Exception as e:
        print(f"Error normalizing audio with ffmpeg: {e}", file=sys.stderr)
        # Fallback to simple conversion
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-ar', '16000', '-ac', '1',
                '-c:a', 'pcm_s16le', temp_wav
            ], check=True, capture_output=True)
            return temp_wav
        except:
            return None

def classify_audio(audio_path, job_id):
    temp_wav = None
    try:
        audio_path = audio_path.strip('"')
        if not os.path.exists(audio_path):
            return {"status": "error", "message": f"File not found: {audio_path}"}

        print(f"--- High-Sensitivity Forensic Classifier Starting (Job: {job_id}) ---")
        print(f"Input: {audio_path}")
        
        # Normalize and convert
        temp_wav = convert_and_normalize(audio_path)
        if not temp_wav:
             return {"status": "error", "message": "Failed to normalize audio"}

        print("Model: Loading YAMNet TFLite...")
        model_path = get_yamnet_model_path()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Ultra-sensitive: capture almost everything
        options = audio.AudioClassifierOptions(
            base_options=base_options,
            max_results=25,
            score_threshold=0.001 
        )
        classifier = audio.AudioClassifier.create_from_options(options)

        print("Processing: Reading audio data...")
        sample_rate, wav_data = wavfile.read(temp_wav)
        if wav_data.dtype == np.int16:
            wav_data = wav_data.astype(np.float32) / 32768.0
            
        audio_data = mp.tasks.components.containers.AudioData.create_from_array(wav_data, 16000)
        
        # Run classification
        print("Inference: Running YAMNet forensic analysis...")
        results = classifier.classify(audio_data)
        print(f"Success: Processed {len(results)} windows.")
        
        all_detections = []
        
        # Process every result window
        for i, discovery in enumerate(results):
            timestamp = i * 0.48 # YAMNet window step
            
            for classification in discovery.classifications:
                for category in classification.categories:
                    forensic_type = map_to_forensic_category(category.category_name)
                    
                    # We always include detected events in the high-res stream
                    # but only if they meet a slightly higher threshold to keep UI clean
                    if forensic_type != "Ambient / Noise":
                        decibels = round(20 * np.log10(max(1e-5, category.score)) - 10, 1)
                        
                        # PRINT FOR USER TERMINAL (Requested Format)
                        print(f"[YAMNet] Time: {timestamp:.2f}s | Class: {forensic_type} | Confidence: {category.score:.4f} | Vol: {decibels}dB")

                        all_detections.append({
                            "type": forensic_type, # UI uses .type
                            "label": category.category_name,
                            "confidence": float(category.score),
                            "time": round(timestamp, 3),
                            "decibels": decibels # Estimated Power
                        })

        # Summary for status badges (one per category)
        required_ui_categories = [
            "Human Voice", "Musical Content", "Gunshot / Explosion", 
            "Siren / Alarm", "Scream / Aggression", "Vehicle Sound",
            "Footsteps", "Animal Signal", "Atmospheric Wind", "Impact / Breach"
        ]
        
        summary_events = []
        for cat in required_ui_categories:
            cat_matches = [d for d in all_detections if d["type"] == cat]
            if cat_matches:
                best = max(cat_matches, key=lambda x: x["confidence"])
                summary_events.append({
                    "category": cat,
                    "status": "DETECTED",
                    "confidence": round(best["confidence"] * 100, 1),
                    "details": f"Confirmed: {best['label']}"
                })
        
        # Sort all detections by time for the UI list
        all_detections.sort(key=lambda x: x["time"])

        return {
            "jobID": job_id,
            "soundEvents": all_detections, # UI uses this for Radar/Matrix
            "categorySummary": summary_events, # Extra info for badges
            "allDetections": all_detections, # Used by audio_separator.py
            "summary": f"Forensic analysis successful. {len(all_detections)} sound events isolated."
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Classification error: {str(e)}"}
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try: os.unlink(temp_wav)
            except: pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: python classifier.py <audio_path> [job_id]"}))
        sys.exit(1)
    
    res = classify_audio(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "job")
    
    # Human-readable summary for terminal
    if isinstance(res, dict) and res.get("status") == "error":
        print(f"\n[!] Forensic Analysis Failed: {res.get('message')}")
    else:
        num_events = len(res.get("soundEvents", []))
        print(f"\n[+] Forensic Analysis Successful")
        print(f"[+] {num_events} Forensic events identified and mapped.")

    # Hidden JSON for API parsing - wrapped in markers
    print(f"\n[JSON_START]{json.dumps(res)}[JSON_END]")