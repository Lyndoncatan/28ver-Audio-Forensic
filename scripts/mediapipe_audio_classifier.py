import numpy as np
import json
import sys
import base64
import os
import warnings
import tempfile
import subprocess
from scipy.io import wavfile

# Silence all background noise from TensorFlow/MediaPipe
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_yamnet_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'yamnet.tflite')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YAMNet model missing at {model_path}")
    return model_path

def map_to_forensic_category(mediapipe_category):
    mapping = {
        "Speech": "Human Voice",
        "Singing": "Human Voice",
        "Male speech": "Male Voice",
        "Female speech": "Female Voice",
        "Music": "Musical Content",
        "Vehicle": "Vehicle Sound",
        "Car": "Vehicle Sound",
        "Bus": "Vehicle Sound",
        "Truck": "Vehicle Sound",
        "Motorcycle": "Vehicle Sound",
        "Footsteps": "Footsteps",
        "Animal": "Animal Signal",
        "Dog": "Animal Signal",
        "Cat": "Animal Signal",
        "Bird": "Animal Signal",
        "Wind": "Atmospheric Wind",
        "Thunder": "Atmospheric Wind",
        "Breeze": "Atmospheric Wind",
        "Silence": "Silence",
        # NEW FORENSIC CATEGORIES
        "Gunshot": "Gunshot / Explosion",
        "Explosion": "Gunshot / Explosion",
        "Cap gun": "Gunshot / Explosion",
        "Fusillade": "Gunshot / Explosion",
        "Artillery": "Gunshot / Explosion",
        "Screaming": "Scream / Aggression",
        "Shout": "Scream / Aggression",
        "Yell": "Scream / Aggression",
        "Siren": "Siren / Alarm",
        "Alarm": "Siren / Alarm",
        "Buzzer": "Siren / Alarm",
        "Glass": "Impact / Breach",
        "Shatter": "Impact / Breach",
        "Smash": "Impact / Breach",
        "Hammer": "Impact / Breach",
        "Door": "Impact / Breach",
        "Knock": "Impact / Breach",
        "Slam": "Impact / Breach"
    }
    for key, value in mapping.items():
        if key.lower() in mediapipe_category.lower():
            return value
    return mediapipe_category

def convert_to_wav(input_path):
    """
    Converts input audio to standard WAV (16kHz, mono) using FFmpeg.
    Returns path to the converted temp file.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.close()
        output_path = tmp.name
        
        # Determine if ffmpeg is available
        # On some systems it might be 'ffmpeg', on others full path. assuming 'ffmpeg' is in PATH.
        cmd = [
            "ffmpeg", "-y", 
            "-i", input_path, 
            "-ar", "16000", 
            "-ac", "1", 
            output_path
        ]
        
        # Suppress output
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except Exception as e:
        # If conversion fails, return original (and hope for best)
        return input_path

def classify_audio(audio_path, job_id):
    converted_path = None
    try:
        # Handle quoted paths if passed
        audio_path = audio_path.strip('"')
        
        if not os.path.exists(audio_path):
            return {"status": "error", "message": f"File not found: {audio_path}"}

        # Attempt to read. If it fails (like ID3 header), try converting.
        try:
            sample_rate, wav_data = wavfile.read(audio_path)
        except Exception:
            # Conversion fallback
            converted_path = convert_to_wav(audio_path)
            try:
                sample_rate, wav_data = wavfile.read(converted_path)
            except Exception as e:
                return {"status": "error", "message": f"Error reading wav file: {str(e)}"}
        
        # Normalize and convert to mono
        if wav_data.dtype == np.int16:
            wav_data = wav_data.astype(float) / 32768.0
        if len(wav_data.shape) > 1:
            wav_data = np.mean(wav_data, axis=1)

        # Import MediaPipe inside function to keep startup silent
        from mediapipe.tasks import python
        from mediapipe.tasks.python.components import containers
        from mediapipe.tasks.python import audio

        options = audio.AudioClassifierOptions(
            base_options=python.BaseOptions(model_asset_path=get_yamnet_model_path()),
            max_results=5,
            score_threshold=0.05
        )
        
        with audio.AudioClassifier.create_from_options(options) as classifier:
            audio_clip = containers.AudioData.create_from_array(wav_data.astype(np.float32), sample_rate)
            results = classifier.classify(audio_clip)
            
            print("--- Running Model: YAMNet / MediaPipe ---", file=sys.stderr)
            
            events = []
            for idx, res in enumerate(results):
                if res.classifications:
                    top = res.classifications[0].categories[0]
                    forensic_cat = map_to_forensic_category(top.category_name)
                    confidence = round(top.score, 4)
                    decibels = round(-60 + (top.score * 60), 1)
                    time_sec = round(idx * 0.975, 2)
                    
                    print(f"[YAMNet] Time: {time_sec}s | Class: {forensic_cat} | Confidence: {confidence} | Vol: {decibels}dB", file=sys.stderr)
                    
                    events.append({
                        "time": time_sec,
                        "type": forensic_cat,
                        "confidence": confidence,
                        "decibels": decibels
                    })
            
            print("--- Classification Complete ---", file=sys.stderr)

            # clean up temp converted file
            if converted_path and converted_path != audio_path and os.path.exists(converted_path):
                os.unlink(converted_path)

            return {
                "status": "success",
                "jobID": job_id,
                "detectedSounds": len(events),
                "soundEvents": events
            }
            
    except Exception as e:
        if converted_path and os.path.exists(converted_path):
            os.unlink(converted_path)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # We use sys.stdout.write to ensure no extra newlines are added
        output = classify_audio(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "job")
        sys.stdout.write(json.dumps(output))
    else:
        sys.stdout.write(json.dumps({"status": "error", "message": "No input"}))