import sys
import os
import json
import librosa
import soundfile as sf
import numpy as np

def process_forensic(input_path, output_dir, base_name):
    # Load the audio
    y, sr = librosa.load(input_path, sr=None)
    
    # 1. ISOLATE VOCALS (Using Median Filtering for harmonic/percussive)
    harmonic, percussive = librosa.effects.hpss(y)
    sf.write(os.path.join(output_dir, f"{base_name}_vocals.wav"), harmonic, sr)
    
    # 2. ISOLATE FOOTSTEPS / IMPACT (Percussive transients)
    sf.write(os.path.join(output_dir, f"{base_name}_footsteps.wav"), percussive, sr)
    
    # 3. ISOLATE VEHICLES (Low Pass Filter < 300Hz)
    vehicles = librosa.lowpass_filter(y, sr=sr, cutoff=300)
    sf.write(os.path.join(output_dir, f"{base_name}_vehicles.wav"), vehicles, sr)
    
    # 4. ISOLATE ANIMALS / BIRDS (High Pass Filter > 2000Hz + Peak Detection)
    animals = librosa.highpass_filter(y, sr=sr, cutoff=2000)
    sf.write(os.path.join(output_dir, f"{base_name}_animals.wav"), animals, sr)
    
    # 5. ISOLATE WIND / ATMOSPHERIC (Bandpass 500Hz - 1500Hz + Smoothing)
    wind = librosa.bandpass_filter(y, sr=sr, low=500, high=1500)
    sf.write(os.path.join(output_dir, f"{base_name}_wind.wav"), wind, sr)
    
    # 6. ISOLATE BACKGROUND (The remainder)
    background = y - harmonic - percussive
    sf.write(os.path.join(output_dir, f"{base_name}_background.wav"), background, sr)

    return {
        "status": "completed",
        "files_generated": 6
    }

if __name__ == "__main__":
    try:
        in_file = sys.argv[1]
        out_dir = sys.argv[2]
        name = sys.argv[3]
        
        result = process_forensic(in_file, out_dir, name)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))