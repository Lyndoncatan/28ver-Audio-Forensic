import librosa
import numpy as np
from scipy.signal import find_peaks
import json
import tempfile
import os
import torch
import torchaudio
from datetime import datetime
import subprocess
import shutil

def generate_live_analysis(audio_bytes, filename="uploaded_audio", output_dir="separated_stems"):
    """
    Forensic Separation: Extracts speech from background noise (animals/shouting).
    Uses Demucs for high-fidelity source isolation.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Save temp file for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # 2. PERFORM SOURCE SEPARATION (Demucs)
        # Using htdemucs - the transformer model for cleaner separation
        print(f"🕵️ Forensic Separation Started: {filename}")
        command = f"demucs --two-stems vocals -n htdemucs --out {output_dir} {temp_path}"
        subprocess.run(command, shell=True, check=True)

        # Path Logic for Demucs output
        base_name = os.path.splitext(os.path.basename(temp_path))[0]
        demucs_folder = os.path.join(output_dir, "htdemucs", base_name)
        vocal_path = os.path.join(demucs_folder, "vocals.wav")
        background_path = os.path.join(demucs_folder, "no_vocals.wav")

        # 3. COMPREHENSIVE AUDIO ANALYSIS
        y, sr = librosa.load(temp_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Signal Characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        energy = librosa.feature.rms(y=y)[0]
        peaks, _ = find_peaks(energy, height=np.mean(energy)*1.5, distance=10)

        sound_events = []
        for peak in peaks:
            idx = min(peak, len(spectral_centroids)-1)
            centroid = spectral_centroids[idx]
            zcr_val = zcr[idx]
            time_pos = (peak * 512) / sr

            # Forensic Logic: Classification based on Spectral Signature
            if 300 < centroid < 2800 and zcr_val < 0.12:
                label, spk_id = "HUMAN SPEECH", "SPEAKER_00"
            elif 800 < centroid < 2200 and 0.1 < zcr_val < 0.22:
                label, spk_id = "CAT MEOW", "SPEAKER_01"
            elif centroid > 2800 and zcr_val > 0.22:
                label, spk_id = "DOG/SHOUT", "SPEAKER_01"
            else:
                label, spk_id = "AMBIENT", "SPEAKER_01"

            sound_events.append({
                "time": round(time_pos, 2),
                "type": label,
                "speaker": spk_id,
                "frequency": round(float(centroid), 1),
                "decibels": round(float(20 * np.log10(max(energy[peak], 1e-6))), 1)
            })

        # 4. PREPARE FINAL JSON
        results = {
            "filename": filename,
            "duration": round(duration, 2),
            "soundEvents": sound_events[:25],
            "stems": {
                "primary_url": vocal_path,
                "background_url": background_path
            },
            "analysisComplete": True,
            "timestamp": datetime.now().isoformat()
        }

        os.unlink(temp_path) 
        return results

    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return {"error": str(e), "analysisComplete": False}