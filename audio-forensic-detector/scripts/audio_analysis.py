# ================================
# Audio Forensic Analysis Script
# ================================

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import json
import sys
import base64
import io
from scipy.io import wavfile
import tempfile
import os

# Set matplotlib to use Agg backend for server environments
matplotlib.use('Agg')

def analyze_audio(audio_data_base64, filename="uploaded_audio"):
    """
    Analyze audio data and return comprehensive forensic analysis results
    """
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data_base64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        # Load audio with librosa
        y, sr = librosa.load(temp_path, sr=None)
        
        print(f"✅ Audio loaded: {filename}")
        print(f"Sample Rate: {sr} Hz")
        print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
        
        # ================================
        # STFT - Short-Time Fourier Transform
        # ================================
        stft_result = librosa.stft(y)
        stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
        
        # ================================
        # FFT - Fast Fourier Transform
        # ================================
        fft_result = np.fft.fft(y)
        magnitude = np.abs(fft_result)
        frequency = np.linspace(0, sr, len(magnitude))
        
        # ================================
        # Detect Sound Events
        # ================================
        frame_length = 1024
        hop_length = 512
        energy = np.array([
            sum(abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        
        # Normalize energy
        energy = energy / np.max(energy) if np.max(energy) > 0 else energy
        
        # Find peaks (sound events)
        peaks, properties = find_peaks(energy, height=0.2, distance=5)
        num_sounds = len(peaks)
        
        # ================================
        # Advanced Analysis
        # ================================
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        dominant_frequency = np.mean(spectral_centroids)
        
        # Convert to decibels
        max_decibels = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else -np.inf
        
        # Detect different types of sounds based on frequency characteristics
        sound_events = []
        for i, peak in enumerate(peaks):
            time_pos = (peak * hop_length) / sr
            freq_at_peak = spectral_centroids[min(peak, len(spectral_centroids)-1)]
            amplitude = energy[peak]
            
            # Classify sound type based on frequency
            if freq_at_peak < 300:
                sound_type = "Low Frequency/Bass"
            elif freq_at_peak < 1000:
                sound_type = "Voice/Mid Range"
            elif freq_at_peak < 4000:
                sound_type = "High Voice/Instruments"
            else:
                sound_type = "High Frequency/Noise"
            
            sound_events.append({
                "time": round(time_pos, 2),
                "frequency": round(freq_at_peak, 1),
                "amplitude": round(amplitude, 3),
                "type": sound_type,
                "decibels": round(20 * np.log10(amplitude) if amplitude > 0 else -np.inf, 1)
            })
        
        # Sort by amplitude (loudest first)
        sound_events.sort(key=lambda x: x["amplitude"], reverse=True)
        
        # Create frequency spectrum data
        freq_spectrum = []
        freq_step = len(frequency) // 100  # Sample 100 points
        for i in range(0, len(frequency)//2, freq_step):
            if i < len(magnitude):
                freq_spectrum.append({
                    "frequency": round(frequency[i], 1),
                    "magnitude": round(magnitude[i] / np.max(magnitude), 3) if np.max(magnitude) > 0 else 0
                })
        
        # ================================
        # Generate Analysis Report
        # ================================
        analysis_results = {
            "filename": filename,
            "duration": round(duration, 2),
            "sampleRate": int(sr),
            "averageRMS": round(float(rms), 6),
            "detectedSounds": num_sounds,
            "dominantFrequency": round(float(dominant_frequency), 1),
            "maxDecibels": round(float(max_decibels), 1),
            "soundEvents": sound_events[:10],  # Top 10 events
            "frequencySpectrum": freq_spectrum,
            "analysisComplete": True,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        print("\n📊 Audio Analysis Report")
        print(f"📁 File Name: {filename}")
        print(f"⏱ Duration: {duration:.2f} seconds")
        print(f"🎙 Sample Rate: {sr} Hz")
        print(f"📈 Average RMS Energy: {rms:.6f}")
        print(f"🔊 Detected Sound Events: {num_sounds}")
        print(f"🎵 Dominant Frequency: {dominant_frequency:.1f} Hz")
        print(f"📢 Max Decibels: {max_decibels:.1f} dB")
        
        print("\n🎯 Top Sound Events:")
        for i, event in enumerate(sound_events[:5]):
            print(f"{i+1}. {event['type']} at {event['time']}s - {event['frequency']:.1f}Hz ({event['decibels']:.1f}dB)")
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return json.dumps(analysis_results, indent=2)
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "analysisComplete": False,
            "message": "Audio analysis failed"
        }
        print(f"❌ Analysis Error: {str(e)}")
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    # Example usage - in real implementation, this would receive base64 data
    print("🎵 Audio Forensic Analysis System Ready")
    print("Waiting for audio data...")
    
    # This script can be called with audio data as argument
    if len(sys.argv) > 1:
        audio_data = sys.argv[1]
        filename = sys.argv[2] if len(sys.argv) > 2 else "uploaded_audio"
        result = analyze_audio(audio_data, filename)
        print(result)
    else:
        print("No audio data provided. Script ready for integration.")
