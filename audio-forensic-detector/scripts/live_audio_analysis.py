# ================================
# Live Audio Analysis Script
# Enhanced version with real-time processing
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
import plotly.graph_objs as go
import plotly.offline as opy
from datetime import datetime

# Set matplotlib to use Agg backend for server environments
matplotlib.use('Agg')

def generate_live_analysis(audio_data_base64, filename="uploaded_audio"):
    """
    Generate comprehensive live audio analysis with multiple visualizations
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
        
        print(f"🎵 Live Analysis Started: {filename}")
        print(f"Sample Rate: {sr} Hz")
        print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
        
        # ================================
        # STFT - Short-Time Fourier Transform
        # ================================
        stft_result = librosa.stft(y, hop_length=512, n_fft=2048)
        stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
        
        # Time and frequency axes
        time_frames = librosa.frames_to_time(np.arange(stft_db.shape[1]), sr=sr, hop_length=512)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Create STFT heatmap data
        stft_data = {
            "z": stft_db.tolist(),
            "x": time_frames.tolist(),
            "y": freq_bins.tolist(),
            "type": "heatmap",
            "colorscale": "Viridis",
            "title": "STFT - Short-Time Fourier Transform"
        }
        
        # ================================
        # Live Spectrogram
        # ================================
        # Enhanced spectrogram with better resolution
        D = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
        spectrogram_data = {
            "z": D.tolist(),
            "x": time_frames.tolist(),
            "y": freq_bins.tolist(),
            "type": "heatmap",
            "colorscale": "Magma",
            "title": "Live Spectrogram"
        }
        
        # ================================
        # FFT - Fast Fourier Transform
        # ================================
        fft_result = np.fft.fft(y)
        magnitude = np.abs(fft_result)
        frequency = np.linspace(0, sr, len(magnitude))
        
        # Take only positive frequencies
        half_length = len(frequency) // 2
        fft_data = {
            "x": frequency[:half_length].tolist(),
            "y": magnitude[:half_length].tolist(),
            "type": "scatter",
            "mode": "lines",
            "title": "FFT - Frequency Spectrum"
        }
        
        # ================================
        # Sound Event Detection
        # ================================
        frame_length = 1024
        hop_length = 512
        
        # Compute energy
        energy = np.array([
            sum(abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        
        # Normalize energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        # Find peaks (sound events)
        peaks, properties = find_peaks(energy, height=0.2, distance=5)
        
        # Create energy detection data
        energy_data = {
            "energy": energy.tolist(),
            "peaks": peaks.tolist(),
            "peak_values": energy[peaks].tolist() if len(peaks) > 0 else [],
            "frames": list(range(len(energy)))
        }
        
        # ================================
        # Advanced Spectral Features
        # ================================
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # ================================
        # Sound Classification
        # ================================
        sound_events = []
        for i, peak in enumerate(peaks):
            time_pos = (peak * hop_length) / sr
            
            # Get spectral features at this time
            frame_idx = min(peak, len(spectral_centroids) - 1)
            centroid = spectral_centroids[frame_idx]
            rolloff = spectral_rolloff[frame_idx]
            zcr_val = zcr[frame_idx] if frame_idx < len(zcr) else 0
            
            # Classify based on spectral features
            if centroid < 1000 and rolloff < 2000:
                sound_type = "Low Frequency/Bass"
                confidence = 0.8
            elif centroid < 3000 and zcr_val < 0.1:
                sound_type = "Voice/Speech"
                confidence = 0.9
            elif centroid > 4000 and rolloff > 8000:
                sound_type = "High Frequency/Noise"
                confidence = 0.7
            elif zcr_val > 0.15:
                sound_type = "Percussive/Transient"
                confidence = 0.85
            else:
                sound_type = "Mixed/Complex"
                confidence = 0.6
            
            amplitude = energy[peak]
            decibels = 20 * np.log10(amplitude) if amplitude > 0 else -np.inf
            
            sound_events.append({
                "time": round(time_pos, 2),
                "frequency": round(float(centroid), 1),
                "amplitude": round(float(amplitude), 3),
                "type": sound_type,
                "confidence": confidence,
                "decibels": round(float(decibels), 1),
                "spectral_rolloff": round(float(rolloff), 1),
                "zero_crossing_rate": round(float(zcr_val), 3)
            })
        
        # Sort by amplitude (loudest first)
        sound_events.sort(key=lambda x: x["amplitude"], reverse=True)
        
        # ================================
        # Generate Comprehensive Report
        # ================================
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Dominant frequency
        dominant_freq = np.mean(spectral_centroids)
        
        # Max decibels
        max_amplitude = np.max(np.abs(y))
        max_decibels = 20 * np.log10(max_amplitude) if max_amplitude > 0 else -np.inf
        
        # Frequency spectrum for visualization
        freq_spectrum = []
        freq_step = len(frequency) // 200  # Sample 200 points
        for i in range(0, len(frequency)//2, max(1, freq_step)):
            if i < len(magnitude):
                freq_spectrum.append({
                    "frequency": round(frequency[i], 1),
                    "magnitude": round(magnitude[i] / np.max(magnitude), 3) if np.max(magnitude) > 0 else 0
                })
        
        # ================================
        # Live Analysis Results
        # ================================
        live_analysis_results = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "duration": round(duration, 2),
            "sampleRate": int(sr),
            "averageRMS": round(float(rms), 6),
            "detectedSounds": len(peaks),
            "dominantFrequency": round(float(dominant_freq), 1),
            "maxDecibels": round(float(max_decibels), 1),
            "soundEvents": sound_events[:15],  # Top 15 events
            "frequencySpectrum": freq_spectrum,
            
            # Visualization data
            "visualizations": {
                "stft": stft_data,
                "spectrogram": spectrogram_data,
                "fft": fft_data,
                "energy": energy_data
            },
            
            # Advanced features
            "spectralFeatures": {
                "meanSpectralCentroid": round(float(np.mean(spectral_centroids)), 1),
                "meanSpectralRolloff": round(float(np.mean(spectral_rolloff)), 1),
                "meanZeroCrossingRate": round(float(np.mean(zcr)), 3),
                "mfccMean": [round(float(np.mean(mfcc)), 3) for mfcc in mfccs]
            },
            
            "analysisComplete": True,
            "analysisType": "live_comprehensive"
        }
        
        print("\n🎯 Live Analysis Report Generated")
        print(f"📁 File: {filename}")
        print(f"⏱ Duration: {duration:.2f} seconds")
        print(f"🎙 Sample Rate: {sr} Hz")
        print(f"📈 Average RMS: {rms:.6f}")
        print(f"🔊 Detected Events: {len(peaks)}")
        print(f"🎵 Dominant Frequency: {dominant_freq:.1f} Hz")
        print(f"📢 Max Decibels: {max_decibels:.1f} dB")
        
        print("\n🎯 Top Sound Events:")
        for i, event in enumerate(sound_events[:5]):
            print(f"{i+1}. {event['type']} at {event['time']}s - {event['frequency']:.1f}Hz ({event['decibels']:.1f}dB) - Confidence: {event['confidence']:.1%}")
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return json.dumps(live_analysis_results, indent=2)
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "analysisComplete": False,
            "analysisType": "live_comprehensive",
            "message": "Live audio analysis failed",
            "timestamp": datetime.now().isoformat()
        }
        print(f"❌ Live Analysis Error: {str(e)}")
        return json.dumps(error_result, indent=2)

def process_real_time_chunk(audio_chunk, sr=44100):
    """
    Process a real-time audio chunk for live visualization
    """
    try:
        # Quick FFT analysis
        fft_result = np.fft.fft(audio_chunk)
        magnitude = np.abs(fft_result)
        
        # Energy calculation
        energy = np.sum(audio_chunk ** 2)
        
        # Spectral centroid
        freqs = np.fft.fftfreq(len(audio_chunk), 1/sr)
        spectral_centroid = np.sum(freqs[:len(freqs)//2] * magnitude[:len(magnitude)//2]) / np.sum(magnitude[:len(magnitude)//2])
        
        return {
            "fft": magnitude[:len(magnitude)//2].tolist(),
            "energy": float(energy),
            "spectral_centroid": float(spectral_centroid),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🎵 Live Audio Analysis System Ready")
    print("Enhanced with real-time visualization capabilities")
    
    # This script can be called with audio data as argument
    if len(sys.argv) > 1:
        audio_data = sys.argv[1]
        filename = sys.argv[2] if len(sys.argv) > 2 else "live_audio"
        result = generate_live_analysis(audio_data, filename)
        print(result)
    else:
        print("No audio data provided. Script ready for live integration.")
