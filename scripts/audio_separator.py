import sys
import os
import json
import subprocess
import shutil
import warnings
import numpy as np
from scipy.io import wavfile

import tempfile

# FORCE SILENCE
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def convert_to_wav_if_needed(input_path, log_func):
    """
    Tries to read the file. If it fails, converts to WAV using FFmpeg.
    Returns (path_to_read, is_temp)
    """
    try:
        # Check if readable
        try:
            wavfile.read(input_path)
            return input_path, False
        except Exception:
            log_func(f"Direct read failed, attempting conversion for {input_path}")
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            tmp.close()
            output_path = tmp.name
            
            cmd = [
                "ffmpeg", "-y", 
                "-i", input_path, 
                "-ar", "44100", 
                 output_path
            ]
            # Suppress output
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            log_func(f"Converted to {output_path}")
            return output_path, True
    except Exception as e:
        log_func(f"Conversion failed: {str(e)}")
        return input_path, False

def separate_audio(input_path, output_dir, job_id, classification_path=None):
    debug_log = []
    
    def log(msg):
        debug_log.append(str(msg))

    converted_audio_path = None
    is_temp_file = False

    try:
        log(f"Start separation. Input: {input_path}, Job: {job_id}")
        input_path = os.path.abspath(input_path.strip('"'))
        output_dir = os.path.abspath(output_dir.strip('"'))
        
        # 0. Ensure Input is Valid WAV
        # Demucs might handle MP3, but since we had ID3 issues, let's normalize first.
        read_path, is_temp = convert_to_wav_if_needed(input_path, log)
        converted_audio_path = read_path
        is_temp_file = is_temp
        
        if classification_path:
            classification_path = os.path.abspath(classification_path.strip('"'))
        
        input_filename = os.path.basename(converted_audio_path)
        input_no_ext = os.path.splitext(input_filename)[0]
        # Demucs output folder is based on the input filename. 
        # If we converted to a temp file 'tmp123.wav', Demucs will output to 'htdemucs/tmp123'.
        # We need to map this back or rename.
        
        # actually, to preserve the job_id or original name context, we might want to check
        # but let's see what Demucs does.

        # 1. Run Demucs (In-process to bypass torchaudio.save issues)
        print(f"[Demucs] Loading model htdemucs...", file=sys.stderr)
        
        # Imports inside function to avoid heavy load if not needed
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio.transforms as T
        
        # Load Model
        model = get_model("htdemucs")
        model.cpu()
        model.eval()
        
        # Load Audio via Scipy (safe)
        sr, audio_data = wavfile.read(read_path)
        
        # Convert to float32 and normalize to [-1, 1]
        if audio_data.dtype == np.int16:
             audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
             audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
             audio_data = (audio_data.astype(np.float32) - 128) / 128.0
             
        # Ensure shape (Channels, Samples)
        if len(audio_data.shape) == 1:
            audio_data = np.expand_dims(audio_data, axis=0) # (1, Samples)
        else:
            audio_data = audio_data.T # (Channels, Samples)
            
        # Demucs expects Stereo (2 channels). Mix/Duplicate if Mono.
        if audio_data.shape[0] == 1:
            audio_data = np.concatenate([audio_data, audio_data], axis=0)
            
        # Convert to Tensor
        wav = torch.tensor(audio_data)
        
        # Resample if needed (Demucs htdemucs is 44100Hz)
        if sr != model.samplerate:
            print(f"[Demucs] Resampling {sr} -> {model.samplerate}Hz", file=sys.stderr)
            resampler = T.Resample(sr, model.samplerate)
            wav = resampler(wav)
            
        # Normalization (Standard Demucs procedure)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        # Separate
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"[Demucs] Separating using {num_workers} workers...", file=sys.stderr)
        
        # sources shape: (Sources, Channels, Samples)
        sources = apply_model(model, wav[None], device="cpu", shifts=1, split=True, 
                             overlap=0.25, progress=True, num_workers=num_workers)[0]
        
        # De-normalize
        sources = sources * ref.std() + ref.mean()
        
        print("[Demucs] Separation finished. Saving stems...", file=sys.stderr)
        
        # Save Stems manually using scipy
        stem_names = model.sources # ['drums', 'bass', 'other', 'vocals'] for htdemucs
        
        # Create output structure matching standard Demucs
        # htdemucs/filename_no_ext/
        demucs_folder_name = os.path.splitext(os.path.basename(read_path))[0]
        separated_folder = os.path.join(output_dir, "htdemucs", demucs_folder_name)
        os.makedirs(separated_folder, exist_ok=True)
        
        sources_np = sources.numpy()
        
        for i, name in enumerate(stem_names):
            stem_audio = sources_np[i] # (Channels, Samples)
            stem_audio = stem_audio.T # (Samples, Channels)
            
            out_file = os.path.join(separated_folder, f"{name}.wav")
            wavfile.write(out_file, model.samplerate, stem_audio)
            
        log(f"Demucs output saved to: {separated_folder}")

        # Populate final_stems
        final_stems = {}
        if os.path.exists(os.path.join(separated_folder, "vocals.wav")):
            final_stems["vocals"] = f"/separated_audio/htdemucs/{demucs_folder_name}/vocals.wav"
        
        # Combine other, bass, and drums into a single "background" stem for the UI
        try:
            bg_parts = []
            for part in ["other.wav", "bass.wav", "drums.wav"]:
                p_path = os.path.join(separated_folder, part)
                if os.path.exists(p_path):
                    psr, pdata = wavfile.read(p_path)
                    bg_parts.append(pdata)
            
            if bg_parts:
                # Sum them up
                combined_bg = sum(bg_parts)
                bg_out = os.path.join(separated_folder, "background_mixed.wav")
                wavfile.write(bg_out, model.samplerate, combined_bg)
                final_stems["background"] = f"/separated_audio/htdemucs/{demucs_folder_name}/background_mixed.wav"
        except Exception as e:
            log(f"Error mixing background: {e}")
            # Fallback to just "other" if mixing fails
            if os.path.exists(os.path.join(separated_folder, "other.wav")):
                final_stems["background"] = f"/separated_audio/htdemucs/{demucs_folder_name}/other.wav"


        # 2. Forensic Event Masking (if classification provided)
        if classification_path and os.path.exists(classification_path):
            try:
                log("Starting forensic masking...")
                with open(classification_path, 'r') as f:
                    classification_data = json.load(f)
                
                log(f"Loaded classification data. Keys: {list(classification_data.keys())}")
                if "status" in classification_data and classification_data["status"] == "error":
                     log(f"Classification ERROR: {classification_data.get('message', 'No message')}")

                # Load original Audio (already converted/validated)
                import librosa
                y, sr = librosa.load(read_path, sr=None)
                log(f"Loaded audio with librosa. SR: {sr}, Shape: {y.shape}")
                
                # 1. SPECIALIZED SPECTRAL PRE-PROCESSING
                # Separate Harmonic (Voice/Music) from Percussive (Gunshots/Footsteps/Impacts)
                log("Performing Harmonic-Percussive Source Separation (HPSS)...")
                # Power-based separation is better for forensic gating
                # Higher margin for more distinct separation
                harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 6.0))
                
                # Prepare empty containers (silence) for forensic stems
                stems_to_generate = {
                    "vocals": "Human Voice",
                    "background": "Musical Content",
                    "vehicles": "Vehicle Sound",
                    "footsteps": "Footsteps",
                    "animals": "Animal Signal",
                    "wind": "Atmospheric Wind",
                    "gunshots": "Gunshot / Explosion",
                    "screams": "Scream / Aggression",
                    "sirens": "Siren / Alarm",
                    "impact": "Impact / Breach"
                }

                # HIGHER SENSITIVITY: Consider any sound with > 0.02 confidence if it matches forensic type
                events = classification_data.get("soundEvents", [])
                log(f"Gating {len(events)} detected events...")
                
                # Initialize forensic stems with zeros (silence)
                generated_audio = { key: np.zeros_like(y) for key in stems_to_generate }
                
                CLIP_DURATION = 0.975 # standard YAMNet window
                count_generated = 0
                
                for event in events:
                    etype = event.get("type", "").lower()
                    conf = event.get("confidence", 0)
                    
                    if conf < 0.05: continue # Ignore very weak detections

                    target_stem = None
                    for stem_key, trigger_word in stems_to_generate.items():
                        if trigger_word.lower() in etype or etype in trigger_word.lower():
                            target_stem = stem_key
                            break
                    
                    if target_stem:
                        # Skip if Demucs handles it better
                        if target_stem == "vocals" and "vocals" in final_stems: continue
                        if target_stem == "background" and "background" in final_stems: continue

                        start_time = float(event.get("time", 0))
                        start_idx = int(start_time * sr)
                        end_idx = start_idx + int(CLIP_DURATION * sr)
                        start_idx = max(0, start_idx)
                        end_idx = min(len(y), end_idx)
                        
                        if start_idx < end_idx:
                            # Choose the most likely signal source based on forensic type
                            is_percussive = any(x in target_stem for x in ["gunshot", "impact", "footsteps"])
                            is_harmonic = any(x in target_stem for x in ["siren", "scream", "vocals"])
                            
                            source = y  # default
                            if is_percussive: source = percussive
                            elif is_harmonic: source = harmonic
                            
                            segment = source[start_idx:end_idx].copy()
                            
                            # APPLY FILTERS
                            if target_stem == "vehicles":
                                segment = librosa.lowpass_filter(segment, sr=sr, cutoff=300)
                            elif target_stem == "sirens":
                                segment = librosa.bandpass_filter(segment, sr=sr, low=500, high=3000)
                            
                            # Aggressive Noise Gate: anything 20dB below peak in segment is zeroed
                            peak = np.max(np.abs(segment))
                            if peak > 0:
                                threshold = peak * 0.15 # Stronger gate
                                segment[np.abs(segment) < threshold] = 0
                            
                            # Smooth fade
                            fade = int(0.05 * sr)
                            if len(segment) > fade * 2:
                                segment[:fade] *= np.linspace(0, 1, fade)
                                segment[-fade:] *= np.linspace(1, 0, fade)

                            generated_audio[target_stem][start_idx:end_idx] += segment
                            count_generated += 1

                log(f"Reconstruction complete ({count_generated} segments).")

                # Save generated stems
                gen_dir = os.path.join(output_dir, "generated", job_id)
                os.makedirs(gen_dir, exist_ok=True)
                
                for stem_key, audio_arr in generated_audio.items():
                    peak = np.max(np.abs(audio_arr))
                    if peak > 0.005: # more strict threshold
                        # Normalize and save
                        audio_arr = audio_arr * (0.8 / peak) # normalize to 0.8
                        out_file = os.path.join(gen_dir, f"{stem_key}.wav")
                        wavfile.write(out_file, sr, audio_arr)
                        final_stems[stem_key] = f"/separated_audio/generated/{job_id}/{stem_key}.wav"
                    elif stem_key not in final_stems:
                        final_stems[stem_key] = "__EMPTY__"
            
            except Exception as e:
                log(f"Masking Exception: {str(e)}")

        if not final_stems:
             log("No stems were generated.")
             return {"status": "error", "message": "Separation failed, no stems found.", "debug": debug_log}

        return {"status": "success", "stems": final_stems, "debug": debug_log}
    except Exception as e:
        return {"status": "error", "message": str(e), "debug": debug_log}
    finally:
        if is_temp_file and converted_audio_path and os.path.exists(converted_audio_path):
            try:
                os.unlink(converted_audio_path)
            except:
                pass

if __name__ == "__main__":
    # Ensure no other prints exist in this file!
    if len(sys.argv) > 3:
        # Check for optional 4th arg
        cls_path = sys.argv[4] if len(sys.argv) > 4 else None
        result = separate_audio(sys.argv[1], sys.argv[2], sys.argv[3], cls_path)
        sys.stdout.write(json.dumps(result))
    else:
        sys.stdout.write(json.dumps({"status": "error", "message": "Insufficient arguments"}))
    sys.stdout.flush()