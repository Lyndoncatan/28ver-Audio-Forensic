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
            
            cmd = ['ffmpeg', '-y', '-i', input_path, '-ar', '44100', '-ac', '2', output_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            log_func(f"Converted to {output_path}")
            return output_path, True
    except Exception as e:
        log_func(f"Conversion failed: {str(e)}")
        return input_path, False

def separate_audio(input_path, output_dir, job_id, classification_path=None):
    debug_log = []
    
    def log(msg):
        msg_str = str(msg)
        debug_log.append(msg_str)
        # Print to stderr to pass through the API's error logger (keeps terminal clean)
        # but with a professional prefix
        print(f"[Demucs] {msg_str}", file=sys.stderr)

    converted_audio_path = None
    is_temp_file = False

    try:
        log(f"Start separation. Input: {input_path}, Job: {job_id}")
        input_path = os.path.abspath(input_path.strip('"'))
        output_dir = os.path.abspath(output_dir.strip('"'))
        
        # 0. Ensure Input is Valid WAV
        read_path, is_temp = convert_to_wav_if_needed(input_path, log)
        converted_audio_path = read_path
        is_temp_file = is_temp
        
        if classification_path:
            classification_path = os.path.abspath(classification_path.strip('"'))
        
        # 1. Run Demucs
        log("Loading model htdemucs...")
        
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio.transforms as T
        
        model = get_model("htdemucs")
        model.cpu()
        model.eval()
        
        sr, audio_data = wavfile.read(read_path)
        
        if audio_data.dtype == np.int16:
             audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
             audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
             audio_data = (audio_data.astype(np.float32) - 128) / 128.0
             
        if len(audio_data.shape) == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        else:
            audio_data = audio_data.T
            
        if audio_data.shape[0] == 1:
            audio_data = np.concatenate([audio_data, audio_data], axis=0)
            
        wav = torch.tensor(audio_data)
        
        if sr != model.samplerate:
            log(f"Resampling {sr} -> {model.samplerate}Hz")
            resampler = T.Resample(sr, model.samplerate)
            wav = resampler(wav)
            
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        log(f"Separating using {num_workers} workers...")
        
        sources = apply_model(model, wav[None], device="cpu", shifts=1, split=True, 
                             overlap=0.25, progress=True, num_workers=num_workers)[0]
        
        sources = sources * ref.std() + ref.mean()
        
        log("Separation finished. Saving stems...")
        
        stem_names = model.sources
        demucs_folder_name = os.path.splitext(os.path.basename(read_path))[0]
        separated_folder = os.path.join(output_dir, "htdemucs", demucs_folder_name)
        os.makedirs(separated_folder, exist_ok=True)
        
        sources_np = sources.numpy()
        final_stems = {}
        for i, name in enumerate(stem_names):
            out_path = os.path.join(separated_folder, f"{name}.wav")
            # Stereo save
            data_to_save = sources_np[i].T
            wavfile.write(out_path, model.samplerate, (data_to_save * 32767).astype(np.int16))
            final_stems[name] = f"/separated_audio/htdemucs/{demucs_folder_name}/{name}.wav"

        # 2. Harmonic/Percussive Masking (Forensic Polish)
        log("Starting forensic masking...")
        try:
            import librosa
            merged_background = (sources_np[0] + sources_np[1] + sources_np[2]) / 3.0
            y_back = merged_background[0]
            
            # Harmonic/Percussive Split for Forensic Clarity
            h, p = librosa.effects.hpss(y_back, margin=(1.0, 5.0))
            
            back_path = os.path.join(separated_folder, "background_mixed.wav")
            wavfile.write(back_path, model.samplerate, (p * 32767).astype(np.int16))
            final_stems["background"] = f"/separated_audio/htdemucs/{demucs_folder_name}/background_mixed.wav"
            log("Background masking complete.")
        except Exception as e:
            log(f"Masking Exception: {str(e)}")

        # 3. Gated Stem Extraction (Advanced Forensic)
        if classification_path and os.path.exists(classification_path):
            try:
                with open(classification_path, 'r') as f:
                    cls_data = json.load(f)
                
                import re
                job_id_clean = re.sub(r'[^a-zA-Z0-9]', '_', job_id).lower()
                gen_dir = os.path.join(output_dir, "generated", job_id_clean)
                os.makedirs(gen_dir, exist_ok=True)
                
                log(f"Loaded classification data. Keys: {list(cls_data.keys())}")
                
                y_full, sr_full = librosa.load(read_path, sr=None)
                log(f"Loaded audio with librosa. SR: {sr_full}, Shape: {y_full.shape}")
                
                # Separation Logic based on detected forensic events
                forensic_targets = {
                    "gunshots": ["Gunshot / Explosion"],
                    "screams": ["Scream / Aggression"],
                    "sirens": ["Siren / Alarm"],
                    "impact": ["Impact / Breach"],
                    "footsteps": ["Footsteps"],
                    "animals": ["Animal Signal"],
                    "wind": ["Atmospheric Wind"]
                }
                
                all_events = cls_data.get("allDetections", [])
                log(f"Gating {len(all_events)} detected forensic events...")
                
                for key, types in forensic_targets.items():
                    out_y = np.zeros_like(y_full)
                    matches = [e for e in all_events if e["type"] in types]
                    
                    if matches:
                        for m in matches:
                            start_s = m["time"]
                            end_s = start_s + 0.5 # YAMNet window
                            start_idx = int(start_s * sr_full)
                            end_idx = int(end_s * sr_full)
                            if end_idx < len(y_full):
                                out_y[start_idx:end_idx] = y_full[start_idx:end_idx]
                        
                        f_path = os.path.join(gen_dir, f"{key}.wav")
                        wavfile.write(f_path, sr_full, (out_y * 32767).astype(np.int16))
                        final_stems[key] = f"/separated_audio/generated/{job_id_clean}/{key}.wav"
                    else:
                        final_stems[key] = "__EMPTY__"
                
                log("Reconstruction complete.")
            except Exception as e:
                log(f"Forensic Gating Exception: {str(e)}")

        if not final_stems:
             log("No stems were generated.")
             return {"status": "error", "message": "Separation failed, no stems found.", "debug": debug_log}

        return {"status": "success", "stems": final_stems, "debug": debug_log}
    except Exception as e:
        return {"status": "error", "message": str(e), "debug": debug_log}
    finally:
        if is_temp_file and converted_audio_path and os.path.exists(converted_audio_path):
            try: os.unlink(converted_audio_path)
            except: pass

if __name__ == "__main__":
    if len(sys.argv) > 3:
        cls_path = sys.argv[4] if len(sys.argv) > 4 else None
        res = separate_audio(sys.argv[1], sys.argv[2], sys.argv[3], cls_path)
        # Clean successful prints
        if res.get("status") == "success":
            print("\n[+] Separation Successful")
            print(f"[+] {len(res.get('stems', {}))} Stems generated and forensic-gated.")
        else:
            print(f"\n[!] Separation Failed: {res.get('message')}")
            
        sys.stdout.write(f"\n[JSON_START]{json.dumps(res)}[JSON_END]\n")
    else:
        err = {"status": "error", "message": "Insufficient arguments"}
        sys.stdout.write(f"\n[JSON_START]{json.dumps(err)}[JSON_END]\n")
    sys.stdout.flush()