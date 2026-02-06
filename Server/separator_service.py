import subprocess
import os

def separate_audio_tracks(input_file_path, output_root):
    """
    Uses Meta's Demucs to separate audio into 4 stems: 
    Vocals, Drums, Bass, and Other (Environment).
    """
    try:
        # Command for Demucs
        # -n htdemucs: uses the high-quality model
        # -o: specifies the output directory
        command = [
            "demucs",
            "-n", "htdemucs",
            "-o", output_root,
            input_file_path
        ]
        
        print(f"--- STARTING AI SEPARATION: {input_file_path} ---")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Demucs Error: {result.stderr}")
            return False
            
        print("--- SEPARATION SUCCESSFUL ---")
        return True
    except Exception as e:
        print(f"Service Error: {str(e)}")
        return False