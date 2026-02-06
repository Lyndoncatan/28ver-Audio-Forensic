import subprocess
import sys
import os

def check_system():
    print("\n=== FORENSIC SONAR: SYSTEM DIAGNOSTIC ===")
    
    # 1. Check Python
    print(f"1. Python Version: {sys.version.split()[0]}")

    # 2. Check Demucs
    try:
        import demucs
        print("2. ✅ Demucs: INSTALLED")
    except ImportError:
        print("2. ❌ Demucs: NOT FOUND")
        print("   FIX: Run 'pip install demucs' in your terminal.")

    # 3. Check FFmpeg
    try:
        ffmpeg_check = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if ffmpeg_check.returncode == 0:
            print("3. ✅ FFmpeg: INSTALLED")
        else:
            print("3. ❌ FFmpeg: ERROR DETECTED")
    except FileNotFoundError:
        print("3. ❌ FFmpeg: NOT FOUND")
        print("   FIX: You must install FFmpeg and add it to your System PATH.")

    # 4. Check Folder Permissions
    public_path = os.path.join(os.getcwd(), "public", "separated_audio")
    if os.path.exists(public_path):
        print(f"4. ✅ Output Folder: EXISTS")
    else:
        print(f"4. ❌ Output Folder: MISSING ({public_path})")

    print("========================================\n")

if __name__ == "__main__":
    check_system()