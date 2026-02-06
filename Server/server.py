from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil

# Importing your custom logic
# from run_yamnet import run_yamnet 
# from live_audio_analysis import generate_live_analysis
from separator_service import separate_audio_tracks

app = FastAPI()

# IMPORTANT: Enable CORS so your Next.js frontend can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths Setup
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "server", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "server", "separated_results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the static directory so the frontend can stream the .wav files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.post("/api/separate-audio")
async def handle_separation(file: UploadFile = File(...)):
    try:
        # 1. Save original file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Run the AI Separator (Demucs)
        # This will create: server/separated_results/htdemucs/[filename]/vocals.wav, etc.
        success = separate_audio_tracks(file_path, OUTPUT_DIR)

        if success:
            # Demucs uses the filename (without extension) for the folder name
            folder_name = os.path.splitext(file.filename)[0]
            
            # This URL matches the app.mount and the Demucs folder structure
            base_url = f"http://localhost:8000/output/htdemucs/{folder_name}"
            
            return {
                "status": "success",
                "tracks": {
                    "vocals": f"{base_url}/vocals.wav",
                    "environment": f"{base_url}/other.wav", # Environment/Car noise
                    "drums": f"{base_url}/drums.wav",
                    "bass": f"{base_url}/bass.wav"
                }
            }
        return {"error": "Separation failed"}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)