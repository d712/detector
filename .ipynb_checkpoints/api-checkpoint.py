from fastapi import FastAPI, UploadFile, File
import joblib, subprocess, shutil


app = FastAPI()
@app.get('/')
def home():
    return {"heyStranger": "sup"}

@app.post('/run/')
def run(img: UploadFile=File(...)):
    with open('uploadedimg.jpg','wb') as pic:
        shutil.copyfileobj(img.file, pic)    
    result = subprocess.run(["python3","run_detector.py", '--input', 'uploadedimg.jpg'], capture_output=True, text=True)
    
    return {
        'output': result.stdout,
        'status': result.returncode,
        'error': result.stderr
    }