from fastapi import FastAPI, UploadFile, File, Request
import joblib, subprocess, shutil
from fastapi.responses import HTMLResponse

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
        'status': result.returncode
    }
#  'error': result.stderr
@app.get("/upload", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <body>
            <h2>Upload an image to run detector</h2>
            <form action="/run/" enctype="multipart/form-data" method="post">
                <input name="img" type="file" accept="image/*">
                <input type="submit" value="Upload and Run">
            </form>
        </body>
    </html>
    """
