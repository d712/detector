from fastapi import FastAPI, UploadFile, File, Request
import joblib, subprocess, shutil
from fastapi.responses import HTMLResponse

app = FastAPI()
@app.get('/')
def home() -> dict:
    """Defines home endpoint and checks connection with the api."""
    return {"heyStranger": "sup"}

@app.post('/run/')
def run(img: UploadFile=File(...)) -> dict:
    """Return model outputs (detection results, connection status, and model messages) for uploaded .jpg picture."""
    with open('uploadedimg.jpg','wb') as pic:
        shutil.copyfileobj(img.file, pic)    
    result = subprocess.run(["python3","run_detector.py", '--input', 'uploadedimg.jpg'], capture_output=True, text=True)
    
    return {
        'output': result.stdout,
        'status': result.returncode,
        'msg': result.stderr
    }
 
@app.get("/upload", response_class=HTMLResponse)
async def upload_form() -> str:
    """Creates the web interface for API users to upload a .jpg picture."""
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
