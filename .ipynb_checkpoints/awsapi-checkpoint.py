from fastapi import FastAPI, UploadFile, File
from datetime import datetime
import os, shutil, uuid, runDetector

awsapi = FastAPI()

@awsapi.get('/')
def homefunc() -> dict:
    """Check if the api is connected."""
    return {"msg":"sup"}

@awsapi.post('/run/')
async def checkPics(pics: list[UploadFile] = File(...)) -> dict:
    """Check uploaded picture(s)."""
    inputfoldername = "uploaded" + datetime.now().strftime("%y%m%d%H%M%S") + uuid.uuid4().hex
    outputfoldername = inputfoldername + "results"
    os.makedirs(inputfoldername, exist_ok=True)
    os.makedirs(outputfoldername, exist_ok=True)
    for p in pics:
        filepath = os.path.join(inputfoldername, p.filename)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(p.file, f)
    df, msgs = runDetector.main(
        input_dir=inputfoldername,
        output_dir=outputfoldername
    )
    return {
        "Status": 'complete',
        "Findings": msgs
    }
    