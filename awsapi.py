from fastapi import FastAPI, UploadFile, File
from datetime import datetime
import os, shutil, uuid, runDetector

awsapi = FastAPI()

@awsapi.get('/')
def homefunc() -> dict:
    """Check if the api is connected."""
    return {"msg":"sup"}

@awsapi.post('/run/')
async def check_pics(pics: list[UploadFile] = File(...)) -> dict:
    """
    Runs the detection model on each photo uploaded.
    
    Parameters
    -----------
    pics `list[fastapi.UploadFile]`: A list of all the uploaded files to be checked.

    Returns
    -----------
    A dict in the format `{"Status": "complete", "Findings": [result_str_1, result_str_2, etc]}
    """
    # First we'll create a folder for the uploaded files based on the start time and a UUID to ensure we don't get any collisions in folder name
    # NOTE: we convert the uuid to hex to avoid characters we can't inclue in a filename
    inputfoldername = "uploaded" + datetime.now().strftime("%y%m%d%H%M%S") + uuid.uuid4().hex
    # The output folder will be the same, appended with "_results"
    outputfoldername = inputfoldername + "_results"
    os.makedirs(inputfoldername, exist_ok=True)
    os.makedirs(outputfoldername, exist_ok=True)
    # For each photo
    for pic in pics:
        # Create a filepath to the photo
        filepath = os.path.join(inputfoldername, pic.filename)
        # Write the file
        with open(filepath, 'wb') as file:
            shutil.copyfileobj(pic.file, file)
    
    # Once we have a folder of photos, we can run the model on them
    _, msgs = runDetector.main(
        input_dir=inputfoldername,
        output_dir=outputfoldername
    )
    # And return the result
    return {
        "status": 'complete', # TODO: fix capitalizations of this dict.
        "findings": msgs
    }
    