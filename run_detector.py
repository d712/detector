import sys
# sys.argv = ['script.py', '--config', 'config.yaml']

import argparse, yaml, detector_model, joblib, os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pandas as pd
import numpy as np


def load_config(path: str) -> dict:
    """Load the configs in the yaml file from path."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def crt_detector(model: str, num_hands: int) -> vision.HandLandmarker:
    """
    Create a Mediapipe detector from chosen Mediapipe model, and max number of hands to be detected.

    Parameters
    ----------
    model: str. Google Mediapipe model for detecting hands in static pictures.
    num_hands: int. Configure Google Mediapipe model for maximum amount of hands to detect in a picture.

    Return
    ------
    A detector model for detecting hands in a static picture.

    """
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=num_hands)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def summarizeFinding(df: pd.DataFrame) -> str:
    """
    Given prediction results for 1 image, create a message that summarizes findings.
    
    Parameter
    ---------
    df: pd.DataFrame. A dataframe that contains file names and prediction results from the gesture detector for all the hands MediaPipe detects in one picture.

    Return
    ------
    A string that summarizes how many hands are flagged for positive cases of the gesture out of all the hands MediaPipe detects in one picture.
    """
    filename = df.iloc[0]['filename'].split('/')[-1]
    msg = f"{df['flag_hand'].sum()} out of {df.shape[0]} hand(s) flagged in {filename}."   
    return msg      
        
def main(
    config_path: str = 'config.yaml',
    model_path_overwrite: str | None = None,
    model_threshold_overwrite: float | None = None,
    input_dir_overwrite: str | None = None,
    output_dir_overwrite: str | None = None,
    point2origin_overwrite: int | None = None,
    point2y_overwrite: int | None = None,
    point2xy_overwrite: int | None = None
) -> tuple[pd.DataFrame | None, list[str]]:   
    """Main function. Detect existance of hands in a picture and flags if middle finger gestures are detected."""   
    config = load_config(config_path)
    
    config_overwrite = {
        'model_path': model_path_overwrite,
        'model_threshold': model_threshold_overwrite,
        'input_dir': input_dir_overwrite,
        'output_dir': output_dir_overwrite,
        'point2origin': point2origin_overwrite,
        'point2y': point2y_overwrite,
        'point2xy': point2xy_overwrite
    }

    for key, value in config_overwrite.items():
        if value is not None:
            config[key] = value
    # create an instance of the hand detection model class
    detector = crt_detector(config['mediapipe_task'], config['num_hands'])
    # create an instance of the gesture detection model class
    model = joblib.load(config['model_path'])
    dm = detector_model.DetectorModel(detector, model, config['model_threshold'], config['point2origin'], config['point2y'], config['point2xy'], config['num_hand_landmarks'])
    # generate predictions based on the gesture detection model
    df = dm.folder2df(config['input_dir'])    
    msgs = []
    if df is None or df.shape[0] == 0:
        msg = 'Mediapipe does not detect any hands in this picture.'
        return None, [msg]
    else:
        df.to_csv(os.path.join(config['output_dir'],'predictions.csv'))
        msgs = [summarizeFinding(group) for _, group in df.groupby('filename')]
        return df, msgs
        
if __name__ == '__main__':
    # for when the script is executed in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    main(args.input)