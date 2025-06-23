import sys
# sys.argv = ['script.py', '--config', 'config.yaml']

import argparse, yaml, detector_model, joblib, os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_config(path: str) -> dict:
    """Load the configs in the yaml file from path."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def createDetector(model: str, num_hands: int) -> 'vision.HandLandmarker':
    """Create a Mediapipe detector from chosen Mediapipe model, and max number of hands to be detected."""
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=num_hands)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def main(
    img_path: str,
    config_path: str = 'config.yaml',
    model_path: str | None = None,
    model_threshold: float | None = None,
    input_dir: str | None = None,
    point2origin: int | None = None,
    point2y: int | None = None,
    point2xy: int | None = None
) -> None:   
    """Main function. Detects existance of hands in a picture and flags if middle finger gestures are detected."""   
    config = load_config(config_path)
    
    if model_path is not None:
        config['model_path'] = model_path
    if model_threshold is not None:
        config['model_threshold'] = model_threshold
    if input_dir is not None:
        config['input_dir'] = input_dir
    
    detector = createDetector(config['mediapipe_task'], config['num_hands'])
    model = joblib.load(config['model_path'])
    dm = detector_model.DetectorModel(detector, model, config['model_threshold'], config['point2origin'], config['point2y'], config['point2xy'])
    df = dm.file2detection(img_path)
    # df = dm.folder2df(config['input_dir'])
    # df.to_csv(os.path.join(config['output_dir'],'predictions.csv'))
    
    if df is None or df.shape[0] == 0:
        print('Mediapipe does not detect any hands in this picture.')
    elif df.flag.sum() > 0:
        print(f'{df.flag.sum()} out of {df.shape[0]} hand(s) flagged.')
    else:
        print(f'Clean. {df.shape[0]} hand(s) detected.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    main(args.input)