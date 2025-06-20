import sys
# sys.argv = ['script.py', '--config', 'config.yaml']

import argparse, yaml, detector_model, joblib, os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def createDetector(model, num_hands):
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=num_hands)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def main(
    img_path,
    config_path='config.yaml',
    model_path=None,
    model_threshold=None,
    input_dir=None,
    point2origin=None,
    point2y=None,
    point2xy=None
):
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
    
    if df is None:
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