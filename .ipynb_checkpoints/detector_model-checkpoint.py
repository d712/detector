import numpy as np
import pandas as pd
import sys, os, pdb, joblib
import mediapipe as mp

class DetectorModel:
    
    def __init__(self, mpdetector, model, model_threshold, point2origin, point2y, point2xy):
        self.detector = mpdetector
        self.model = model
        self.threshold = model_threshold
        self.point2origin = point2origin
        self.point2y = point2y
        self.point2xy = point2xy
                
    def detection2obs(self, detection_result):
        obs_list = []
        for i in range(len(detection_result.hand_landmarks)):
            hand = int(detection_result.handedness[i][0].category_name == 'Left')
            hand_score = detection_result.handedness[i][0].score
            hlm = detection_result.hand_landmarks
            hlm_list = [value for j in range(len(hlm[i])) for value in [hlm[i][j].presence, hlm[i][j].visibility, hlm[i][j].x, hlm[i][j].y, hlm[i][j].z]]
            obs = [hand, hand_score, *hlm_list]
            obs_list.append(obs)
        return(obs_list)
 
    def file2detection(self, data_location):
        image = mp.Image.create_from_file(data_location)
        detection_result = self.detector.detect(image)
        df = pd.DataFrame(self.detection2obs(detection_result))
        if df.shape[0] == 0:
            return None
        col_names = []
        for i in range(0, 21):
            col_names[i*5+2:i*5+2+5] = [f'presence_{i}',f'visibility_{i}',f'x_{i}',f'y_{i}',f'z_{i}']
        df.columns = ['hand','hand_score'] + col_names
        
        df_clean = self.makeOrigin(df, self.point2origin)
        df_clean = self.R_df_byPoint(df_clean, self.point2y)
        df_clean = self.R_df_byY(df_clean, self.point2xy)

        probs = self.model.predict_proba(df_clean)[:,1]
        df_clean['probs'] = probs
        df_clean['data_location'] = data_location
        df_clean['flag'] = (self.model.predict_proba(df_clean)[:,1]>=self.threshold).sum()>0 
        #print(df['flag'])
        return df_clean

    def folder2df(self, data_folder_location):
        obs_df = pd.DataFrame()
        for subdir, dirs, files in os.walk(data_folder_location):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]
            for file in files:
                df = self.file2detection(os.path.join(subdir,file))
                if df is None:
                    continue
                obs_df = pd.concat([obs_df, df], ignore_index=True)
        return obs_df

    def makeOrigin(self, df, point):   
        for coord in ['x', 'y', 'z']:
            cols = df.columns[df.columns.str.startswith(coord)]
            df[cols] = df[cols].apply(lambda col: col - df[f'{coord}_{point}'])
        return df

    def R_df_byPoint(self, df, point):
        p9 = df[df.columns[df.columns.str.endswith(f'_{point}')][2:5]].copy()
        p9['len'] = np.sqrt(p9[f'x_{point}']**2 + p9[f'y_{point}']**2 + p9[f'z_{point}']**2)
        v9 = p9[p9.columns[p9.columns.str.endswith(f'_{point}')]].div(p9['len'], axis=0)
        p9.drop('len', axis=1,inplace=True)
        axis = pd.DataFrame(np.cross(v9,(0,1,0)))
        theta = np.arccos(np.clip(np.dot(v9, (0,1,0)), -1.0, 1.0))
        R = []
        for i in range(len(axis)):
            if np.linalg.norm(axis.iloc[[i],:]) <1e-6:
                R.append(np.eye(3))
            else:
                u_axis = axis.iloc[[i],:]/np.linalg.norm(axis.iloc[[i],:])
                R.append(self.R_byPoint(u_axis, theta[i]))
        df = df.reset_index(drop=True)
        for j in range(df.shape[0]):
            for i in range(0, 21):
                df.loc[j,[f'x_{i}', f'y_{i}', f'z_{i}']] = R[j] @ df.loc[j,[f'x_{i}', f'y_{i}', f'z_{i}']]
        return df

    def R_byPoint(self, axis, theta):
        axis_np = axis.iloc[0].to_numpy().astype(float)
        K = np.array([
            [0, -axis_np[2], axis_np[1]],
            [axis_np[2], 0, -axis_np[0]],
            [-axis_np[1], axis_np[0], 0]
        ])
        I = np.eye(3)
        R = I + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
        return R

    def R_df_byY(self, df, point):    
        p5 = df[df.columns[df.columns.str.endswith(f'_{point}')][2:5]].copy()
        p5['theta'] = p5.apply(lambda x: np.arctan2(x[2],x[0]), axis=1)
        Ry = []
        for i in range(len(p5)):
            Ry.append(self.R_byY(p5['theta'][i]))
        p5.drop('theta', axis=1, inplace=True)
        df = df.reset_index(drop=True)
        for j in range(df.shape[0]):
            for i in range(0, 21):
                df.loc[j,[f'x_{i}', f'y_{i}', f'z_{i}']] = Ry[j] @ df.loc[j,[f'x_{i}', f'y_{i}', f'z_{i}']]
        return df

    def R_byY(self, theta):
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [np.sin(-theta), 0, np.cos(-theta)]
        ])
        return R

