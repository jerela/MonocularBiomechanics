"""
Changeable variables for main.py, easily controllable from this file instead of having to navigate the main script and editing them there.
"""

import os
import numpy as np

# default 800 on both
video_height = 800
video_width = 800

# default "mp4v", but doesn't seem to work on Gradio Video with Windows 11 and Firefox
video_codec = "avc1"

# how many frames to skip when detecting keypoints with MeTRAbs; keep at 1 if you want all frames to be analyzed
frame_step = 10

# maximum iterations of the biomechanics processing stage, 10000 in the original
max_iters_biomechanics = 500
# path to the root folder where the main Python file is
path_root = 'C:/Users/jerela/Documents/GitHub/MonocularBiomechanics'
# path to the folder where intermediary and output data is stored
path_data = os.path.join(path_root,'data')
# path where the metrabs model is
path_metrabs_model = os.path.join(path_data,'models')
# path where the input videos are
path_input_video = os.path.join(path_data,'input')
# path where the output of keypoint detection with METRABS is stored
path_keypoints = os.path.join(path_data,'keypoints')
# path where the output of biomechanical fitting is stored
path_biomechanics = os.path.join(path_data,'biomechanics')
# path where the video output is stored
path_output_video = os.path.join(path_data,'video')
# WIP; path to the calibration file for the camera
path_calibration = None


def get_biocv_calibration(file_names):

    ret, C, S, D, K, R, T = calib_biocv_fun(file_names)

    #mtx = np.array(
    #    [[1.43333476e03, 0.00000000e00, 5.39500000e02], [0.00000000e00, 1.43333476e03, 9.59500000e02], [0.00000000e00, 0.00000000e00, 1.00000000e00]]
    #)
    mtx = K[0]
    mtx = (mtx[[0, 1, 0, 1], [0, 1, 2, 2]] / 1000).reshape(1, -1)
    #dist = np.zeros(5).reshape(1, -1)
    dist = np.array(D[0]).reshape(1,-1)
    # rotation and translation are not necessary when we are using a single camera instead of triangulating keypoints from several cameras
    rvec = np.zeros(3).reshape(1, -1)
    tvec = np.zeros(3).reshape(1, -1)
    return dict(mtx=mtx, dist=dist, tvec=tvec, rvec=rvec)

# adapted from Pose2Sim: https://github.com/perfanalytics/pose2sim/blob/main/Pose2Sim/calibration.py#L387C1-L420C33
def calib_biocv_fun(calibration_file_paths):
    '''
    Convert bioCV calibration files.

    INPUTS:
    - files_to_convert_paths: paths of the calibration files to convert (no extension)

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path) as f:
            calib_data = f.read().split('\n')
            ret += [np.nan]
            C += [f'cam_{str(i).zfill(2)}']
            S += [[int(calib_data[0]), int(calib_data[1])]]
            D += [[float(d) for d in calib_data[-2].split(' ')[:5]]]
            K += [np.array([k.strip().split(' ') for k in calib_data[2:5]], np.float32)]
            RT = np.array([k.strip().split(' ') for k in calib_data[6:9]], np.float32)
            R += [cv2.Rodrigues(RT[:,:3])[0].squeeze()]
            T += [RT[:,3]/1000]
                        
    return ret, C, S, D, K, R, T

