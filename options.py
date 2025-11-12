"""
Changeable variables for main.py, easily controllable from this file instead of having to navigate the main script and editing them there.
"""

import os

# maximum iterations of the biomechanics processing stage, 10000 in the original
max_iters_biomechanics = 500
# path to the root folder where the main Python file is
path_root = 'C:/Users/jerela/Documents/GitHub/MonocularBiomechanics'
# path to the folder where intermediary and output data is stored
path_data = os.path.join(path_root,'data')
# path where the output of keypoint detection with METRABS is stored
path_keypoints = os.path.join(path_data,'keypoints')
# path where the output of biomechanical fitting is stored
path_biomechanics = os.path.join(path_data,'biomechanics')