from monocular_demos.utils import jax_memory_limit, tensorflow_memory_limit
tensorflow_memory_limit()
jax_memory_limit()
import os
from typing import List

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tqdm import tqdm
import time
from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics
from monocular_demos.biomechanics_mjx.visualize import render_trajectory
from monocular_demos.biomechanics_mjx.monocular_trajectory import (
    fit_model,
    get_model,
)
from monocular_demos.utils import load_metrabs_from_path, joint_names, video_reader
from monocular_demos.dataset import MonocularDataset,get_samsung_calibration

# import options.py where some parameters are defined
import options

fk = ForwardKinematics(
    xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml",
)

jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)


def save_metrabs_data(accumulated, video_name):
    fname = os.path.normpath(os.path.join(options.path_keypoints, video_name.split('.')[0]))
    boxes, pose3d, pose2d, confs = [], [], [], []
    for i, (box, p3d, p2d) in enumerate(
        zip(accumulated["boxes"], accumulated["poses3d"], accumulated["poses2d"])
    ):
        # TODO: write logic for better box tracking
        if len(box) == 0:
            boxes.append(np.zeros((5)))
            pose3d.append(np.zeros((87, 3)))
            pose2d.append(np.zeros((87, 2)))
            confs.append(np.zeros((87)))
            print("no boxes")
            continue
        boxes.append(box[0].numpy())
        pose3d.append(p3d[0].numpy())
        pose2d.append(p2d[0].numpy())
        confs.append(np.ones((87)))

        with open(f"{fname}_keypoints.npz", "wb") as f:
            np.savez(f, keypoints3d=pose3d, keypoints2d=pose2d, boxes=boxes, confs=confs)
            print(f'Saved METRABS data to {fname}_keypoints.npz')

def render_mjx(selected_file):
    """Load saved data and create visualizations"""
    if not selected_file or selected_file == "No fitted models found":
        return "Please select a fitted model file first.", None, None
    
    fname = selected_file.split('.')[0]
    
    result_text = ""
    video_filename = f"{fname}_mjx.mp4"
    
    # include paths so the files are correctly found on Windows
    biomech_file_with_path = os.path.normpath(os.path.join(options.path_biomechanics,f'{fname}_fitted_model.npz'))
    video_with_path = os.path.normpath(os.path.join(options.path_output_video,video_filename))
    
    print(f'Biomech file with path: {biomech_file_with_path}')
    if os.path.exists(biomech_file_with_path):
        with open(biomech_file_with_path, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded biomechanics data: {biomech_file_with_path}\n"
            qpos = data['qpos']
    
    render_trajectory(
        qpos,
        filename = video_with_path,
        xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque_vis.xml",
        height=options.video_height,
        width=options.video_width,
        video_codec=options.video_codec,
    )
    result_text += f"Rendered visualization: {video_with_path}\n"

    return result_text, video_with_path



def get_framerate(video_path):
    """
    Get the framerate of a video file.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def load_metrabs_data(video_name):
    fname = os.path.normpath(os.path.join(options.path_keypoints, video_name.split('.')[0]))
    try:
        with open(f"{fname}_keypoints.npz", "rb") as f:
            data = np.load(f, allow_pickle=True)
            boxes = data["boxes"]
            keypoints2d = data["keypoints2d"]
            keypoints3d = data["keypoints3d"]
            confs = data["confs"]

        return boxes, keypoints2d, keypoints3d, confs
    except FileNotFoundError:
        print("No saved data found for this video.")
        return None, None, None, None


def process_videos_with_metrabs(video_files: List[str]) -> str:
    """
    Process the uploaded videos. Replace this with your actual processing logic.
    """
    if not video_files:
        return "No videos uploaded."

    print('Loading MeTRAbs model...')
    t1 = time.time()
    model = load_metrabs_from_path(os.path.normpath(options.path_metrabs_model))
    t2 = time.time()
    print(f'MeTRAbs model loaded in {t2-t1} seconds.')
    skeleton = "bml_movi_87"
    
    video_count = 0
    for video_idx, video_path in enumerate(video_files):
        video_path_full = os.path.normpath(os.path.join(options.path_input_video,video_path))
        print(f'Video {video_idx} at path {video_path_full}')
        if video_path is not None:
            vid, n_frames = video_reader(video_path_full)
            accumulated = None
            for frame_idx, frame_batch in enumerate(tqdm(vid, desc=f'Processing video {video_idx}')):
                if frame_idx%options.frame_step == 0:
                    
                    pred = model.detect_poses_batched(frame_batch, skeleton=skeleton)

                    if accumulated is None:
                        accumulated = pred

                    else:
                        # concatenate the ragged tensor along the batch for each element in the dictionary
                        for key in accumulated.keys():
                            accumulated[key] = tf.concat(
                                [accumulated[key], pred[key]], axis=0
                            )
                else:
                    continue
            video_filename = os.path.split(video_path_full)[1]
            video_save_path = os.path.join(options.path_keypoints, video_filename)
            save_metrabs_data(accumulated, video_save_path)
            video_count += 1

    return f"Successfully processed {video_count} videos with Metrabs."


def process_videos_with_biomechanics(video_files: List[str]) -> str:
    """
    Process the uploaded videos with biomechanics fitting. Replace this with your actual processing logic.
    """
    import equinox as eqx
    import jax
    jax.clear_caches()
    eqx.clear_caches()

    max_iters = options.max_iters_biomechanics

    if not video_files:
        return "No videos uploaded."

    timestamps_list = []
    keypoints2d_list = []
    keypoints3d_list = []
    confs_list = []
    for i, video_path in enumerate(video_files):
        if video_path is not None:
            boxes, keypoints2d, keypoints3d, confs = load_metrabs_data(video_path)
            if boxes is None:
                print(f"Video {video_path}: No Metrabs data found.")
                continue

            video_path_full = os.path.normpath(os.path.join(options.path_input_video, video_path))
            fps = get_framerate(video_path_full)
            timestamps = np.arange(0, len(keypoints2d)) / fps
            timestamps_list.append(timestamps)
            keypoints2d_list.append(keypoints2d[jnp.newaxis])  # Add camera dimension
            keypoints3d_list.append(keypoints3d[jnp.newaxis])  # Add camera dimension
            confs_list.append(
                jnp.ones_like(keypoints2d[..., 0])[jnp.newaxis]
            )  # fake confidences

    dataset = MonocularDataset(
        timestamps=timestamps_list,
        keypoints_2d=keypoints2d_list,
        keypoints_3d=keypoints3d_list,
        keypoint_confidence=confs_list,
        camera_params=get_samsung_calibration(),
        phone_attitude=None,
    )

    model = get_model(
        dataset, xml_path="monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml", joint_names=joint_names
    )  # might need to change the site names
    model, metrics = fit_model(
        model,
        dataset,
        lr_init_value=1e-3,
        max_iters=max_iters,
    )
    
    for i, video_path in enumerate(video_files):

        timestamps = dataset.get_all_timestamps(i)

        (state, _, _), (qpos, qvel, _), rnc = model(
            timestamps,
            trajectory_selection=i,
            steps=0,
            skip_action=True,
            fast_inference=True,
            check_constraints=False,
        )

        # save zip archive
        fname = os.path.normpath(os.path.join(options.path_biomechanics, video_path.split(".")[0]))
        print(f'fname for saving biomechanics: {fname}')
        with open(f"{fname}_fitted_model.npz", "wb") as f:
            np.savez(
                f,
                timestamps=np.array(timestamps),
                qpos=np.array(qpos),
                qvel=np.array(qvel),
                rnc=np.array(rnc),
                sites=np.array(state.site_xpos),
                joints=np.array(state.xpos),
                scale=np.array(model.body_scale)
            )
            print(f'Fitted biomechanical model saved with filename: {fname}_fitted_model.npz')

    return f"Successfully processed {len(dataset)} videos with biomechanics fitting."


def get_available_fitted_models():
    """Get list of available fitted model files"""
    fitted_files = [f for f in os.listdir(os.path.normpath(options.path_biomechanics)) if f.endswith('_fitted_model.npz')]
    return fitted_files if fitted_files else ["No fitted models found"]

def load_and_visualize_data(selected_file, selected_joints=None):
    """Load saved data and create visualizations"""
    if not selected_file or selected_file == "No fitted models found":
        return "Please select a fitted model file first.", None, None
    
    selected_joint_inds = [fk.joint_names.index(joint) for joint in selected_joints] if selected_joints else []
    
    fname = selected_file.replace('_fitted_model.npz', '')
    
    # Try to load keypoints data
    keypoints_file = f"{fname}_keypoints.npz"
    biomech_file = selected_file
    keypoints_file = os.path.normpath(os.path.join(options.path_keypoints,keypoints_file))
    biomech_file = os.path.normpath(os.path.join(options.path_biomechanics,biomech_file))
    
    result_text = ""
    plot1 = None
    
    if os.path.exists(keypoints_file):
        with open(keypoints_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded keypoints data: {keypoints_file}\n"
            result_text += f"- Frames: {len(data['keypoints3d'])}\n"
            result_text += f"- 3D keypoints shape: {data['keypoints3d'].shape}\n"
            result_text += f"- 2D keypoints shape: {data['keypoints2d'].shape}\n\n"
    else:
        result_text += f"No keypoints data found for {fname}\n\n"
    
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            result_text += f"Loaded biomechanics data: {biomech_file}\n"
            result_text += f"- Timesteps: {len(data['qpos'])}\n"
            result_text += f"- Joint positions shape: {data['qpos'].shape}\n"
            result_text += f"- Joint velocities shape: {data['qvel'].shape}\n"

            # Create Plot 1: Joint angles over time
            qpos = data['qpos']
            time_steps = np.arange(len(qpos))
            
            fig1 = go.Figure()
            # Plot selected joints
            if selected_joints:
                for joint_idx in selected_joint_inds:
                    if joint_idx < qpos.shape[1]:
                        fig1.add_trace(go.Scatter(
                            x=time_steps, 
                            y=qpos[:, joint_idx], 
                            mode='lines',
                            name=f'Joint {fk.joint_names[joint_idx]}'
                        ))
            else:
                # Default to first 6 joints if none selected
                for i in range(min(6, qpos.shape[1])):
                    fig1.add_trace(go.Scatter(
                        x=time_steps, 
                        y=qpos[:, i], 
                        mode='lines',
                        name=f'Joint {i+1}'
                    ))
            fig1.update_layout(
                title="Joint Angles Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Angle (radians)"
            )
            plot1 = fig1
    else:
        result_text += f"No biomechanics data found for {fname}\n"
    
    return result_text, plot1

def get_joint_options(selected_file):
    """Get available joint options for the selected model"""
    if not selected_file or selected_file == "No fitted models found":
        return gr.Dropdown(choices=[], value=[])
    
    biomech_file = selected_file
    
    if os.path.exists(biomech_file):
        with open(biomech_file, "rb") as f:
            data = np.load(f, allow_pickle=True)
            num_joints = data['qpos'].shape[1]
            joint_choices = [(f"Joint {i+1}", i) for i in range(num_joints)]
            default_selection = list(range(min(6, num_joints)))  # Default to first 6
            
            return gr.Dropdown(
                choices=joint_choices,
                value=default_selection,
                multiselect=True
            )
    
    return gr.Dropdown(choices=[], value=[])



def main():

    # find .mp4 files in input directory
    input_files = [f for f in os.listdir(os.path.normpath(options.path_input_video)) if f.endswith('.mp4')]
    print(f'Found input files: {input_files}')
    
    print(f'Beginning MeTRAbs processing of {len(input_files)} video(s).')
    process_videos_with_metrabs(input_files)
    print(f'MeTRAbs processing finished. Beginning biomechanical fitting.')
    process_videos_with_biomechanics(input_files)
    print(f'Biomechanical fitting finished. Beginning output video rendering.')
    for f in input_files:
        render_mjx(f)
    print('All done!')

if __name__ == '__main__':
    main()
