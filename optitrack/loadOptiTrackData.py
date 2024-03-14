import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import plot_utils
import optitrack.plot as opt_plot
# Load the Optitrack CSV file parser module.
import optitrack.csv_reader as opt_csv
import optitrack.geometry as opt_geometry
from scipy.signal import savgol_filter


def convertPositionsToFrames(positions):
    """
    Convert frame in [num_joint,num_frames,num_coordinates] to [num_frames,num_joint,num_coordinates]
    :param positions:
    :return:
    """
    frames=[]
    num_frames=len(positions[0])
    for frame_idx in range(num_frames):
        body = []
        hasNone=False
        for joint in positions:
            body.append(joint[frame_idx])
        for joint in body:
            if joint is None:
                hasNone=True
        if len(body)==21 and hasNone==False: #Only save complete skeletons, discard skeletons with missing joints
            frames.append(body)
    return frames

def loadData(take,smooth=False,show=False):

    #legend = ['Hip','Left thigh','Left shin','Left foot','Right thigh','Right shin','Right foot','Left toe','Right toe']
    #legend = ['Hip', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand', 'LThigh', 'LShin', 'LFoot', 'RThigh', 'RShin', 'RFoot', 'LToe', 'RToe']
    #run
    bodies = take.rigid_bodies
    positions = opt_plot.get_joint_pos(bodies, take)#contiene 21 liste (per ogni joint) che contiene per ogni frame le posizioni x,y,z

    if smooth:
        positions=apply_temporal_smoothing(positions,window_size=60)

    frames=convertPositionsToFrames(positions)
    if show:
        plot_utils.plot_take(frames,skip=60)
    return frames

def plot_takes_comparison(original,enhanced):
    figure = plt.figure(figsize=(16, 8))
    #ax = plt.axes(projection='3d')

    ax1 = figure.add_subplot(121, projection='3d')  # 1 row, 2 columns, first subplot
    ax2 = figure.add_subplot(122, projection='3d')

    take_duration = len(original[0])
    skip=10

    # plt.legend(legend)
    ax1.set_xlabel('X')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylabel('Y')
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_zlabel('Z')
    ax1.set_zlim(0, 1.2)
    ax1.set_title("Original")

    ax2.set_xlabel('X')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylabel('Y')
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_zlabel('Z')
    ax2.set_zlim(0, 1.2)
    ax2.set_title("Enhanced")

    plt.ion()

    frame=0
    for i in range(frame, take_duration, skip):
        plot_utils.plot_3d_skeleton_OPT(original, ax1, 'black')
        plot_utils.plot_3d_skeleton_OPT(enhanced, ax2, 'black')
        figure.canvas.draw()
        figure.canvas.flush_events()
        ax1.clear()
        ax1.set_xlabel('X')
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylabel('Y')
        ax1.set_ylim(0, 4)
        ax1.set_zlabel('Z')
        ax1.set_zlim(0, 2)
        ax1.set_title(f"Original frame {i}/{take_duration}")
        ax2.clear()
        ax2.set_xlabel('X')
        ax2.set_xlim(-2.5, 2.5)
        ax2.set_ylabel('Y')
        ax2.set_ylim(0, 4)
        ax2.set_zlabel('Z')
        ax2.set_zlim(0, 2)
        ax2.set_title(f"Enhanced frame {i}/{take_duration}")
        plt.show()

def apply_temporal_smoothing(skeleton, window_size=11):
    #num_joints, num_frames = skeleton.shape

    num_joints = len(skeleton)  # Number of joints
    num_frames = len(skeleton[0])  # Number of frames
    num_dimensions = len(skeleton[0][0]) #Cordinates cardinality

    smoothed_skeleton = [[[0.0] * num_dimensions for _ in range(num_frames)] for _ in range(num_joints)]

    # Iterate through all joints
    for joint_id in range(num_joints):
        for frame_id in range(num_frames):
            # Extract the window of frames centered at the current frame
            window_start = max(0, frame_id - window_size // 2)
            window_end = min(num_frames, frame_id + window_size // 2 + 1)

            # Calculate the moving average for each dimension (x, y, z)
            for dim in range(num_dimensions):
                smoothed_skeleton[joint_id][frame_id][dim] = sum(
                    skeleton[joint_id][i][dim] for i in range(window_start, window_end)) / (window_end - window_start)

    return smoothed_skeleton

def main():
    dir = os.path.dirname(__file__)
    path= os.path.join(dir, '../data/csv/baseline_skeleton')
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(dir, path,file)
            take = opt_csv.Take().readCSV(file_path)
            frames=loadData(take,show=False)
            pickle_out_name=file.replace(".csv",".pkl")
            out_file_path=path.replace("csv","pkl/optitrack")
            if not os.path.exists(out_file_path):
                os.mkdir(out_file_path)
            out_file_name=os.path.join(out_file_path,pickle_out_name)
            with open(out_file_name, 'wb') as pkl_file:
                pickle.dump(frames, pkl_file)

main()