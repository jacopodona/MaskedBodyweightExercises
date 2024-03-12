import time
import numpy as np
import matplotlib.pyplot as plt
from optitrack.geometry import *


#######################################----- INITIALIZATION ------#######################################
#
# Select choosen joint from the 21 body parts
#
def get_joints(array, joints, legend):
    tmp = []
    legend_f = []
    for i in range(0, len(joints)):
        if joints[i] == True:
            tmp.append(array[i])
            legend_f.append(legend[i])
    return tmp, legend_f

def get_bone_pos(bodies, take):
#
# Get the marker positions from the bodies and the time stamp
#
    bones_pos = []
    #b = [0, 13, 14, 15, 16, 17, 18, 19, 20]
    b = list(range(21))
    if len(bodies) > 0:
        for body in bodies: 
            bones = take.rigid_bodies[body]
            bones_pos.append(bones.positions)   # take position of each body part
    bones_pos = [bones_pos[i] for i in b]
    return bones_pos


def get_joint_pos(bodies, take):
#
# Get the marker positions from the bodies and the time stamp
#
    bones_pos = []
    if len(bodies) > 0:
        for body in bodies:
            bones = take.rigid_bodies[body]
            bones_pos.append(bones.positions)   # take position of each body part
    return bones_pos



#######################################----- 3D SKELETON -----#######################################

def plot_3d_joints(joints, ax, frame):
#
#   Plot 3D point plot is parametric with frame istance
#
    # plot the hip
    #if(joints[0][frame]!= None):
    #    ax.scatter(joints[0][frame][2], joints[0][frame][0], joints[0][frame][1])

    # plot the lower body
    #points_of_interest=[0,13,14,15,16,17,18,19,20]  # joints id of hip and lower body

    #Plot the whole body
    points_of_interest = list(range(21))
    for i in points_of_interest:    # Plot the desired joints of interes
        if(joints[i][frame] != None):
            ax.scatter(joints[i][frame][2], joints[i][frame][0], joints[i][frame][1])
            ax.text(joints[i][frame][2], joints[i][frame][0], joints[i][frame][1],str(i))


def plot_3d_line(ax, first, second, color):
#
# Plots a 3d line between two 3d points
#
    x=[first[2], second[2]]
    y=[first[0], second[0]]
    z=[first[1], second[1]]
    ax.plot(x, y, z, color)


def plot_3d_skeleton(joints, ax, frame, color):
    #
    # Plot all points
    #
        plot_3d_joints(joints, ax, frame)
        # Plot edge connections
        #body_edges = [[0, 1], [0, 13], [13, 14], [14, 15],[0, 16], [16, 17], [17, 18], [18, 20], [15, 19]]
        body_edges = [
            [0, 1],  # Hip to Ab
            [1, 2],  # Ab to Chest
            [2, 3],  # Chest to Neck
            [3, 4],  # Neck to Head
            [2, 5],  # Chest to LShoulder
            [5, 6],  # LShoulder to LUArm
            [6, 7],  # LUArm to LFArm
            [7, 8],  # LFArm to LHand
            [2, 9],  # Chest to RShoulder
            [9, 10],  # RShoulder to RUArm
            [10, 11],  # RUArm to RFArm
            [11, 12],  # RFArm to RHand
            [0, 13],  # Hip to LThigh
            [13, 14],  # LThigh to LShin
            [14, 15],  # LShin to LFoot
            [0, 16],  # Hip to RThigh
            [16, 17],  # RThigh to RShin
            [17, 18],  # RShin to RFoot
            [15, 19],  # LFoot to LToe
            [18, 20],  # RFoot to RToe
        ]
        for joint1,joint2 in body_edges:
            if (joints[joint1][frame]!=None and joints[joint2][frame]!=None):
                plot_3d_line(ax,joints[joint1][frame],joints[joint2][frame],color)
      