import matplotlib.pyplot as plt

import skeleton_utils


def plot_3d_joints(joints, ax):
#
#   Plot 3D point plot is parametric with frame istance
#
    for point_id in range(len(joints)):
        point=joints[point_id]
        x=point[0]
        y=point[1]
        z=point[2]
        #ax.scatter(point[2],point[0],point[1])
        #ax.scatter(z, x, y)
        ax.scatter(z, x, y,s=3,c="red")
        #ax.text(point[2], point[0], point[1],str(point_id) , color='red')

def plot_2d_joints(joints, ax, projection):
#
#   Plot 3D point plot is parametric with frame istance
#
    for point_id in range(len(joints)):
        point=joints[point_id]
        x = point[0]
        y = point[1]
        z = point[2]
        if projection=="XY":
            ax.scatter(x, y,s=15,c="red")
        if projection=="ZY":
            ax.scatter(z, y,s=20,c="red")

def plot_3d_line(ax, first, second, color):
#
# Plots a 3d line between two 3d points
#
    #x=[first[2], second[2]]
    #y=[first[0], second[0]]
    #z=[first[1], second[1]]
    x=[first[0], second[0]]
    y=[first[1], second[1]]
    z=[first[2], second[2]]
    ax.plot(z, x, y, color)

def plot_2d_line(ax, first, second, color,projection):
#
# Plots a 3d line between two 3d points
#
    #x=[first[2], second[2]]
    #y=[first[0], second[0]]
    #z=[first[1], second[1]]
    x=[first[0], second[0]]
    y=[first[1], second[1]]
    z=[first[2], second[2]]
    if projection=="XY":
        ax.plot(x,y, color)
    if projection=="ZY":
        ax.plot(z, y, color)

def plot_3d_skeleton_OPT(joints, ax, color):
    # Prepare axis visualization

    ax.set_xlabel('Z',labelpad=-10)
    ax.set_ylabel('X',labelpad=-10)
    ax.set_zlabel('Y',labelpad=-10)

    # Plot all points
    #
    plot_3d_joints(joints, ax)
    # Plot edge connections
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
    for joint1, joint2 in body_edges:
        if (joints[joint1] is not None and joints[joint2] is not None):
            plot_3d_line(ax, joints[joint1], joints[joint2], color)

def plot_2d_skeleton_OPT(joints, ax, color,projection):
    # Prepare axis visualization

    # Plot all points
    #
    plot_2d_joints(joints, ax,projection)
    # Plot edge connections
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
    for joint1, joint2 in body_edges:
        if (joints[joint1] is not None and joints[joint2] is not None):
            plot_2d_line(ax, joints[joint1], joints[joint2], color,projection)


def plot_3d_skeleton_ZED(joints, ax, color):
    # Prepare axis visualization

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    # Plot all points
    #
    plot_3d_joints(joints, ax)
    # Plot edge connections
    body_edges=[]
    if len(joints)==19:
        body_edges = [
            [0,1], #Head to neck
            [0,14], #Head to REye
            [0,15], #Head to LEye
            [1, 2], #Neck to RShoulder
            [1, 5], #Neck to LShoulder
            [1,18], #Neck to Pelvis
            [2, 3], #RShoulder to RElbow
            #[2, 8], #Rshoulder to RHip
            [3,4], #RElbow to RHand
            [5, 6], #LShoulder to LElbow
            #[5, 11],#Lshoulder to LHip
            [6, 7], #LElbow to LHand
            [8, 9], #RHip to RKnee
            [8,18],#RHip to Pelvis
            [9, 10], #RKnee to RAnkle
            [11, 12], #LHip to LKnee
            [11, 18],  # LHip to Pelvis
            [12, 13], #LKnee to LAnkle
            [14, 16], #REye to REar
            [15, 17], #LEye to LEar
        ]
    elif len(joints)==34:
        body_edges = [
            [0,1], #Pelvis to Ab
            [0,18], #Pelvis to LHip
            [0,22], #Pelvis to RHip
            [1,2], #Ab to Torso
            [2,3], #Torso to Neck
            [3,4], #Neck to LClavicle
            [3,11], #Neck to RClavicle
            [3,26], #Neck to Head(Mouth)
            [4,5], #LClavicle to LShoulder
            [5,6], #LShoulder to LElbow
            [6,7], #LElbow to LWrist
            [7,8], #LWrist to LHand
            [7,10],  #LWrist to LThumb
            [8,9], #LHand to LHandTip
            [11,12], #RClavicle to RShoulder
            [12,13], #RShoulder to RElbow
            [13,14], #RElbow to RWrist
            [14,15], #RWrist to RHand
            [14,17], #RWrist to RThumb
            [15,16], #RHand to RHandTip
            [18, 19],  #LHip to LKnee
            [19, 20],  #LKnee to LAnkle
            [20, 21],  #LAnkle to LFoot
            [20, 32],  # LFoot to LHeel
            [21, 32],  #LFoot to LHeel
            [22, 23],  #RHip to RKnee
            [23, 24],  #RKnee to RAnkle
            [24, 25],  #RAnkle to RFoot
            [24, 33],  #RAnkle to RHeel
            [25, 33],  #RFoot to RHeel
            [26, 27],  #Head(Mouth) to Nose
            [27, 28], #Nose to LEye
            [27, 30],  #Nose to REye
            [28, 29],  # LEye to LEar
            [30,31] #REye to REar
        ]
    for joint1, joint2 in body_edges:
        if (joints[joint1] is not None and joints[joint2] is not None):
            plot_3d_line(ax, joints[joint1], joints[joint2], color)


def plot_frame_debug(joints,center_on_hips=True,source="OPT"):
    ax = plt.axes(projection='3d')
    ax.azim = 60
    resetAxis(ax,center_on_hips=center_on_hips)
    if center_on_hips:
        if source == "MPOSE":
            joints = skeleton_utils.centerSkeletonAroundHip(joints, hip_id=33)
        else:
            joints = skeleton_utils.centerSkeletonAroundHip(joints)
    if source=="OPT":
        plot_3d_skeleton_OPT(joints, ax, 'black')
    elif source=="ZED":
        plot_3d_skeleton_ZED(joints, ax, 'black')
    elif source == "MPOSE":
        plot_3d_skeleton_MPOSE(joints, ax, 'black')
    plt.show()

def resetAxis(ax, title=None,center_on_hips=False):
    ax.clear()
    if center_on_hips:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    else:
        ax.set_xlim(-1, 2)
        ax.set_ylim(1, 3)
        ax.set_zlim(0, 2)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Skeleton Take Animation")


def plot_3d_skeleton_MPOSE(joints, ax, color):
    # Prepare axis visualization

    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    # Plot all points
    #
    plot_3d_joints(joints, ax)
    # Plot edge connections
    body_edges = [
        [0, 1], #Head
        [1,2],
        [2,3],
        [3,7],
        [4,5],
        [5,6],
        [6,8],
        [9,10],
        [11,13], #Left Arm
        [13,15],
        [15,21],
        [15,19],
        [15,17],
        [19,17],
        [12,14], #Right Arm
        [14,16],
        [16,22],
        [16,18],
        [16,20],
        [18,20],
        [23,25], #Left leg
        [25,27],
        [27,29],
        [27,31],
        [29,31],
        [24,26], #Right Leg
        [26,28],
        [28,30],
        [28,32],
        [30,32],
    ]
    if len(joints)==35:
        # Connect using joints of neck and pelvis artificially computed
        extra_joints=[
            [0,34], #Head to neck
            [34,11],#Neck to l_should
            [34,12],#Neck to r_should
            [33, 34],#Neck to pelvis
            [33,23],#Pelvis to l_hip
            [33,24],#Pelvis to r_hip
        ]
        body_edges=body_edges+extra_joints
    elif len(joints)==33:
        extra_joints = [
            [11,12],  #Torse
            [11,23],
            [12,24],
            [23,24],
        ]
        body_edges=body_edges+extra_joints #Append standard torso connection
    for joint1, joint2 in body_edges:
        if (joints[joint1] is not None and joints[joint2] is not None):
            plot_3d_line(ax, joints[joint1], joints[joint2], color)


def plot_take(take,skip=1,center_on_hips=False,title=None,source="OPT"):
    figure = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')

    take_duration = len(take)
    #skip=int(take_duration/144)
    # plt.legend(legend)
    resetAxis(ax,title=title,center_on_hips=center_on_hips)
    ax.azim=60
    plt.ion()

    frame=0
    for frame_idx in range(frame, take_duration,skip):
        joints=take[frame_idx]
        if center_on_hips:
            if source=="OPT" or (source=="ZED" and (len(joints)==18) or len(joints)==34):
                joints= skeleton_utils.centerSkeletonAroundHip(joints)
            elif (source=="ZED" and len(joints)==19):
                joints = skeleton_utils.centerSkeletonAroundHip(joints,hip_id=18)
            elif source=="MPOSE":
                joints= skeleton_utils.centerSkeletonAroundHip(joints,hip_id=33)
        if source=="OPT":
            plot_3d_skeleton_OPT(joints, ax, 'black')
        elif source=="ZED":
            plot_3d_skeleton_ZED(joints, ax, 'black')
        elif source=="MPOSE":
            plot_3d_skeleton_MPOSE(joints, ax, 'black')
        # plot_3d_skeleton(pos_walk_2, ax, i,'red')
        figure.canvas.draw()
        figure.canvas.flush_events()
        ax.clear()
        resetAxis(ax,title,center_on_hips)
        plt.show()
    plt.close()