import numpy as np
import copy
import random
import torch
import math
import plot_utils

MASKING_TOKEN=float(-1e10)

def x_rotation(theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]],dtype=float)
    return R

def y_rotation(theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]],dtype=float)
    return R

def z_rotation(theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]],dtype=float)
    return R

def maskSequence(sequence,mask_joints_id):
    masked_sequence=sequence.copy()
    for frame_idx in range(len(sequence)):
        skeleton=sequence[frame_idx]
        masked_skeleton = []
        for i in range(0,len(skeleton)):
            if i in mask_joints_id:
                masked_skeleton.append([MASKING_TOKEN]*3)
            else:
                masked_skeleton.append(skeleton[i])
        masked_sequence[frame_idx]=masked_skeleton
    return masked_sequence

def maskSkeleton(skeleton,mask_joints_id):
    masked_skeleton = []
    for i in range(0,len(skeleton)):
        if i in mask_joints_id:
            #masked_skeleton.append([float("-inf")]*3)
            masked_skeleton.append([MASKING_TOKEN] * 3)
        else:
            masked_skeleton.append(skeleton[i])
    return masked_skeleton

def normalize_skeleton_joints_distance(skeleton_to_normalize, head_index = 4, hip_index = 0):
    original_skeleton = np.array(skeleton_to_normalize)
    # Extract the head and hip joint positions
    head_position = original_skeleton[head_index]
    hip_position = original_skeleton[hip_index]

    # Calculate the distance between the head and hip
    distance_head_hip = np.linalg.norm(head_position - hip_position)

    # Normalize each joint position
    normalized_skeleton = (original_skeleton - hip_position) / distance_head_hip
    joint_offset = original_skeleton - normalized_skeleton * distance_head_hip
    normalized_skeleton_list = normalized_skeleton.tolist()
    return normalized_skeleton_list,(distance_head_hip,joint_offset)

def centerSkeletonAroundHip(skeleton,hip_id=0):
    centered_skeleton=copy.deepcopy(skeleton)
    hip_absolute=copy.deepcopy(skeleton[hip_id])
    for joint_id in range(len(skeleton)):
        joint=skeleton[joint_id]
        for coordinate_id in range(len(joint)):
            centered_skeleton[joint_id][coordinate_id]=skeleton[joint_id][coordinate_id]-hip_absolute[coordinate_id]
    return centered_skeleton


def augmentSkeleton(skeleton):
    xz_range=10
    augmented_skeleton=copy.deepcopy(skeleton)
    if (np.random.randint(low=0, high=2, size=(1,))): #Rotate around x axis
        angle = random.uniform(-xz_range, xz_range)
        R = torch.tensor(x_rotation(np.radians(angle)), dtype=torch.float)
        augmented_skeleton = R.matmul(torch.tensor(augmented_skeleton).T).T.numpy().tolist()
    if (np.random.randint(low=0, high=2, size=(1,))): #Rotate around Z axis
        angle = random.uniform(-xz_range, xz_range)
        R = torch.tensor(z_rotation(np.radians(angle)), dtype=torch.float)
        augmented_skeleton = R.matmul(torch.tensor(augmented_skeleton).T).T.numpy().tolist()
    if (np.random.randint(low=0, high=2, size=(1,))):#Rotate around Y axis
        angle = random.uniform(0, 360)
        R = torch.tensor(y_rotation(np.radians(angle)), dtype=torch.float)
        augmented_skeleton = R.matmul(torch.tensor(augmented_skeleton).T).T.numpy().tolist()
    return augmented_skeleton

def euclidean_distance_3d(point1, point2):
    """
    Compute the Euclidean distance between two points in 3D space.

    Parameters:
    - point1: Tuple or list representing the coordinates of the first point (x, y, z).
    - point2: Tuple or list representing the coordinates of the second point (x, y, z).

    Returns:
    - Euclidean distance between the two points.
    """
    if len(point1) != 3 or len(point2) != 3:
        raise ValueError("Both points must be in 3D space with coordinates (x, y, z)")

    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
    return distance

def printPerJointError(per_joint_error):
    results={}
    for joint_idx in range(len(per_joint_error)):
        joint_error=per_joint_error[joint_idx]
        results[str(joint_idx)]=joint_error
    #print("Joint errors"+str(results))
    return results

def computeBatchReconstructionError(predictions:torch.Tensor,targets:torch.Tensor):
    predictions=predictions.cpu().detach().numpy()
    targets=targets.cpu().detach().numpy()
    num_samples,num_joints,_=targets.shape
    errors=np.zeros(shape=(num_samples,num_joints))
    for sample_idx in range(len(predictions)):
        skeleton=predictions[sample_idx]
        for joint_idx in range(len(skeleton)):
            errors[sample_idx][joint_idx]=euclidean_distance_3d(predictions[sample_idx][joint_idx],targets[sample_idx][joint_idx])
    return errors

def computeReconstructionError(predicted_skeleton:np.ndarray,gt_skeleton:np.ndarray):
    num_joints,num_coordinates=gt_skeleton.shape
    errors=np.zeros(shape=num_joints)
    for joint_idx in range(len(predicted_skeleton)):
        errors[joint_idx]=euclidean_distance_3d(predicted_skeleton[joint_idx],gt_skeleton[joint_idx])
    return errors


def augmentSequence(sequence):
    x_angle = random.uniform(-10, 10) #Sample augmentation degree for whole sequence
    y_angle=random.uniform(0, 360) #Sample augmentation degree for whole sequence
    z_angle=random.uniform(-10, 10) #Sample augmentation degree for whole sequence
    augment_x=np.random.randint(low=0, high=2, size=(1,)) #Sample augmentation probability for whole sequence
    augment_y = np.random.randint(low=0, high=2, size=(1,)) #Sample augmentation probability for whole sequence
    augment_z = np.random.randint(low=0, high=2, size=(1,)) #Sample augmentation probability for whole sequence
    for frame_idx in range(len(sequence)):
        augmented_skeleton=sequence[frame_idx]
        if augment_x:  # Rotate around X axis
            R = torch.tensor(x_rotation(np.radians(x_angle)), dtype=torch.float)
            augmented_skeleton = R.matmul(torch.tensor(augmented_skeleton).T).T.numpy().tolist()
        if augment_z:  # Rotate around Z axis
            R = torch.tensor(z_rotation(np.radians(z_angle)), dtype=torch.float)
            augmented_skeleton = R.matmul(torch.tensor(augmented_skeleton).T).T.numpy().tolist()
        if augment_y:  # Rotate around Y axis
            R = torch.tensor(y_rotation(np.radians(y_angle)), dtype=torch.float)
            augmented_skeleton = R.matmul(torch.tensor(augmented_skeleton).T).T.numpy().tolist()
        sequence[frame_idx]=augmented_skeleton
    return sequence


def normalize_sequence(clean_sequence,head_index=4,hip_index=0,return_normalization_values=False):
    normalized_sequence=clean_sequence.copy()
    normalization_values=[]
    for frame_idx in range(len(clean_sequence)):
        original_skeleton = clean_sequence[frame_idx]
        modified_skeleton, frame_normalization_values = normalize_skeleton_joints_distance(skeleton_to_normalize=original_skeleton,head_index=head_index,hip_index=hip_index)
        normalized_sequence[frame_idx] = modified_skeleton
        normalization_values.append(frame_normalization_values)
    if return_normalization_values:
        return normalized_sequence,normalization_values
    else:
        return normalized_sequence