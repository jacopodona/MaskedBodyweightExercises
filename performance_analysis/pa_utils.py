import os.path

import numpy as np
import math
import skeleton_utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def createHeatmap(similarity_matrix,columns,rows,cmap="YlGnBu",path=None,title=None):
    plt.figure(figsize=(8, 5))
    sns.heatmap(similarity_matrix, annot=True,cmap=cmap, cbar=True, xticklabels=columns,
                yticklabels=rows)
    plt.tick_params(axis='both', which='major', labelsize=8)
    if title is None:
        plt.title("Full body positional similarity Score")
    else:
        plt.title(title)
    plt.tight_layout()
    # plt.show()
    if path is not None:
        plt.savefig(path, dpi=150)
    plt.show()
    plt.close()

def sigmoid_activation(value,alpha=1,beta=0):
    """
    Sigmoid activation function integrated with alpha and beta values
    :param value: x variable
    :param alpha: a controls the steepness of the curve. <1 to make it less steep, >1 to make it more steep
    :param beta: b controls the
    :return:
    """
    # Apply the sigmoid function to map to the range [0, 1]
    sigmoid_output = 1 / (1 + np.exp(-alpha*(value+beta)))

    return sigmoid_output

def getLogarithmicScore(mpjpe):
    if (mpjpe<=1):
        value=(math.log(1 / mpjpe)) **2
    else:
        value=0
    #return sigmoid_activation(value)
    return sigmoid_activation(value,alpha=1/5,beta=-1)
def computeSequenceSimilarityScore_v1(reference_sequence,testing_sequence,hip_index=0,head_index=4):
    """
    WHOLE BODY MPJPE SIMILARITY SCORE
    Compute error scalar between reference repetition and testing one by computing frame-by-frame error, average throughout the take, then averaging again.
    The error is then fed through a logarithm and a sigmoid activation function to be placed inside the [0,1] range. 1 means the testing sequence is very similar, 0 means the testing sequence is very different
    :param reference_sequence:
    :param testing_sequence:
    :return:
    """
    assert len(reference_sequence)==len(testing_sequence)
    errors=[]
    for frame_idx in range(len(reference_sequence)):
        reference_skeleton=skeleton_utils.centerSkeletonAroundHip(reference_sequence[frame_idx],hip_id=hip_index)
        reference_skeleton,_ = skeleton_utils.normalize_skeleton_joints_distance(reference_skeleton,hip_index=hip_index,head_index=head_index)
        reference_skeleton=np.array(reference_skeleton)
        test_skeleton=skeleton_utils.centerSkeletonAroundHip(testing_sequence[frame_idx])
        test_skeleton, _ = skeleton_utils.normalize_skeleton_joints_distance(test_skeleton,hip_index=hip_index,head_index=head_index)
        test_skeleton = np.array(test_skeleton)

        frame_error=skeleton_utils.computeReconstructionError(gt_skeleton=reference_skeleton,predicted_skeleton=test_skeleton)
        errors.append(frame_error)
    errors=np.array(errors)
    #avg_joint_errors = np.mean(errors, axis=0) #
    avg_frame_errors = np.mean(errors, axis=1) #Equal to array of MPJPE error
    try:
        error_scalar=np.mean(avg_frame_errors).item()
        similarity_score=getLogarithmicScore(error_scalar)
    except ZeroDivisionError:
        error_scalar=0
        similarity_score=1
    print("\tSimilarity score=",similarity_score,"Sequence error=",error_scalar,)
    return similarity_score

def computeSequenceSimilarityScore_v2(reference_sequence,testing_sequence,upper_joints_index,lower_joints_index):
    """
    UPPER AND LOWER BODY SEPARATE MPJPE SIMILARITY SCORE
    Compute error scalar between reference repetition and testing one by computing frame-by-frame error, average throughout the take, then averaging again.
    The error is then fed through a logarithm and a sigmoid activation function to be placed inside the [0,1] range. 1 means the testing sequence is very similar, 0 means the testing sequence is very different
    :param reference_sequence:
    :param testing_sequence:
    :return:
    """
    assert len(reference_sequence)==len(testing_sequence)
    upper_errors=[]
    lower_errors=[]
    for frame_idx in range(len(reference_sequence)):
        upper_reference_skeleton=skeleton_utils.centerSkeletonAroundHip(reference_sequence[frame_idx][upper_joints_index])
        upper_test_skeleton=skeleton_utils.centerSkeletonAroundHip(testing_sequence[frame_idx][upper_joints_index])
        lower_reference_skeleton = skeleton_utils.centerSkeletonAroundHip(reference_sequence[frame_idx][lower_joints_index])
        lower_test_skeleton = skeleton_utils.centerSkeletonAroundHip(testing_sequence[frame_idx][lower_joints_index])
        upper_frame_error=skeleton_utils.computeReconstructionError(gt_skeleton=upper_reference_skeleton,predicted_skeleton=upper_test_skeleton)
        lower_frame_error = skeleton_utils.computeReconstructionError(gt_skeleton=lower_reference_skeleton,predicted_skeleton=lower_test_skeleton)
        upper_errors.append(upper_frame_error)
        lower_errors.append(lower_frame_error)
    upper_errors=np.array(upper_errors)
    lower_errors=np.array(lower_errors)
    avg_upper_frame_errors = np.mean(upper_errors, axis=1) #Equal to array of MPJPE error
    avg_lower_frame_errors = np.mean(lower_errors,axis=1)

    upper_error_scalar=np.mean(avg_upper_frame_errors).item()
    upper_similarity_score=getLogarithmicScore(upper_error_scalar)

    lower_error_scalar = np.mean(avg_lower_frame_errors).item()
    lower_similarity_score = getLogarithmicScore(lower_error_scalar)

    return upper_similarity_score,lower_similarity_score

def computeBackAngle(p1,p2,frame_idx=None,plot_debug=False):
    x1,y1=p1
    x2,y2=p2
    if (x2 - x1)!=0:
        slope1 = (y2 - y1) / (x2 - x1)
    elif (x2-x1)==0:
        slope1=math.inf

    angle1 = math.degrees(math.atan(slope1))

    angle_diff = abs(90 - angle1)

    if angle_diff > 90:
        angle_diff=180-angle_diff
    if plot_debug:
        intercept = p1[1] - slope1 * p1[0]
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]

        # Create x values for the first line
        x_values_line1 = np.linspace(min(p1[0], p2[0]), max(p1[0], p2[0]), 100)

        # Calculate corresponding y values for the first line
        y_values_line1 = slope * x_values_line1 + intercept

        x_values_line2 = np.linspace(min(p1[0], p1[0]), max(p1[0], p1[0]), 100)
        y_values_line2 = np.linspace(min(p1[1], p1[1]), max(p1[1], p1[1] + 1), 100)


        plt.plot(*p1, 'ro', label='Point 1')
        plt.plot(*p2, 'bo', label='Point 2')
        plt.plot(x_values_line1, y_values_line1, label='Line 1')
        plt.plot(x_values_line2, y_values_line2, label='Line 2 (180° angle)')
        plt.text(*p1,f'a={round(angle_diff,2)}', ha='right')

        # Set labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.xlim(-1,1)
        plt.ylim(-1, 1)
        plt.legend()
        plt.title(f'2D Matplotlib Plot frame {frame_idx}')
        plt.show()

    return angle_diff

def are_hips_aligned_along_axis(hip_joint1, hip_joint2, axis='X', tolerance=0.05):
    """
    Check if the hip joints are aligned along the specified axis.

    Parameters:
        hip_joint1 (tuple): Coordinates of the first hip joint in the form (x1, y1, z1).
        hip_joint2 (tuple): Coordinates of the second hip joint in the form (x2, y2, z2).
        axis (str): Axis to check alignment along ('X' or 'Z').
        tolerance (float): Tolerance threshold for alignment check.

    Returns:
        bool: True if the hip joints are aligned along the specified axis, False otherwise.
    """
    if axis == 'X':
        # Check if x-coordinates are approximately equal within the tolerance
        return abs(hip_joint1[0] - hip_joint2[0]) <= tolerance
    elif axis == 'Z':
        # Check if z-coordinates are approximately equal within the tolerance
        return abs(hip_joint1[2] - hip_joint2[2]) <= tolerance
    else:
        raise ValueError("Invalid axis. Must be 'X' or 'Z'.")

def computeSequenceSimilarityScore_v3(reference_sequence,testing_sequence,hip_index=0,chest_index=2,projection_2d="XY"):
    """
    COMPUTE BACK BODY ANGLE ERROR
    Compute error scalar between reference repetition and testing one by computing frame-by-frame error, average throughout the take, then averaging again.
    The error is then fed through a logarithm and a sigmoid activation function to be placed inside the [0,1] range. 1 means the testing sequence is very similar, 0 means the testing sequence is very different
    :param reference_sequence:
    :param testing_sequence:
    :return:
    """
    assert len(reference_sequence) == len(testing_sequence)
    errors = []
    for frame_idx in range(len(reference_sequence)):
        reference_skeleton = skeleton_utils.centerSkeletonAroundHip(reference_sequence[frame_idx])
        test_skeleton = skeleton_utils.centerSkeletonAroundHip(testing_sequence[frame_idx])

        reference_hip=reference_skeleton[hip_index]
        reference_chest=reference_skeleton[chest_index]
        test_hip=test_skeleton[hip_index]
        test_chest = test_skeleton[chest_index]
        if projection_2d=="XY":
            reference_theta=computeBackAngle((reference_hip[0],reference_hip[1]),(reference_chest[0],reference_chest[1]),frame_idx) #Compute angle between XY 2D projection of the skeleton
            test_theta=computeBackAngle((test_hip[0],test_hip[1]),(test_chest[0],test_chest[1]),frame_idx) #Compute angle between XY 2D projection of the skeleton
        if projection_2d=="ZY":
            reference_theta=computeBackAngle((reference_hip[2],reference_hip[1]),(reference_chest[2],reference_chest[1]),frame_idx) #Compute angle between XY 2D projection of the skeleton
            test_theta=computeBackAngle((test_hip[2],test_hip[1]),(test_chest[2],test_chest[1]),frame_idx) #Compute angle between XY 2D projection of the skeleton
        angle_diff=abs(reference_theta-test_theta)
        errors.append(angle_diff)
    errors = np.array(errors)
    avg_theta_displacement = np.mean(errors)
    #similarity_score=(math.log(1 / avg_theta_displacement))
    #try:
    #    error_scalar = np.mean(avg_frame_errors).item()
    #    similarity_score = (math.log(1 / error_scalar))
    #except ZeroDivisionError:
    #    error_scalar = 0
    #    similarity_score = 1
    #print("\tSimilarity score=", similarity_score, "Sequence error=", error_scalar, )
    return avg_theta_displacement


def saveDataframe(similarity_matrix, columns, rows, dataset, exercise,method_flag):
    """

    :param similarity_matrix: matrix in numpy format
    :param columns: columnn names of matrix
    :param rows: row names of matrix
    :param dataset: source of data
    :param exercise: involved exercise
    :param method_flag: 1 for total body mpjpe, 2u for upper body. 2l for lower body, 3 for angle
    :return:
    """
    output_path=os.path.join("dataframes",dataset)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df=pd.DataFrame(similarity_matrix, columns=columns, index=rows)
    df.to_csv(os.path.join(output_path,f"{exercise}_{method_flag}.csv"),sep=";")