import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.nn
import torch.optim
import math
from tqdm import tqdm
from model import MaskPoseVIT,SpatioTemporalMaskPoseVIT,SpatialMaskPoseVITWithTemporalDependency
import data_utils
import plot_utils
import copy
import pickle as pkl
import json
import skeleton_utils
import plot_utils
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def custom_formatter(x):
    return "{:,.4f}".format(x)


def loadSequence(exercise="squat"):
    dataset_path = "data/pkl/optitrack/baseline_skeleton"
    exercise_to_load=[]
    if exercise == "squat":
        exercise_to_load = ["squat_bad.pkl"]
    elif exercise=="lunge":
        exercise_to_load = ["lunge_no_equilibrium.pkl"]
    elif exercise=="plank":
        exercise_to_load = ["plank_bad.pkl"]
    for s in exercise_to_load:
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
    return data

def loadRepetitionSequence(frames_variant,exercise):
    dataset_path = os.path.join("data/pkl/optitrack/single_repetitions", frames_variant)
    exercise_to_load = []
    keypoint_sequence=[]
    if exercise == "squat":
        variant_to_load ="squat_bad"
        repetitions=os.listdir(os.path.join(dataset_path,variant_to_load))
        exercise_to_load += [os.path.join(variant_to_load, filename) for filename in repetitions]
    elif exercise == "lunge":
        variant_to_load = "lunge_no_equilibrium"
        repetitions = os.listdir(os.path.join(dataset_path, variant_to_load))
        exercise_to_load += [os.path.join(variant_to_load, filename) for filename in repetitions]
    for s in exercise_to_load:
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        keypoint_sequence+=data
    return keypoint_sequence



def centerSkeletonsAroundHip(sequence):
    modified_sequence = copy.deepcopy(sequence)
    for skeleton_id in range(len(sequence)):
        original_skeleton = sequence[skeleton_id]
        hip_absolute = copy.deepcopy(original_skeleton[0])
        for joint_id in range(len(original_skeleton)):
            joint = original_skeleton[joint_id]
            for coordinate_id in range(len(joint)):
                modified_sequence[skeleton_id][joint_id][coordinate_id] = sequence[skeleton_id][joint_id][coordinate_id] - hip_absolute[coordinate_id]
    return modified_sequence


def subSampleSequence(sequence):
    original_fps = 340
    target_fps = 60
    subsampling_ratio = int(original_fps / target_fps)
    return sequence[::subsampling_ratio]


def forward(model, masked_sample):
    """
    Sends sample to torch tensor in correct format for pt model and returns output
    :param model: Can be single frame or multi frame model
    :param masked_sample: Masked Skeleton or Masked Sequence of frames based on the model
    :return:
    """
    input=torch.tensor(masked_sample).double()
    input=input.unsqueeze(0)
    input=input.to(device)
    output=model(input)
    output=output.squeeze(0)
    output_array=output.detach().cpu().numpy()
    return output_array



def plot_reconstructed_take_multiview(sequence,maskedSequence,model):
    figure = plt.figure(figsize=(16, 8))
    # ax = plt.axes(projection='3d')

    ax1 = figure.add_subplot(121, projection='3d')  # 1 row, 2 columns, first subplot
    ax2 = figure.add_subplot(122, projection='3d')

    take_duration = len(sequence)
    skip = 10

    plot_utils.resetAxis(ax1, title="Original")
    plot_utils.resetAxis(ax2, title="Reconstructed")

    ax1.azim = 60
    ax2.azim = 60

    plt.ion()
    for frame_idx in range(0, take_duration, skip):
        joints = sequence[frame_idx]
        reconstructed_joints=forward(model, maskedSequence[frame_idx])
        plot_utils.plot_3d_skeleton_OPT(joints, ax1, 'black')
        plot_utils.plot_3d_skeleton_OPT(reconstructed_joints, ax2, 'black')
        figure.canvas.draw()
        figure.canvas.flush_events()
        plot_utils.resetAxis(ax1,title="Original")
        plot_utils.resetAxis(ax2,title="Reconstructed")
        plt.show()


def reconstructSequenceSingleFrame(sequence, model, maskedJoints, normalize):
    reconstructed_sequence=[]#Array of lenght equal to sequence containing 3d position of reconstructed skeleton coordinates
    reconstruction_errors=np.zeros(shape=(len(sequence),len(sequence[0])))#Array of lenght equal to sequence containing per joint error of each reconstructed skeleton
    for frame_idx in tqdm(range(len(sequence)),desc="Reconstructing sequence"):
        distance_head_hip=None
        joint_offset=None
        joints = sequence[frame_idx]
        centered_joints = skeleton_utils.centerSkeletonAroundHip(joints)
        if normalize:
            centered_joints, (distance_head_hip, joint_offset) = skeleton_utils.normalize_skeleton_joints_distance(centered_joints)
            original_skeleton_absolute = (np.array(centered_joints) * distance_head_hip) + joint_offset
        masked_skeleton = skeleton_utils.maskSkeleton(skeleton=centered_joints, mask_joints_id=maskedJoints)
        reconstructed_joints = forward(model, masked_skeleton)
        if normalize:
            reconstructed_absolute_skeleton = (np.array(reconstructed_joints) * distance_head_hip) + joint_offset
            error=skeleton_utils.computeReconstructionError(predicted_skeleton=reconstructed_absolute_skeleton,gt_skeleton=original_skeleton_absolute)
            reconstructed_sequence.append(reconstructed_absolute_skeleton)
        else:
            error=skeleton_utils.computeReconstructionError(predicted_skeleton=reconstructed_joints,gt_skeleton=centered_joints)
            reconstructed_sequence.append(reconstructed_joints)
        reconstruction_errors[frame_idx]=error
    per_joint_error = np.mean(reconstruction_errors, axis=0)
    results = skeleton_utils.printPerJointError(per_joint_error)
    return reconstructed_sequence,results


def reconstructSequenceTemporalWindow(sequence, model, maskedJoints, normalize,window_size):
    reconstructed_sequence=[]#Array of lenght equal to sequence containing 3d position of reconstructed skeleton coordinates
    reconstruction_errors=[]#Array of lenght equal to sequence containing per joint error of each reconstructed skeleton
    sub_sequences=data_utils.slice_sequence_into_windows(sequence,window_size=window_size)
    for window_idx in range(len(sub_sequences)):
        clean_sub_sequence=sub_sequences[window_idx]
        distance_head_hip=None
        joint_offset=None

        #STEP 1 CENTER AROUND HIP
        for frame_idx in range(len(clean_sub_sequence)):
            clean_sub_sequence[frame_idx] = skeleton_utils.centerSkeletonAroundHip(clean_sub_sequence[frame_idx])
        #STEP 2 NORMALIZE
        if normalize:
            #centered_joints, (distance_head_hip, joint_offset) = skeleton_utils.normalize_skeleton_joints_distance(centered_joints)
            ##original_skeleton_absolute = (np.array(centered_joints) * distance_head_hip) + joint_offset
            clean_sub_sequence,normalization_info=skeleton_utils.normalize_sequence(clean_sub_sequence,return_normalization_values=True)
        #STEP 3 MASK
        masked_sub_sequence=skeleton_utils.maskSequence(clean_sub_sequence,maskedJoints)
        #STEP 4 RECONSTRUCT WINDOW
        reconstructed_sub_sequence = forward(model, masked_sub_sequence)
        #STEP 5 COMPUTE ERROR (RECONSTRUCT ORIGINAL SKELETON COORDINATES IF NORMALIZATION WAS MADE)
        if normalize:
            for frame_idx in range(len(reconstructed_sub_sequence)):
                reconstructed_joints=reconstructed_sub_sequence[frame_idx]
                frame_distance_head_hip,frame_joint_offset=normalization_info[frame_idx]
                reconstructed_absolute_skeleton = (np.array(reconstructed_joints) * frame_distance_head_hip) + frame_joint_offset
                original_skeleton_absolute= (np.array(clean_sub_sequence[frame_idx]) * frame_distance_head_hip) + frame_joint_offset
                error=skeleton_utils.computeReconstructionError(predicted_skeleton=reconstructed_absolute_skeleton,gt_skeleton=original_skeleton_absolute)
                reconstructed_sequence.append(reconstructed_absolute_skeleton)
                reconstruction_errors.append(error)
    reconstruction_errors=np.array(reconstruction_errors)
    per_joint_error = np.mean(reconstruction_errors, axis=0)
    results = skeleton_utils.printPerJointError(per_joint_error)
    return reconstructed_sequence,results



def plot_reconstructed_take_single_view(sequence,model,maskedJoints,normalize,use_rep_dataset):
    sequence=np.array(sequence)
    figure = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection='3d')

    colors = ['black', 'red']
    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
    labels = ['Ground truth', 'Prediction']
    plt.legend(lines, labels)

    take_duration = len(sequence)
    skip = 1

    plot_utils.resetAxis(ax, title="Original",center_on_hips=True)

    reconstructed_sequence,errors=reconstructSequenceSingleFrame(sequence,model,maskedJoints,normalize)
    print("Joint errors:")
    print(errors)
    ax.azim = 60

    plt.ion()
    for frame_idx in range(0, take_duration, skip):
        joints = sequence[frame_idx]
        reconstructed_joints=reconstructed_sequence[frame_idx]
        centered_joints=skeleton_utils.centerSkeletonAroundHip(joints)
        plot_utils.plot_3d_skeleton_OPT(centered_joints, ax, 'black')
        plot_utils.plot_3d_skeleton_OPT(reconstructed_joints, ax, 'red')
        figure.canvas.draw()
        figure.canvas.flush_events()
        plot_utils.resetAxis(ax,title="Original",center_on_hips=True)
        plt.show()
        plt.legend(lines, labels)


def normalizeSequence(sequence):
    modified_sequence = copy.deepcopy(sequence)
    for skeleton_id in range(len(sequence)):
        original_skeleton = sequence[skeleton_id]
        normalized_skeleton,_=skeleton_utils.normalize_skeleton_joints_distance(original_skeleton)
        modified_sequence[skeleton_id]=normalized_skeleton
    return modified_sequence

def normalizeSkeleton(skeleton):
    skeleton_copy = copy.deepcopy(skeleton)
    normalized_skeleton,=skeleton_utils.normalize_skeleton_joints_distance(skeleton_copy)
    return normalized_skeleton


def loadRepetitionSequences(exercise,frames=21):
    dataset_path= f"data/pkl/optitrack/single_repetitions/{frames}_frames"
    variants = os.listdir(dataset_path)
    exercise_to_load=[]
    samples=[]
    return_names=[]
    val_repetitions = ["rep_4.pkl", "rep_7.pkl"]
    if exercise == "squat" or exercise=="combined":
        for variant in variants:
            if "squat" in variant:
                exercise_to_load += [os.path.join(variant, filename) for filename in val_repetitions]
    if exercise == "lunge" or exercise=="combined":
        for variant in variants:
            if "lunge" in variant:
                exercise_to_load += [os.path.join(variant, filename) for filename in val_repetitions]
    if exercise == "plank" or exercise=="combined":
        for variant in variants:
            if "plank" in variant:
                exercise_to_load += [os.path.join(variant, filename) for filename in val_repetitions]
    for s in exercise_to_load:
        name=s.split("\\")[0]
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        samples.append(data)
        return_names.append(name)
    return return_names,samples


def plot_reconstructed_take_single_view_repetition_dataset_SINGLE_FRAME(sequences, repetitions_name, model, masked_joints, normalize,csv_save_path,show=True):
    reconstructed_sequences=[]
    reconstruction_errors=[]
    figure = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection='3d')


    colors = ['black', 'red']
    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
    labels = ['Ground truth', 'Prediction']
    plt.legend(lines, labels)
    for sequence_idx in range(len(sequences)):
        sequence=sequences[sequence_idx]
        name=repetitions_name[sequence_idx]
        sequence = np.array(sequence)
        take_duration = len(sequence)
        skip = 1

        plot_utils.resetAxis(ax, title="Original", center_on_hips=True)
        reconstructed_sequence, errors = reconstructSequenceSingleFrame(sequence, model, masked_joints, normalize)
        reconstructed_sequences.append(reconstructed_sequence)
        header=list(errors.keys())
        reconstruction_errors.append(list(errors.values()))

    print()
    # Create a DataFrame
    df = pd.DataFrame({'label': repetitions_name, 'values': reconstruction_errors})

    # Convert the 'values' column to a NumPy array
    df['values'] = df['values'].apply(np.array)

    # Group by 'label' and calculate the mean for each group
    result = df.groupby('label')['values'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index().to_numpy()

    # Set the print options
    np.set_printoptions(formatter={'float_kind': custom_formatter})
    average=np.zeros(len(result[0][1]))
    header=["Exercise"]+header
    rows=[header]
    for idx in range(len(result)):
        print(f"Exercise {result[idx][0]}: {result[idx][1]}")
        formatted_values = ['%.4f' % value for value in result[idx][1].tolist()]
        rows.append([result[idx][0]]+formatted_values)
        average+=result[idx][1]/len(result)
    average_formatted=['%.4f' % value for value in average.tolist()]
    rows.append(["Average"] + average_formatted)
    print(f"Average: {average}")

    with open(os.path.join(csv_save_path,"output.csv"), mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file, delimiter=';')

        # Write the list to the CSV file
        csv_writer.writerows(rows)
    ax.azim = 60

    if show:
        plt.ion()
        skip = 1
        for sequence_idx in range(len(sequences)):
            sequence=sequences[sequence_idx]
            reconstructed_sequence=reconstructed_sequences[sequence_idx]
            take_duration=len(reconstructed_sequence)
            title = repetitions_name[sequence_idx]
            for frame_idx in range(0, take_duration, skip):
                joints = sequence[frame_idx]
                reconstructed_joints = reconstructed_sequence[frame_idx]
                centered_joints = skeleton_utils.centerSkeletonAroundHip(joints)
                plot_utils.plot_3d_skeleton_OPT(centered_joints, ax, 'black')
                plot_utils.plot_3d_skeleton_OPT(reconstructed_joints, ax, 'red')
                figure.canvas.draw()
                figure.canvas.flush_events()
                plot_utils.resetAxis(ax, title=title, center_on_hips=True)
                plt.show()
                plt.legend(lines, labels)
    plt.close()

def plot_reconstructed_take_single_view_repetition_dataset_TEMPORAL_WINDOW(sequences, repetitions_name, model, masked_joints,
                                                                           normalize,window_size, csv_save_path,show=True):
    reconstructed_sequences = []
    reconstruction_errors = []
    figure = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection='3d')

    colors = ['black', 'red']
    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
    labels = ['Ground truth', 'Prediction']
    plt.legend(lines, labels)
    for sequence_idx in range(len(sequences)):
        sequence = sequences[sequence_idx]
        sequence = np.array(sequence)

        plot_utils.resetAxis(ax, title="Original", center_on_hips=True)
        reconstructed_sequence, errors = reconstructSequenceTemporalWindow(sequence, model, masked_joints, normalize,window_size)
        reconstructed_sequences.append(reconstructed_sequence)
        header = list(errors.keys())
        reconstruction_errors.append(list(errors.values()))

    print()
    # Create a DataFrame
    df = pd.DataFrame({'label': repetitions_name, 'values': reconstruction_errors})

    # Convert the 'values' column to a NumPy array
    df['values'] = df['values'].apply(np.array)

    # Group by 'label' and calculate the mean for each group
    result = df.groupby('label')['values'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index().to_numpy()

    # Set the print options
    np.set_printoptions(formatter={'float_kind': custom_formatter})
    average = np.zeros(len(result[0][1]))
    header = ["Exercise"] + header
    rows = [header]
    for idx in range(len(result)):
        print(f"Exercise {result[idx][0]}: {result[idx][1]}")
        formatted_values = ['%.4f' % value for value in result[idx][1].tolist()]
        rows.append([result[idx][0]] + formatted_values)
        average += result[idx][1] / len(result)
    average_formatted = ['%.4f' % value for value in average.tolist()]
    rows.append(["Average"] + average_formatted)
    print(f"Average: {average}")

    with open(os.path.join(csv_save_path, "output.csv"), mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file, delimiter=';')

        # Write the list to the CSV file
        csv_writer.writerows(rows)
    ax.azim = 60

    if show:
        plt.ion()
        skip = 1
        for sequence_idx in range(len(sequences)):
            sequence = sequences[sequence_idx]
            reconstructed_sequence = reconstructed_sequences[sequence_idx]
            take_duration = len(reconstructed_sequence)
            title = repetitions_name[sequence_idx]
            for frame_idx in range(0, take_duration, skip):
                joints = sequence[frame_idx]
                reconstructed_joints = reconstructed_sequence[frame_idx]
                centered_joints = skeleton_utils.centerSkeletonAroundHip(joints)
                plot_utils.plot_3d_skeleton_OPT(centered_joints, ax, 'black')
                plot_utils.plot_3d_skeleton_OPT(reconstructed_joints, ax, 'red')
                figure.canvas.draw()
                figure.canvas.flush_events()
                plot_utils.resetAxis(ax, title=title, center_on_hips=True)
                plt.show()
                plt.legend(lines, labels)
        plt.close()


def visualize_single_frame(PATH,show=True):
    val_data = "sequence"  # sequence or repetitions
    frames = "21_frames"

    model_path = os.path.join(PATH, "model.pt")
    json_path = os.path.join(PATH, "parameters.json")
    with open(json_path, 'r', ) as json_file:
        params = json.load(json_file)

    visible_joints = params["visible_joints"]
    exercise = params["exercise"]
    num_joints = params["num_joints"]
    normalize_skeleton = params["normalize_skeleton"]
    try:
        use_rep_dataset = params["use_rep_dataset"]
    except Exception:
        use_rep_dataset = False

    embed_dimensionality = params["linear_embed_dimensionality"]
    attention_heads = params["vit_attention_heads"]
    hidden_dimensionality = params["vit_hidden_dimensionality"]
    num_encoder_layers = params["vit_encoder_layers"]

    masked_joints = [item for item in range(21) if item not in visible_joints]
    # model=MaskPoseVIT(num_joints=21,embed_dim=64,num_heads=8,hidden_dim=1536,num_layers=6).double().to(device)
    model = MaskPoseVIT(num_joints=num_joints, embed_dim=embed_dimensionality, num_heads=attention_heads,
                        hidden_dim=hidden_dimensionality, num_layers=num_encoder_layers,
                        input_visible_joints=visible_joints).double().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if use_rep_dataset:
        repetitions, sequences = loadRepetitionSequences(exercise=exercise)
        plot_reconstructed_take_single_view_repetition_dataset_SINGLE_FRAME(sequences, repetitions, model,
                                                                            masked_joints, normalize=normalize_skeleton,
                                                                            csv_save_path=PATH,show=show)
    else:
        if val_data == "sequence":
            sequence = loadSequence(exercise=exercise)
            sequence = subSampleSequence(sequence)
        elif val_data == "repetitions":
            sequence = loadRepetitionSequence(frames_variant=frames, exercise=exercise)

        plot_reconstructed_take_single_view(sequence, model, masked_joints, normalize=normalize_skeleton)


def visualize_temporal_window(PATH,show=True):

    model_path = os.path.join(PATH, "model.pt")
    json_path = os.path.join(PATH, "parameters.json")
    with open(json_path, 'r', ) as json_file:
        params = json.load(json_file)

    num_joints=params["num_joints"]
    visible_joints = params["visible_joints"]
    exercise = params["exercise"]
    normalize_skeleton = params["normalize_skeleton"]
    window_size = params["window_size"]
    frames=params["rep_size"]

    embed_dimensionality = params["linear_embed_dimensionality"]
    attention_heads = params["vit_attention_heads"]
    hidden_dimensionality = params["vit_hidden_dimensionality"]
    num_encoder_layers = params["vit_encoder_layers"]
    decoder_hidden_dim = params["decoder_hidden_dim"]
    decoder_num_layers = params["decoder_num_layers"]

    masked_joints = [item for item in range(21) if item not in visible_joints]

    try:
        model = SpatioTemporalMaskPoseVIT(num_joints=num_joints,num_frames=window_size,embed_dim=embed_dimensionality,num_heads=attention_heads,hidden_dim=hidden_dimensionality,num_layers=num_encoder_layers,input_visible_joints=visible_joints,
                                                    decoder_hidden_dim=decoder_hidden_dim,decoder_num_layers=decoder_num_layers).double().to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loaded SpatioTemporalMaskPoseVIT")
    except:
        model = SpatialMaskPoseVITWithTemporalDependency(num_joints=num_joints, num_frames=window_size, embed_dim=embed_dimensionality,
                                          num_heads=attention_heads, hidden_dim=hidden_dimensionality,
                                          num_layers=num_encoder_layers, input_visible_joints=visible_joints,
                                          decoder_hidden_dim=decoder_hidden_dim,
                                          decoder_num_layers=decoder_num_layers).double().to(device)

        model.load_state_dict(torch.load(model_path))
        print("Loaded SpatialMaskPoseVITWithTemporalDependency")
    #model.load_state_dict(torch.load(model_path))
    model.eval()
    repetitions, sequences = loadRepetitionSequences(exercise=exercise,frames=frames)
    plot_reconstructed_take_single_view_repetition_dataset_TEMPORAL_WINDOW(sequences, repetitions, model, masked_joints,
                                                                        normalize=normalize_skeleton,window_size=window_size,
                                                                        csv_save_path=PATH,show=show)

def main():
    PATH = "nets/single_frame/sample_networks/squat"
    #PATH = "nets/single_frame/sample_networks/lunge"
    #PATH = "nets/single_frame/sample_networks/plank"
    if "single_frame" in PATH:
        visualize_single_frame(PATH)
    elif "temporal_window" in PATH:
        visualize_temporal_window(PATH)

if __name__ == '__main__':
    main()