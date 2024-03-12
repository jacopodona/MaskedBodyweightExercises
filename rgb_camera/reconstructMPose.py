import os
import json
import pickle as pkl
import torch
import numpy as np
import skeleton_utils
import plot_utils
from model import MaskPoseVIT
from visualizeReconstructedSkeleton import forward
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import pickle

device="cuda" if torch.cuda.is_available() else "cpu"

# best models
SQUAT_MODEL_PATH = "../nets/single_frame/sample_networks/squat"
LUNGE_MODEL_PATH = "../nets/single_frame/sample_networks/lunge"
PLANK_MODEL_PATH = "../nets/single_frame/sample_networks/plank"

def reconstructMPOSESequence(model, sequence):
    OPT_MPOSE_mapping={0:33,4:0,6:11,10:12,14:25,17:26, #for standard reconstruction joints
                       8:15,12:16, #for variant using wrist instead of shoulders
                       15:27#for variant using only left side
                       }
    opt_joints=model.visible_joints_index
    vit_joints_to_mask=model.masked_joints_index
    mpose_joints = [OPT_MPOSE_mapping[joint] for joint in opt_joints]
    reconstructed_sequence=[]
    for frame_idx in range(len(sequence)):
        mpose_skeleton=np.array(sequence[frame_idx])
        skeleton_to_reconstruct=np.zeros(shape=(21,3))
        skeleton_to_reconstruct[opt_joints]=mpose_skeleton[mpose_joints]
        skeleton_to_reconstruct=skeleton_utils.centerSkeletonAroundHip(skeleton_to_reconstruct)
        skeleton_to_reconstruct,(distance_head_hip, joint_offset)=skeleton_utils.normalize_skeleton_joints_distance(skeleton_to_reconstruct)
        skeleton_to_reconstruct=skeleton_utils.maskSkeleton(skeleton_to_reconstruct,vit_joints_to_mask)
        reconstructed_skeleton=forward(model,skeleton_to_reconstruct)
        reconstructed_sequence.append(reconstructed_skeleton)
    return reconstructed_sequence



def plotSkeletonComparisonSingleView(mpose_sequence, opt_sequence,skip=1,title=None):
    figure = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection='3d')

    colors = ['black', 'red']
    lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
    labels = ['Ground truth', 'Prediction']
    plt.legend(lines, labels)

    take_duration = len(mpose_sequence)

    plot_utils.resetAxis(ax, title=title, center_on_hips=True)
    plt.ion()
    for frame_idx in range(0, take_duration, skip):
        mpose_joints = mpose_sequence[frame_idx]
        opt_joints = opt_sequence[frame_idx]
        plot_utils.plot_3d_skeleton_MPOSE(mpose_joints, ax, 'black')
        plot_utils.plot_3d_skeleton_OPT(opt_joints, ax, 'red')
        figure.canvas.draw()
        figure.canvas.flush_events()
        plot_utils.resetAxis(ax, title=title, center_on_hips=True)
        plt.show()
        plt.legend(lines, labels)


def main(model_dir,data_root_dir):
    model_path=os.path.join(model_dir,"model.pt")
    with open(os.path.join(model_dir,"parameters.json"), 'r') as file:
        params = json.load(file)
    exercise=params["exercise"]
    num_joints=params["num_joints"]
    embed_dimensionality=params["linear_embed_dimensionality"]
    attention_heads=params["vit_attention_heads"]
    hidden_dimensionality=params["vit_hidden_dimensionality"]
    num_encoder_layers=params["vit_encoder_layers"]
    visible_joints=params["visible_joints"]
    model = MaskPoseVIT(num_joints=num_joints, embed_dim=embed_dimensionality, num_heads=attention_heads,
                        hidden_dim=hidden_dimensionality, num_layers=num_encoder_layers,
                        input_visible_joints=visible_joints).double().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    takes_list = os.listdir(data_root_dir)
    reconstructed_take_output_dir = data_root_dir.replace("rgb", "reconstructed_rgb").replace("single_repetitions_ok","single_repetitions")
    if not os.path.exists(reconstructed_take_output_dir):
        os.makedirs(reconstructed_take_output_dir)
    for idx in range(len(takes_list)):
        takename=takes_list[idx]
        sequence_path=os.path.join(data_root_dir,takename)
        with open(sequence_path, 'rb') as file:
            mpose_sequence = pkl.load(file)
        reconstructed_opt_sequence=reconstructMPOSESequence(model, mpose_sequence)
        for frame_idx in range(len(mpose_sequence)):
            original_skeleton=mpose_sequence[frame_idx]
            updated_skeleton=skeleton_utils.centerSkeletonAroundHip(original_skeleton)
            if len(original_skeleton)==35:
                updated_skeleton,_=skeleton_utils.normalize_skeleton_joints_distance(updated_skeleton,head_index=0,hip_index=33)
            mpose_sequence[frame_idx]=updated_skeleton
        #plot_utils.plot_take(reconstructed_opt_skeleton,skip=5,center_on_hips=True)
        #plotSkeletonComparisonSingleView(mpose_sequence, reconstructed_opt_sequence,skip=5,title=takename)
        output_path = os.path.join(reconstructed_take_output_dir, takename)
        with open(output_path, 'wb') as file:
            pickle.dump(reconstructed_opt_sequence, file)

def reconstruct_single_sequence(mpose_sequence,model_dir):
    model_path = os.path.join(model_dir, "model.pt")
    with open(os.path.join(model_dir, "parameters.json"), 'r') as file:
        params = json.load(file)
    exercise = params["exercise"]
    num_joints = params["num_joints"]
    embed_dimensionality = params["linear_embed_dimensionality"]
    attention_heads = params["vit_attention_heads"]
    hidden_dimensionality = params["vit_hidden_dimensionality"]
    num_encoder_layers = params["vit_encoder_layers"]
    visible_joints = params["visible_joints"]
    model = MaskPoseVIT(num_joints=num_joints, embed_dim=embed_dimensionality, num_heads=attention_heads,
                        hidden_dim=hidden_dimensionality, num_layers=num_encoder_layers,
                        input_visible_joints=visible_joints).double().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    reconstructed_opt_sequence=reconstructMPOSESequence(model, mpose_sequence)
    for frame_idx in range(len(mpose_sequence)):
        original_skeleton=mpose_sequence[frame_idx]
        updated_skeleton=skeleton_utils.centerSkeletonAroundHip(original_skeleton)
        if len(original_skeleton)==35:
            updated_skeleton,_=skeleton_utils.normalize_skeleton_joints_distance(updated_skeleton,head_index=0,hip_index=33)
        mpose_sequence[frame_idx]=updated_skeleton
    plotSkeletonComparisonSingleView(mpose_sequence, reconstructed_opt_sequence,skip=10)

def reconstruct_repetition_dataset():
    datapath = "../data/pkl/rgb/single_repetitions_ok/21_frames"
    exercises = os.listdir(datapath)
    print(f"Reconstructing MPose takes of path:{datapath}")
    for idx in tqdm(range(len(exercises))):
        e = exercises[idx]
        if "squat" in e:
            reps_path = os.path.join(datapath, e)
            main(data_root_dir=reps_path, model_dir=SQUAT_MODEL_PATH)
        if "lunge" in e:
            reps_path = os.path.join(datapath, e)
            main(data_root_dir=reps_path, model_dir=LUNGE_MODEL_PATH)
        elif "plank" in e:
            reps_path = os.path.join(datapath, e)
            main(data_root_dir=reps_path, model_dir=PLANK_MODEL_PATH)



if __name__ == '__main__':
    reconstruct_repetition_dataset()