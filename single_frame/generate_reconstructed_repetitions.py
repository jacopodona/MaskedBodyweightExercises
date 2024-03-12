import torch
from model import MaskPoseVIT
import json
import pickle as pkl
import os
import skeleton_utils
import numpy as np
import plot_utils

device="cuda" if torch.cuda.is_available() else "cpu"


def loadRepetitions(exercise_path):
    repetitions=[]
    names=[]
    for rep in os.listdir(exercise_path):
        with open(os.path.join(exercise_path,rep), 'rb') as file:
            frames = pkl.load(file)
        repetitions.append(frames)
        names.append(rep)
    return repetitions,names


def main(source_dir):
    best_models_dirs=[
    "../nets/single_frame/sample_networks/squat","../nets/single_frame/sample_networks/lunge","../nets/single_frame/sample_networks/plank"]
    repetition_exercises=os.listdir(source_dir)
    for model_dir in best_models_dirs:
        with open(os.path.join(model_dir, "parameters.json"), 'r') as file:
            params = json.load(file)
        exercise = params["exercise"]
        num_joints = params["num_joints"]
        embed_dimensionality = params["linear_embed_dimensionality"]
        attention_heads = params["vit_attention_heads"]
        hidden_dimensionality = params["vit_hidden_dimensionality"]
        num_encoder_layers = params["vit_encoder_layers"]
        visible_joints = params["visible_joints"]

        model_path=os.path.join(model_dir, "model.pt")
        model = MaskPoseVIT(num_joints=num_joints, embed_dim=embed_dimensionality, num_heads=attention_heads,
                            hidden_dim=hidden_dimensionality, num_layers=num_encoder_layers,
                            input_visible_joints=visible_joints).double().to(device)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        for repetitions_dir in repetition_exercises:
            if exercise in repetitions_dir: #if the loaded net is trained on the exercise
                exercise_path=os.path.join(source_dir,repetitions_dir)
                repetitions,names=loadRepetitions(exercise_path)

                for rep_idx in range(len(repetitions)):
                    sequence=repetitions[rep_idx]
                    reconstructed_sequence=[]
                    for frame_idx in range(len(sequence)):
                        skeleton=sequence[frame_idx]
                        skeleton_to_reconstruct=skeleton_utils.centerSkeletonAroundHip(skeleton)
                        skeleton_to_reconstruct, (distance_head_hip, joint_offset) = skeleton_utils.normalize_skeleton_joints_distance(skeleton_to_reconstruct)
                        original_skeleton_absolute = (np.array(skeleton_to_reconstruct) * distance_head_hip) + joint_offset
                        skeleton_to_reconstruct=skeleton_utils.maskSkeleton(skeleton_to_reconstruct,model.masked_joints_index)
                        reconstructed_skeleton=model.inference(skeleton_to_reconstruct)
                        reconstructed_absolute_skeleton = (np.array(reconstructed_skeleton) * distance_head_hip) + joint_offset
                        reconstructed_sequence.append(reconstructed_absolute_skeleton)
                    output_path=exercise_path.replace("optitrack/","reconstructed_optitrack_test/")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    output_pkl_name=os.path.join(output_path,names[rep_idx])
                    with open(output_pkl_name, 'wb') as pickle_file:
                        pkl.dump(reconstructed_sequence, pickle_file)

if __name__ == '__main__':
    repetition_source="../data/pkl/optitrack/single_repetitions/21_frames"
    main(repetition_source)