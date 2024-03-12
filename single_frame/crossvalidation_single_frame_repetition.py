import csv
import json
import os
import cv2
import numpy as np
import torch.nn
import torch.optim
from tqdm import tqdm
from model import MaskPoseVIT
import data_utils
import datetime
import plot_utils
import skeleton_utils
import pickle as pkl
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def custom_formatter(x):
    return "{:,.4f}".format(x)

def loadRepetitionSequences(exercise,crossvalidation_id):
    dataset_path="../data/pkl/optitrack/single_repetitions/21_frames"
    variants = os.listdir(dataset_path)
    exercise_to_load=[]
    samples=[]
    return_names=[]
    if crossvalidation_id==0:
        val_repetitions = ["rep_0.pkl","rep_5.pkl"]
    if crossvalidation_id==1:
        val_repetitions = ["rep_1.pkl", "rep_6.pkl"]
    if crossvalidation_id==2:
        val_repetitions = ["rep_2.pkl", "rep_7.pkl"]
    if crossvalidation_id==3:
        val_repetitions = ["rep_3.pkl", "rep_8.pkl"]
    if crossvalidation_id==4:
        val_repetitions = ["rep_4.pkl", "rep_8.pkl"]
    if exercise == "squat":
        for variant in variants:
            if "squat" in variant:
                exercise_to_load += [os.path.join(variant, filename) for filename in val_repetitions]
    elif exercise == "lunge":
        for variant in variants:
            if "lunge" in variant:
                exercise_to_load += [os.path.join(variant, filename) for filename in val_repetitions]
    elif exercise == "plank":
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

def reconstructSequence(sequence, model, maskedJoints, normalize):
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
        reconstructed_joints = reconstructJoints(model, masked_skeleton)
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

def reconstructJoints(model, maskedSkeleton):
    input=torch.tensor(maskedSkeleton).double()
    input=input.unsqueeze(0)
    input=input.to(device)
    output=model(input)
    output=output.squeeze(0)
    output_array=output.detach().cpu().numpy()
    return output_array

def evaluateModel(model, exercise, crossvalidation_id, masked_joints,output_path):
    reconstructed_sequences = []
    reconstruction_errors = []

    repetitions_name, sequences = loadRepetitionSequences(exercise=exercise,crossvalidation_id=crossvalidation_id)
    for sequence_idx in range(len(sequences)):
        sequence = sequences[sequence_idx]
        name = repetitions_name[sequence_idx]
        sequence = np.array(sequence)
        take_duration = len(sequence)
        skip = 1

        reconstructed_sequence, errors = reconstructSequence(sequence, model, masked_joints, normalize=True)
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

    with open(os.path.join(output_path,f"result_{exercise}_fold_id_{str(crossvalidation_id)}.csv"), mode='w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file, delimiter=';')

        # Write the list to the CSV file
        csv_writer.writerows(rows)


def compute_cross_validation_average(output_path,exercise):
    filepaths=[f"{output_path}/result_{exercise}_fold_id_{idx}.csv" for idx in range(0,5)]
    dataframes=[]
    for file_path in filepaths:
        df = pd.read_csv(file_path,sep=";")
        first_column=df["Exercise"]
        df=df.drop("Exercise", axis=1)
        dataframes.append(df)
    result=pd.concat([dataframes[0], dataframes[1], dataframes[2],dataframes[3],dataframes[4]]).groupby(level=0).mean()
    result.insert(0, "Exercise", first_column)
    #result.to_csv(f"../nets/single_frame_crossvalidation_repetitions/result_{exercise}_crossvalidation.csv", index=False,sep=";")
    result.to_csv(os.path.join(output_path, f"result_{exercise}_crossvalidation.csv"), index=False, sep=";")


def main():
    size=505
    augment=True
    for exercise in ["squat","lunge","plank"]:
        for crossvalidation_id in [0,1,2,3,4]:
            params={
                #dataset definition
                "exercise": exercise,
                "use_rep_dataset": True,
                "rep_size": size,

                #skeleton definition
                "visible_joints":[0, 4, 6, 10, 14, 17],
                "normalize_skeleton":True,
                "augmentation":augment,

                #net definition
                "linear_embed_dimensionality": 32,
                "vit_attention_heads":8,
                "vit_hidden_dimensionality":512,
                "vit_encoder_layers":3,

                #training definition
                "batch_size":64,
                "epochs":200,
                "learning_rate":0.001,
                "loss_function": "L1" #L1 or L2
            }

            visible_joints = params["visible_joints"]
            exercise = params["exercise"]
            use_rep_dataset=params["use_rep_dataset"]
            normalize_skeleton=params["normalize_skeleton"]
            augment_skeleton=params["augmentation"]

            embed_dimensionality=params["linear_embed_dimensionality"]
            attention_heads=params["vit_attention_heads"]
            hidden_dimensionality=params["vit_hidden_dimensionality"]
            num_encoder_layers=params["vit_encoder_layers"]

            batch_size=params["batch_size"]
            num_epochs=params["epochs"]
            learning_rate=params["learning_rate"]
            loss_function=params["loss_function"]


            rep_size=params["rep_size"]
            datapath = f"../data/pkl/optitrack/single_repetitions/{rep_size}_frames"
            train_dataset = data_utils.SkeletonRepetitionMaskingDatasetSingleFrame(dataset_path=datapath, visible_joints=visible_joints, exercise=exercise, set="train",
                                                                                   normalize=normalize_skeleton, augmentation=augment_skeleton,crossValidation=crossvalidation_id)
            val_dataset = data_utils.SkeletonRepetitionMaskingDatasetSingleFrame(dataset_path=datapath, visible_joints=visible_joints, exercise=exercise, set="val",
                                                                                 normalize=normalize_skeleton,crossValidation=crossvalidation_id)
            num_joints = train_dataset.getJointsNumber()
            params["num_joints"]=num_joints

            model=MaskPoseVIT(num_joints=num_joints,embed_dim=embed_dimensionality,num_heads=attention_heads,hidden_dim=hidden_dimensionality,num_layers=num_encoder_layers,input_visible_joints=visible_joints).double()
            # Count the number of parameters
            num_params = sum(p.numel() for p in model.parameters())
            #print(f"Training on {exercise} reconstruction with {visible_joints} visible joints")
            print(f"Training on {exercise} reconstruction with {visible_joints} visible joints, fold {crossvalidation_id}")
            print("Training on:")
            print(params)
            print(f"Number of parameters in the model: {num_params}")
            model.to(device)
            if loss_function=="L1":
                loss_function=torch.nn.L1Loss()
            else:
                loss_function=torch.nn.MSELoss()
            #train_loader,val_loader=data_utils.getTrainValDataloader(dataset,validation_split_ratio=0.2,batch_size=32)
            train_loader=data_utils.getDataloaderFromDataset(train_dataset,set="train",batch_size=batch_size)
            val_loader=data_utils.getDataloaderFromDataset(val_dataset,set="val",batch_size=batch_size)
            optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
            model,results=model.training_loop(train_loader=train_loader, val_loader=val_loader, criterion=loss_function,optimizer=optimizer,num_epochs=num_epochs)
            #print("Done")
            # Get the current date and time
            current_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
            output_dir = current_time + " " + exercise + " fold "+str(crossvalidation_id)+" with joints " + '_'.join(map(str, visible_joints))
            output_root = os.path.join("../nets", "single_frame_crossvalidation_repetitions")
            output_path = os.path.join(output_root, output_dir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            torch.save(model.state_dict(), f"{output_path}/model.pt")
            with open(f'{output_path}/parameters.json', 'w') as json_file:
                json.dump(params, json_file, indent=4)
            #with open(f'{output_path}/results.json', 'w') as json_file:
            #    json.dump(results, json_file, indent=4)
            masked_joints = [item for item in range(21) if item not in visible_joints]
            evaluateModel(model, exercise, crossvalidation_id, masked_joints,output_root)
        compute_cross_validation_average(output_root,exercise)


if __name__ == '__main__':
    main()