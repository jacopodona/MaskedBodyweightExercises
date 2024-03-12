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


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    for exercise in ["plank"]:
        for size in [505]:
            for augment in [True]:
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
                    "vit_encoder_layers":4,

                    #training definition
                    "batch_size":128,
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


                if use_rep_dataset:
                    rep_size=params["rep_size"]
                    datapath = f"../data/pkl/optitrack/single_repetitions/{rep_size}_frames"
                    train_dataset = data_utils.SkeletonRepetitionMaskingDatasetSingleFrame(dataset_path=datapath, visible_joints=visible_joints, exercise=exercise, set="train",
                                                                                           normalize=normalize_skeleton, augmentation=augment_skeleton)
                    val_dataset = data_utils.SkeletonRepetitionMaskingDatasetSingleFrame(dataset_path=datapath, visible_joints=visible_joints, exercise=exercise, set="val",
                                                                                         normalize=normalize_skeleton)
                else:
                    datapath = "../data/pkl/optitrack/baseline_skeleton"
                    train_dataset = data_utils.SkeletonMaskingDataset(dataset_path=datapath, visible_joints=visible_joints,
                                                                      exercise=exercise, set="train",normalize=normalize_skeleton,augmentation=augment_skeleton)
                    val_dataset = data_utils.SkeletonMaskingDataset(dataset_path=datapath, visible_joints=visible_joints,
                                                                    exercise=exercise, set="val",normalize=normalize_skeleton)
                num_joints = train_dataset.getJointsNumber()
                params["num_joints"]=num_joints

                model=MaskPoseVIT(num_joints=num_joints,embed_dim=embed_dimensionality,num_heads=attention_heads,hidden_dim=hidden_dimensionality,num_layers=num_encoder_layers,input_visible_joints=visible_joints).double()
                # Count the number of parameters
                num_params = sum(p.numel() for p in model.parameters())
                #print(f"Training on {exercise} reconstruction with {visible_joints} visible joints")
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
                output_dir=current_time+" "+exercise+" with joints "+'_'.join(map(str, visible_joints))
                output_path=os.path.join("../nets","single_frame", output_dir)
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                torch.save(model.state_dict(), f"{output_path}/model.pt")
                with open(f'{output_path}/parameters.json', 'w') as json_file:
                    json.dump(params, json_file,indent=4)
                #with open(f'{output_path}/results.json', 'w') as json_file:
                #    json.dump(results, json_file, indent=4)