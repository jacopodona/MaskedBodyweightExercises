import json
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from model import SpatioTemporalMaskPoseVIT,SpatialMaskPoseVITWithTemporalDependency
import data_utils
import datetime
import plot_utils
import skeleton_utils

device = "cuda" if torch.cuda.is_available() else "cpu"

class TemporalSmoothnessLoss(nn.Module):
    def __init__(self,type):
        super(TemporalSmoothnessLoss, self).__init__()
        self.type=type

    def forward(self, predictions, ground_truth=None):
        if self.type==1:
            # Calculate temporal gradients for predictions
            pred_temporal_gradients = torch.abs(predictions[:, 1:, :, :] - predictions[:, :-1, :, :])

            # Calculate temporal gradients for ground truth
            gt_temporal_gradients = torch.abs(ground_truth[:, 1:, :, :] - ground_truth[:, :-1, :, :])

            # Calculate the smoothness loss based on the difference between predictions and ground truth
            smoothness_loss = torch.mean(torch.abs(pred_temporal_gradients - gt_temporal_gradients))

            return smoothness_loss
        elif self.type==2:
            # Calculate temporal gradients
            temporal_gradients = torch.abs(predictions[:, 1:, :, :] - predictions[:, :-1, :, :])
            # Sum over all dimensions and frames
            smoothness_loss = torch.mean(temporal_gradients)
            return smoothness_loss


class TemporalL1(torch.nn.Module):
    def __init__(self,beta,type):
        super(TemporalL1, self).__init__()

        # Weights for the loss terms
        assert beta>=0 and beta<=1
        self.l1_weight = beta
        self.smoothness_weight = 1-beta
        self.smoothness_loss_type=type
        self.smoothness_loss=TemporalSmoothnessLoss(type)
        self.l1_loss=nn.L1Loss()

    def forward(self, predictions, ground_truth):
        # Calculate the L1 loss
        l1_term =self.l1_loss(predictions, ground_truth)

        # Calculate the smoothness loss
        if self.smoothness_loss_type==1 or self.smoothness_loss_type==3:
            smoothness_term = self.smoothness_loss(predictions,ground_truth)
        elif self.smoothness_loss_type==2:
            smoothness_term = self.smoothness_loss(predictions)
        #smoothness_term = smoothness_term / smoothness_term.mean()  # Normalize by mean or any other suitable values

        # Combine the terms
        combined_loss = self.l1_weight * l1_term + self.smoothness_weight * smoothness_term

        return combined_loss

if __name__ == '__main__':
    for exercise in ["squat"]:
        for size in [505]:
            for loss in ["TEMPORAL_L1"]:
                for window in [4]:
                    for weight in [0.7]:
                        params={
                            #dataset definition
                            "exercise": exercise,
                            "rep_size": size,

                            #skeleton definition
                            "visible_joints":[0, 4, 6, 10, 14, 17],
                            "normalize_skeleton":True,
                            "augmentation":True,

                            #net definition
                            "linear_embed_dimensionality": 32,
                            "vit_attention_heads":8,
                            "vit_hidden_dimensionality":512,
                            "vit_encoder_layers":4,
                            "window_size":window,
                            "decoder_hidden_dim":32,
                            "decoder_num_layers":2,

                            #training definition
                            "batch_size":64,
                            "epochs":200,
                            "learning_rate":0.001,
                            "loss_function": loss, #L1 or TEMPORAL_L1 or L2
                            "temporal_l1_type":2, #type 1 is learning gradients from gt and apply to pred, type 2 is smoothing gradient on pred, type 3 is combination
                            "weight_l1":weight
                        }

                        visible_joints = params["visible_joints"]
                        exercise = params["exercise"]
                        normalize_skeleton=params["normalize_skeleton"]
                        augment_skeleton=params["augmentation"]
                        rep_size = params["rep_size"]
                        window_size=params["window_size"]

                        embed_dimensionality=params["linear_embed_dimensionality"]
                        attention_heads=params["vit_attention_heads"]
                        hidden_dimensionality=params["vit_hidden_dimensionality"]
                        num_encoder_layers=params["vit_encoder_layers"]
                        decoder_hidden_dim=params["decoder_hidden_dim"]
                        num_decoder_layers=params["decoder_num_layers"]

                        batch_size=params["batch_size"]
                        num_epochs=params["epochs"]
                        learning_rate=params["learning_rate"]
                        loss_function=params["loss_function"]
                        temporal_l1_type=params["temporal_l1_type"]
                        weight_l1_function=params["weight_l1"]

                        datapath = f"data/pkl/optitrack/single_repetitions/{rep_size}_frames"
                        train_dataset = data_utils.SequenceMaskingRepetitionDataset(dataset_path=datapath, visible_joints=visible_joints, exercise=exercise,window_size=window_size, set="train",
                                                                                               normalize=normalize_skeleton, augmentation=augment_skeleton)
                        prova_true,prova_masked=train_dataset.__getitem__(0)
                        val_dataset = data_utils.SequenceMaskingRepetitionDataset(dataset_path=datapath, visible_joints=visible_joints, exercise=exercise,window_size=window_size, set="val",
                                                                                             normalize=normalize_skeleton)

                        num_joints = train_dataset.getJointsNumber()
                        params["num_joints"]=num_joints

                        model=SpatioTemporalMaskPoseVIT(num_joints=num_joints,num_frames=window_size,embed_dim=embed_dimensionality,num_heads=attention_heads,hidden_dim=hidden_dimensionality,num_layers=num_encoder_layers,input_visible_joints=visible_joints,
                                                        decoder_hidden_dim=decoder_hidden_dim,decoder_num_layers=num_decoder_layers).double()
                        #model=SpatialMaskPoseVITWithTemporalDependency(num_joints=num_joints,num_frames=window_size,embed_dim=embed_dimensionality,num_heads=attention_heads,hidden_dim=hidden_dimensionality,num_layers=num_encoder_layers,input_visible_joints=visible_joints,
                        #                                decoder_hidden_dim=decoder_hidden_dim,decoder_num_layers=num_decoder_layers).double()
                        # Count the number of parameters
                        num_params = sum(p.numel() for p in model.parameters())
                        #print(f"Training on {exercise} reconstruction with {visible_joints} visible joints")
                        print("Training on:")
                        print(params)
                        print(f"Number of parameters in the model: {num_params}")
                        model.to(device)
                        if loss_function=="L1":
                            loss_function=nn.L1Loss()
                        elif loss_function=="TEMPORAL_L1":
                            loss_function=TemporalL1(beta=weight_l1_function,type=temporal_l1_type)
                        elif loss_function=="L2":
                            loss_function=nn.MSELoss()
                        #train_loader,val_loader=data_utils.getTrainValDataloader(dataset,validation_split_ratio=0.2,batch_size=32)
                        train_loader=data_utils.getDataloaderFromDataset(train_dataset,set="train",batch_size=batch_size)
                        val_loader=data_utils.getDataloaderFromDataset(val_dataset,set="val",batch_size=batch_size)
                        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
                        model,results=model.training_loop(train_loader=train_loader, val_loader=val_loader, criterion=loss_function,optimizer=optimizer,num_epochs=num_epochs,plot=False)
                        #print("Done")
                        # Get the current date and time
                        current_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
                        output_dir=current_time+" "+exercise+" with joints "+'_'.join(map(str, visible_joints))
                        #output_dir+=f"setup 4 rep_size {rep_size} window_size {window}"
                        #output_dir += f" {loss_function.l1_weight} L1 {round(loss_function.smoothness_weight, 1)} smoothness"
                        #output_dir += f"window_size {window}"
                        output_dir += f"rep_size {size}"
                        output_path=os.path.join(
                            "../nets/temporal_window/", output_dir)
                        if not os.path.exists(output_path):
                            os.mkdir(output_path)
                        torch.save(model.state_dict(), f"{output_path}/model.pt")
                        with open(f'{output_path}/parameters.json', 'w') as json_file:
                            json.dump(params, json_file,indent=4)
                        #with open(f'{output_path}/results.json', 'w') as json_file:
                        #    json.dump(results, json_file, indent=4)