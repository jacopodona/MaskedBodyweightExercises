from builtins import int
import plot_utils
import torch
import torch.nn as nn
import skeleton_utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class MaskPoseVIT(nn.Module):
    def __init__(self,num_joints,embed_dim,num_layers,num_heads,hidden_dim,input_visible_joints=None,dropout=0.1):
        """

        :param num_joints:  The number of 3D joints in the skeleton
        :param embed_dim: The dimension of the embedding vector (embed_dim must be divisible by num_heads)
        :param num_layers:  The number of transformer encoder layers
        :param num_heads: The number of attention heads
        :param hidden_dim:  The transformer latent representation size
        :param dropout: The dropout probability.
        """
        super(MaskPoseVIT,self).__init__()
        self.num_joints=num_joints
        self.embed_dim=embed_dim
        self.visible_joints_index=input_visible_joints
        self.masked_joints_index=[item for item in range(num_joints) if item not in self.visible_joints_index]
        self.predicted_joints=len(self.masked_joints_index)
        # Linear projection for joints
        #self.joint_embeddings = nn.Linear(in_features=3,out_features=embed_dim)
        self.joint_embeddings = nn.Sequential(nn.Linear(in_features=3,out_features=embed_dim),nn.ReLU())
        # Define a linear layer
        #self.joint_embeddings = nn.Linear(num_joints * 3, num_joints * embed_dim)

        # Spatial Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_joints, embed_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )
        self.reconstruction_head=nn.Linear(in_features=num_joints * embed_dim,out_features=self.predicted_joints*3)

    def forward(self,x):
        batch_size = x.size(0)
        input_joints = x.size(1)
        joint_coordinates = x.size(2)
        reshaped_input = x.view(-1, x.size(-1))
        joint_embeddings = self.joint_embeddings(reshaped_input)
        skeleton_representation = joint_embeddings.view(batch_size, input_joints, -1)  # Reshape to (batch_size, num_joints, embed_dim)

        # Add spatial positional embeddings
        net_input = skeleton_representation + self.positional_embeddings

        # Pass through the transformer encoder
        latent_representation = self.transformer_encoder(net_input)

        reshaped_output = latent_representation.view(batch_size, -1)
        # Apply global average pooling
        #x = x.mean(1)
        if torch.isnan(reshaped_output).any() or torch.isinf(reshaped_output).any():
            print("Reshaped output from VIT encoder contains NaN or inf values!")

        #Reconstruct
        predicted_joints = self.reconstruction_head(reshaped_output)
        predicted_joints = predicted_joints.view(batch_size, self.predicted_joints, 3)
        if torch.isnan(reshaped_output).any() or torch.isinf(reshaped_output).any():
            print("Predicted coordinates contains NaN or inf values!")

        output=x.clone()
        # Replace the masked entries of the input tensor with the predicted values
        for sample_index in range(batch_size):
            #x[i, self.masked_joints_index[i]] = predicted_joints[i,:len(index_lists[i])]
            output[sample_index,self.masked_joints_index] = predicted_joints[sample_index]
        return output

    def training_loop(self, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        best_model = None
        best_results = None
        best_loss = float("inf")
        for epoch in tqdm(range(num_epochs)):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            # Training loop
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate the loss
                loss.backward()  # Backpropagate
                optimizer.step()  # Update the model's parameters
                running_loss += loss.item()
                for param in self.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print("Model parameters contain NaN or inf values!")
                        break

            # Calculate the average training loss for the epoch
            average_train_loss = running_loss / len(train_loader)

            # Validation loop
            self.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_error_list = []
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, targets)
                    reconstruction_error = skeleton_utils.computeBatchReconstructionError(outputs, targets)
                    val_error_list.append(reconstruction_error)
                    val_loss += loss.item()
            val_set_errors = np.vstack(val_error_list)
            per_joint_error = np.mean(val_set_errors, axis=0)

            # Calculate the average validation loss for the epoch
            average_val_loss = val_loss / len(val_loader)
            results = skeleton_utils.printPerJointError(per_joint_error)
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                best_model = self
                best_results = results

            print(
                f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
            print("Joint errors" + str(results))
            print("=" * 80)
        print('Training completed.')
        return best_model, best_results

    def inference(self, masked_sample):
        """
        Sends sample to torch tensor in correct format for pt model and returns output
        :param masked_sample: Masked Skeleton sample (ndarray)
        :return:
        """
        input = torch.tensor(masked_sample).double()
        input = input.unsqueeze(0)
        input = input.to(device)
        output = self(input)
        output = output.squeeze(0)
        output_array = output.detach().cpu().numpy()
        return output_array

class SpatioTemporalMaskPoseVIT(nn.Module):
    def __init__(self,num_joints,num_frames,embed_dim,num_layers,num_heads,hidden_dim,decoder_hidden_dim,decoder_num_layers,input_visible_joints=None,dropout=0.1):
        """

        :param num_joints:  The number of 3D joints in the skeleton
        :param num_frames:  The number of frames in input temporal window
        :param embed_dim: The dimension of the embedding vector (embed_dim must be divisible by num_heads)
        :param num_layers:  The number of transformer encoder layers
        :param num_heads: The number of attention heads
        :param hidden_dim:  The transformer latent representation size
        :param dropout: The dropout probability.
        """
        super(SpatioTemporalMaskPoseVIT,self).__init__()
        self.num_joints=num_joints
        self.window_size=num_frames
        self.embed_dim=embed_dim
        self.visible_joints_index=input_visible_joints
        self.rnn_hidden_dim=decoder_hidden_dim
        self.rnn_layers=decoder_num_layers
        self.masked_joints_index=[item for item in range(num_joints) if item not in self.visible_joints_index]
        self.predicted_joints=len(self.masked_joints_index)
        # Linear projection for joints
        #self.joint_embeddings = nn.Linear(in_features=3,out_features=embed_dim)
        self.joint_embeddings = nn.Sequential(nn.Linear(in_features=3, out_features=embed_dim), nn.LeakyReLU())
        # Define a linear layer
        #self.joint_embeddings = nn.Linear(num_joints * 3, num_joints * embed_dim)

        # Spatial Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, 1, num_joints, embed_dim))

        # For temporal embeddings you need to add parameter as long as the number of frames in the temporal window
        # Temporal Positional Embeddings
        self.temporal_embeddings = nn.Parameter(torch.randn(1, num_frames, 1, embed_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )
        #SETUP 1 (ok)
        self.reconstruction_head = nn.Linear(in_features=self.window_size * self.num_joints * embed_dim,out_features=self.window_size*self.predicted_joints*3)
        #SETUP 2 (ok)
        #self.reconstruction_head=nn.Sequential(nn.Linear(in_features=self.window_size*self.num_joints*embed_dim,out_features=512),nn.Linear(in_features=512,out_features=self.window_size*self.predicted_joints*3))

        #SETUP 3 (no)
        #self.reconstruction_head = nn.Linear(in_features=embed_dim, out_features=3)

        #SETUP 4 (ok)
        #self.reconstruction_head=TemporalReconstructionHead(input_size=self.embed_dim,input_joints=self.num_joints,output_joints=self.predicted_joints, hidden_size=self.rnn_hidden_dim, num_layers=self.rnn_layers, output_size=3)

    def forward(self,x):
        batch_size = x.size(0)

        #Prepare joint embeddings for transformer
        #reshaped_input = x.view(batch_size,-1, x.size(-1)) #Flatten to [batch_size,temporal_window*num_joints,3]
        joint_embeddings = self.joint_embeddings(x) #Get embeddings for each joint

        # Add spatial positional embeddings
        net_input = joint_embeddings + self.positional_embeddings + self.temporal_embeddings

        #Reshape to (batchsize,sequence_frame*sequence_joint,embed_dim)
        net_input = net_input.view(batch_size, self.window_size * self.num_joints, -1)
        # Pass through the transformer encoder
        latent_representation = self.transformer_encoder(net_input)

        #Reshape
        reshaped_output = latent_representation.view(batch_size, -1) #SETUP 1 E 2
        #reshaped_output = latent_representation.view(batch_size, self.window_size,self.num_joints,-1) #SETUP 3
        #reshaped_output = latent_representation.view(batch_size, self.window_size,self.num_joints,-1) #SETUP 4

        #Reconstruct
        predicted_joints = self.reconstruction_head(reshaped_output)
        predicted_joints = predicted_joints.view(batch_size,self.window_size, self.predicted_joints, 3) #SETUP 1 2 e 4
        #predicted_joints = predicted_joints.view(batch_size, self.window_size, self.num_joints, 3)#SETUP 3

        output=x.clone()
        # Replace the masked entries of the input tensor with the predicted values
        for sample_index in range(batch_size):
            for frame_idx in range(self.window_size):
                #x[i, self.masked_joints_index[i]] = predicted_joints[i,:len(index_lists[i])]
                output[sample_index,frame_idx,self.masked_joints_index] = predicted_joints[sample_index,frame_idx]
                #output[sample_index, frame_idx,self.masked_joints_index] = predicted_joints[sample_index, frame_idx,self.masked_joints_index] #setup 3
        return output

    def training_loop(self, train_loader, val_loader, criterion, optimizer, num_epochs,plot=False):
        train_losses=[]
        val_losses=[]
        best_model = None
        best_results = None
        best_loss = float("inf")
        best_epoch=0
        for epoch in tqdm(range(num_epochs)):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            # Training loop
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate the MSE loss
                loss.backward()  # Backpropagate
                optimizer.step()  # Update the model's parameters
                running_loss += loss.item()
                for param in self.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print("Model parameters contain NaN or inf values!")
                        break

            # Calculate the average training loss for the epoch
            average_train_loss = running_loss / len(train_loader)

            # Validation loop
            self.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_error_list = []
            with torch.no_grad():
                for batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    outputs = self(val_inputs)
                    loss = criterion(outputs, val_targets)
                    # reconstruction_error = skeleton_utils.computeBatchReconstructionError(outputs, targets)
                    # val_error_list.append(reconstruction_error)
                    val_loss += loss.item()
            # val_set_errors = np.vstack(val_error_list)
            # per_joint_error = np.mean(val_set_errors, axis=0)

            # Calculate the average validation loss for the epoch
            average_val_loss = val_loss / len(val_loader)
            # results = skeleton_utils.printPerJointError(per_joint_error)
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                best_model = self
                best_epoch=epoch
                # best_results = results

            print(
                f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
            train_losses.append(average_train_loss)
            val_losses.append(average_val_loss)
            print("=" * 80)
            if(epoch==num_epochs-1):
                print("Debug point")


        if plot:
            x_values=list(range(1, num_epochs+1))
            plt.plot(x_values, train_losses, label='Train Loss',c="blue")

            # Plotting the second line
            plt.plot(x_values, val_losses, label='Val Loss',c="orange")

            # Adding labels and a legend
            plt.xlabel('Epochs')
            plt.ylabel('Loss Value')
            plt.title('Training Loss')
            plt.legend()
            plt.grid(axis='y', alpha=0.7)

            # Display the plot
            plt.show()
            plt.clf()
        print('Training completed.')
        print(f'\tBest model at epoch {best_epoch}')
        return best_model, best_results

    def test_architecture(self,):
        pass

class SpatialMaskPoseVITWithTemporalDependency(nn.Module):
    def __init__(self,num_joints,num_frames,embed_dim,num_layers,num_heads,hidden_dim,decoder_hidden_dim,decoder_num_layers,input_visible_joints=None,dropout=0.1):
        """

        :param num_joints:  The number of 3D joints in the skeleton
        :param num_frames:  The number of frames in input temporal window
        :param embed_dim: The dimension of the embedding vector (embed_dim must be divisible by num_heads)
        :param num_layers:  The number of transformer encoder layers
        :param num_heads: The number of attention heads
        :param hidden_dim:  The transformer latent representation size
        :param dropout: The dropout probability.
        """
        super(SpatialMaskPoseVITWithTemporalDependency,self).__init__()
        self.num_joints=num_joints
        self.window_size=num_frames
        self.embed_dim=embed_dim
        self.visible_joints_index=input_visible_joints
        self.rnn_hidden_dim=decoder_hidden_dim
        self.rnn_layers=decoder_num_layers
        self.masked_joints_index=[item for item in range(num_joints) if item not in self.visible_joints_index]
        self.predicted_joints=len(self.masked_joints_index)
        # Linear projection for joints
        self.joint_embeddings = nn.Sequential(nn.Linear(in_features=3,out_features=embed_dim),nn.LeakyReLU())
        # Define a linear layer
        #self.joint_embeddings = nn.Linear(num_joints * 3, num_joints * embed_dim)

        # Spatial Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, 1, num_joints, embed_dim))

        # For temporal embeddings you need to add parameter as long as the number of frames in the temporal window
        # Temporal Positional Embeddings
        self.temporal_embeddings = nn.Parameter(torch.randn(1, num_frames, 1, embed_dim))

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )

        self.reconstruction_head=SpatialReconstructionHeadWithTemporalDependency(num_joints=num_joints,embed_dim=embed_dim,predicted_joints=self.predicted_joints,
                                                                                 hidden_size=64, num_layers=2)



    def forward(self,x):
        batch_size = x.size(0)

        #Prepare joint embeddings for transformer
        #reshaped_input = x.view(batch_size,-1, x.size(-1)) #Flatten to [batch_size,temporal_window*num_joints,3]
        joint_embeddings = self.joint_embeddings(x) #Get embeddings for each joint

        # Add spatial positional embeddings
        net_input = joint_embeddings + self.positional_embeddings + self.temporal_embeddings

        #Reshape to (batchsize,sequence_frame*sequence_joint,embed_dim)
        net_input = net_input.view(batch_size, self.window_size * self.num_joints, -1)
        # Pass through the transformer encoder
        latent_representation = self.transformer_encoder(net_input)

        #Reshape
        reshaped_output = latent_representation.view(batch_size, self.window_size,self.num_joints, -1)

        #Reconstruct
        predicted_joints = self.reconstruction_head(reshaped_output)
        predicted_joints = predicted_joints.view(batch_size,self.window_size, self.predicted_joints, 3)

        output=x.clone()
        # Replace the masked entries of the input tensor with the predicted values
        for sample_index in range(batch_size):
            for frame_idx in range(self.window_size):
                #x[i, self.masked_joints_index[i]] = predicted_joints[i,:len(index_lists[i])]
                output[sample_index,frame_idx,self.masked_joints_index] = predicted_joints[sample_index,frame_idx]
                #output[sample_index, frame_idx,self.masked_joints_index] = predicted_joints[sample_index, frame_idx,self.masked_joints_index] #setup 3
        return output

    def training_loop(self, train_loader, val_loader, criterion, optimizer, num_epochs):
        best_model = None
        best_results = None
        best_loss = float("inf")
        best_epoch=0
        for epoch in tqdm(range(num_epochs)):
            self.train()  # Set the model to training mode
            running_loss = 0.0

            # Training loop
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Calculate the MSE loss
                loss.backward()  # Backpropagate
                optimizer.step()  # Update the model's parameters
                running_loss += loss.item()

            # Calculate the average training loss for the epoch
            average_train_loss = running_loss / len(train_loader)

            # Validation loop
            self.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_error_list = []
            with torch.no_grad():
                for batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    outputs = self(val_inputs)
                    loss = criterion(outputs, val_targets)
                    # reconstruction_error = skeleton_utils.computeBatchReconstructionError(outputs, targets)
                    # val_error_list.append(reconstruction_error)
                    val_loss += loss.item()
            # val_set_errors = np.vstack(val_error_list)
            # per_joint_error = np.mean(val_set_errors, axis=0)

            # Calculate the average validation loss for the epoch
            average_val_loss = val_loss / len(val_loader)
            # results = skeleton_utils.printPerJointError(per_joint_error)
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                best_model = self
                best_epoch=epoch
                # best_results = results

            print(
                f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')
            # print("Joint errors" + str(results))
            print("=" * 80)
            if(epoch==num_epochs-1):
                print("Debug point")

        print('Training completed.')
        print(f'\tBest model at epoch {best_epoch}')
        return best_model, best_results


class TemporalReconstructionHead(nn.Module):
    def __init__(self, input_size,input_joints,output_joints, hidden_size, num_layers, output_size=3):
        super(TemporalReconstructionHead, self).__init__()

        self.input_joints=input_joints
        self.output_joints=output_joints

        self.rnn = nn.GRU(input_size*self.input_joints, hidden_size, num_layers,dropout=0.1,batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size*self.output_joints)

    def forward(self, x):
        # x shape: [batch_size, num_frames, num_joints, embed_dimensionality]

        # Reshape input for LSTM
        batch_size, num_frames, num_joints, embed_dim = x.size()
        x = x.view(batch_size, num_frames, -1)

        # LSTM layer
        out, _ = self.rnn(x)

        # Fully connected layer
        out = self.fc(out)

        # Reshape back to the original shape
        out = out.view(batch_size, num_frames, self.output_joints, -1)

        return out

class SpatialReconstructionHeadWithTemporalDependency(nn.Module):
    def __init__(self, num_joints,embed_dim,predicted_joints, hidden_size, num_layers):
        super(SpatialReconstructionHeadWithTemporalDependency, self).__init__()

        self.reconstruction_head = nn.Linear(in_features=num_joints * embed_dim, out_features=predicted_joints * 3)

        self.temporal_dependency_layer = nn.GRU(input_size=predicted_joints * 3, hidden_size=predicted_joints * 3, num_layers=num_layers,
                                                dropout=0.1, batch_first=True)

    def forward(self, x):
        # x shape: [batch_size, num_frames, num_joints, embed_dimensionality]

        # Reshape input for LSTM
        batch_size, num_frames, num_joints, embed_dim = x.size()
        x = x.view(batch_size, num_frames, -1)

        # LSTM layer
        out= self.reconstruction_head(x)

        # Fully connected layer
        out, _ = self.temporal_dependency_layer(out)

        return out

class MaskFullPoseVIT(nn.Module): #Old model
    def __init__(self,num_joints,embed_dim,num_layers,num_heads,hidden_dim,dropout=0.1):
        """

        :param num_joints:  The number of 3D joints in the skeleton
        :param embed_dim: The dimension of the embedding vector (embed_dim must be divisible by num_heads)
        :param num_layers:  The number of transformer encoder layers
        :param num_heads: The number of attention heads
        :param hidden_dim:  The transformer latent representation size
        :param dropout: The dropout probability.
        """
        super(MaskFullPoseVIT,self).__init__()
        self.num_joints=num_joints
        self.embed_dim=embed_dim
        # Linear projection for joints
        self.joint_embeddings = nn.Linear(in_features=3,out_features=embed_dim)
        # Define a linear layer
        #self.joint_embeddings = nn.Linear(num_joints * 3, num_joints * embed_dim)

        # Spatial Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_joints, embed_dim))

        # For temporal embeddings you need to add parameter as long as the number of frames in the temporal window

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers
        )
        self.reconstruction_head=nn.Linear(in_features=num_joints * embed_dim,out_features=num_joints*3)

    def forward(self,x):
        batch_size = x.size(0)
        input_joints = x.size(1)
        joint_coordinates = x.size(2)
        reshaped_input = x.view(-1, x.size(-1))
        joint_embeddings = self.joint_embeddings(reshaped_input)
        skeleton_representation = joint_embeddings.view(batch_size, input_joints, -1)  # Reshape to (batch_size, num_joints, embed_dim)

        # Add spatial positional embeddings
        net_input = skeleton_representation + self.positional_embeddings

        # Pass through the transformer encoder
        latent_representation = self.transformer_encoder(net_input)

        reshaped_output = latent_representation.view(batch_size, -1)
        # Apply global average pooling
        #x = x.mean(1)
        if torch.isnan(reshaped_output).any() or torch.isinf(reshaped_output).any():
            print("Reshaped output from VIT encoder contains NaN or inf values!")

        #Reconstruct
        predicted_coordinates = self.reconstruction_head(reshaped_output)
        predicted_coordinates = predicted_coordinates.view(batch_size, self.num_joints, 3)
        if torch.isnan(reshaped_output).any() or torch.isinf(reshaped_output).any():
            print("Predicted coordinates contains NaN or inf values!")
        return predicted_coordinates