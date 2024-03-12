import os
import torch
import pickle as pkl
from torch.utils.data import Dataset,random_split,DataLoader
import json
import copy
import numpy as np
from itertools import chain
import skeleton_utils
import plot_utils

def loadAnnotationsJSONS(dataset_path):
    samples=[]
    subjects=os.listdir(dataset_path)
    for s in subjects:
        file_path=os.path.join(dataset_path,s,"joints3d_25/warmup_7.json")
        with open(file_path, 'r') as file:
            data = json.load(file)
        joints=data["joints3d_25"]
        samples.append(joints)
    return samples

def loadAllAnnotationsPickle(dataset_path,exercise_to_get="squat"):
    samples = []
    exercises = os.listdir(dataset_path)
    for s in exercises:
        if exercise_to_get in s:
            file_path = os.path.join(dataset_path, s)
            with open(file_path, 'rb') as file:
                data = pkl.load(file)
            #joints = data["joints3d_25"]
            samples.append(data)
    return samples

def loadAnnotationsPickle(dataset_path,exercise_to_get="squat",set="train"):
    samples = []
    exercise_to_load=[]
    if exercise_to_get == "squat" and set=="train":
        exercise_to_load=["squat_good.pkl","squat_feet.pkl","squat_back_forward.pkl","squat_knees.pkl","squat_too_deep.pkl","squat_too_high.pkl"]
    elif exercise_to_get =="squat" and set=="val":
        exercise_to_load = ["squat_bad.pkl"]
    elif exercise_to_get == "lunge" and set=="train":
        exercise_to_load=["lunge_good.pkl","lunge_high.pkl","lunge_long.pkl","lunge_back_bent.pkl","lunge_back_behind.pkl"]
    elif exercise_to_get =="lunge" and set=="val":
        exercise_to_load = ["lunge_no_equilibrium.pkl"]
    elif exercise_to_get == "plank" and set=="train":
        exercise_to_load=["plank_good.pkl","plank_butt_high.pkl","plank_butt_low.pkl","plank_head_down.pkl","plank_head_up.pkl","plank_shoulders_forward.pkl"]
    elif exercise_to_get =="plank" and set=="val":
        exercise_to_load = ["plank_bad.pkl"]
    for s in exercise_to_load:
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        samples.append(data)
    return samples

def loadRepetitionAnnotationsPickle(dataset_path,exercise_to_get="squat",set="train"):
    samples = []
    exercise_to_load=[]
    variants = os.listdir(dataset_path)
    train_repetitions = ["rep_0.pkl","rep_1.pkl","rep_2.pkl","rep_3.pkl","rep_5.pkl","rep_6.pkl","rep_8.pkl"]
    val_repetitions = ["rep_4.pkl","rep_7.pkl"]
    if (exercise_to_get == "squat" or exercise_to_get == "combined") and set=="train":
        for variant in variants:
            if "squat" in variant:
                exercise_to_load+=[os.path.join(variant, filename) for filename in train_repetitions]
    if (exercise_to_get == "squat" or exercise_to_get == "combined") and set=="val":
        for variant in variants:
            if "squat" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in val_repetitions]
    if (exercise_to_get == "lunge" or exercise_to_get == "combined") and set=="train":
        for variant in variants:
            if "lunge" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in train_repetitions]
    if (exercise_to_get == "lunge" or exercise_to_get == "combined") and set=="val":
        for variant in variants:
            if "lunge" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in val_repetitions]
    if (exercise_to_get == "plank" or exercise_to_get == "combined") and set=="train":
        for variant in variants:
            if "plank" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in train_repetitions]
    if (exercise_to_get == "plank" or exercise_to_get == "combined") and set=="val":
        for variant in variants:
            if "plank" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in val_repetitions]
    for s in exercise_to_load:
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        samples.append(data)
    return samples

def loadRepetitionAnnotationsPickleCrossValidation(dataset_path,exercise_to_get="squat",set="train",cross_validation_flag=0):
    samples = []
    exercise_to_load=[]
    variants = os.listdir(dataset_path)
    if cross_validation_flag==0:
        train_repetitions = ["rep_1.pkl","rep_2.pkl","rep_3.pkl","rep_4.pkl","rep_6.pkl","rep_7.pkl","rep_8.pkl"]
        val_repetitions = ["rep_0.pkl","rep_5.pkl"]
    elif cross_validation_flag==1:
        train_repetitions = ["rep_0.pkl", "rep_2.pkl", "rep_3.pkl", "rep_4.pkl", "rep_5.pkl", "rep_7.pkl",
                             "rep_8.pkl"]
        val_repetitions = ["rep_1.pkl", "rep_6.pkl"]
    elif cross_validation_flag == 2:
        train_repetitions = ["rep_0.pkl", "rep_1.pkl", "rep_3.pkl", "rep_4.pkl", "rep_5.pkl","rep_6.pkl",
                             "rep_8.pkl"]
        val_repetitions = ["rep_2.pkl", "rep_7.pkl"]
    elif cross_validation_flag == 3:
        train_repetitions = ["rep_0.pkl", "rep_1.pkl", "rep_2.pkl", "rep_4.pkl", "rep_5.pkl", "rep_6.pkl",
                             "rep_7.pkl"]
        val_repetitions = ["rep_3.pkl", "rep_8.pkl"]
    elif cross_validation_flag == 4:
        train_repetitions = ["rep_0.pkl", "rep_1.pkl", "rep_2.pkl", "rep_4.pkl", "rep_5.pkl", "rep_6.pkl",
                             "rep_7.pkl"]
        val_repetitions = ["rep_4.pkl", "rep_8.pkl"]
    if exercise_to_get == "squat" and set=="train":
        for variant in variants:
            if "squat" in variant:
                exercise_to_load+=[os.path.join(variant, filename) for filename in train_repetitions]
    elif exercise_to_get =="squat" and set=="val":
        for variant in variants:
            if "squat" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in val_repetitions]
    elif exercise_to_get == "lunge" and set=="train":
        for variant in variants:
            if "lunge" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in train_repetitions]
    elif exercise_to_get =="lunge" and set=="val":
        for variant in variants:
            if "lunge" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in val_repetitions]
    elif exercise_to_get == "plank" and set=="train":
        for variant in variants:
            if "plank" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in train_repetitions]
    elif exercise_to_get =="plank" and set=="val":
        for variant in variants:
            if "plank" in variant:
                exercise_to_load+= [os.path.join(variant, filename) for filename in val_repetitions]
    for s in exercise_to_load:
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        samples.append(data)
    return samples

def loadAnnotationsPickleCrossValidation(dataset_path, exercise_to_get, set, crossValidation_id):
    samples = []
    exercise_to_load = []
    variants = os.listdir(dataset_path)
    filtered_variants = [variant_name for variant_name in variants if exercise_to_get in variant_name]
    if set=="train":
        _=filtered_variants.pop(crossValidation_id)
        exercise_to_load=filtered_variants
        print(f"Training folds: {exercise_to_load}")
    elif set=="val":
        exercise_to_load=[filtered_variants.pop(crossValidation_id)]
        print(f"Validation folds: {exercise_to_load}")
    for s in exercise_to_load:
        file_path = os.path.join(dataset_path, s)
        with open(file_path, 'rb') as file:
            data = pkl.load(file)
        samples.append(data)
    return samples,exercise_to_load





def getTrainValDataloaderSplit(dataset,validation_split_ratio,batch_size = 4):

    # Calculate the number of samples for the validation set
    num_samples = len(dataset)
    num_validation = int(validation_split_ratio * num_samples)
    num_train = num_samples - num_validation

    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(dataset, [num_train, num_validation])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,collate_fn=collate_fn)
    return train_loader,val_loader

def getDataloaderFromDataset(dataset,batch_size=32,set="train"):
    if set=="train":
        return DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    else:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def collate_fn(data):
    # 'data' is a list of (input, target) tuples
    inputs, targets = zip(*data)
    stacked_inputs = torch.stack(inputs, dim=0)  # Stack inputs into a single tensor
    stacked_targets = torch.stack(targets, dim=0)   # Convert targets into a tensor

    return stacked_inputs, stacked_targets

def slice_sequence_into_windows(original_sequence,window_size):
    len_original_sequence = len(original_sequence)
    # Create a list to store the subvectors for each sequence
    subvectors = []

    num_subvectors = len_original_sequence // window_size

    # Iterate over the original sequence to extract subvectors
    for i in range(num_subvectors):
        start_index = i * window_size
        end_index = start_index + window_size
        subvector = original_sequence[start_index:end_index]
        subvectors.append(subvector)
    return subvectors

class SequenceMaskingDataset(Dataset):
    def __init__(self, dataset_path,masked_joints):
        #self.img_labels = pd.read_csv(annotations_file)
        self.sequences=loadAnnotationsJSONS(dataset_path)
        self.joints_to_mask=masked_joints
        #self.labels = img_dir
        #self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        clean_sequence=self.sequences[idx]
        masked_sequence=skeleton_utils.maskSequence(clean_sequence.copy(),self.joints_to_mask)
        return np.array(masked_sequence), np.array(clean_sequence)



class SequenceMaskingRepetitionDataset(Dataset):
    def __init__(self, dataset_path,visible_joints,hip_id=0,window_size=10,exercise="squat",set="train",normalize=False,augmentation=False):
        """

        :param dataset_path:
        :param visible_joints:
        :param hip_id: Used to know from which joint rescale all the other joints. In both formats hip joint id is 0
        """
        self.visible_joints=visible_joints
        self.sequences_raw=loadRepetitionAnnotationsPickle(dataset_path,exercise_to_get=exercise,set=set)
        self.window_size=window_size
        self.sequences=self.get_temporal_windows()
        self.masked_joints = [item for item in range(21) if item not in visible_joints]
        self.hip_id=hip_id
        self.normalize_skeleton = normalize
        self.augment_sequence=augmentation

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        clean_sequence=self.sequences[idx]
        for frame_idx in range(len(clean_sequence)):
            clean_sequence[frame_idx] = skeleton_utils.centerSkeletonAroundHip(clean_sequence[frame_idx], hip_id=self.hip_id)
        if self.normalize_skeleton:
            clean_sequence=skeleton_utils.normalize_sequence(clean_sequence)
        if self.augment_sequence:
            clean_sequence = skeleton_utils.augmentSequence(clean_sequence)
        modified_sequence=copy.copy(clean_sequence)
        for frame_idx in range(len(modified_sequence)):
            modified_skeleton=modified_sequence[frame_idx]
            masked_skeleton=skeleton_utils.maskSkeleton(skeleton=modified_skeleton,mask_joints_id=self.masked_joints)
            modified_sequence[frame_idx]=masked_skeleton
        return torch.tensor(modified_sequence).double(),torch.tensor(clean_sequence).double()

    def getJointsNumber(self):
        if len(self.sequences)!=0:
            return len(self.sequences[0][0])
        else:
            return None

    def getJointCardinality(self):
        if len(self.sequences[0])!=0:
            return len(self.sequences[0][0][0])
        else:
            return None

    def get_temporal_windows(self):
        windows=[]
        for original_sequence in self.sequences_raw:
           windows=windows+slice_sequence_into_windows(original_sequence,window_size=self.window_size)
        return windows


class SkeletonRepetitionMaskingDatasetSingleFrame(Dataset):
    def __init__(self, dataset_path,visible_joints,hip_id=0,exercise="squat",set="train",normalize=False,augmentation=False,crossValidation=None):
        """

        :param dataset_path:
        :param visible_joints:
        :param hip_id: Used to know from which joint rescale all the other joints. In both formats hip joint id is 0
        """
        self.visible_joints=visible_joints
        if crossValidation is None:
            self.sequences=loadRepetitionAnnotationsPickle(dataset_path,exercise_to_get=exercise,set=set)
        else:
            self.sequences=loadRepetitionAnnotationsPickleCrossValidation(dataset_path,exercise_to_get=exercise,set=set,cross_validation_flag=crossValidation)
        self.masked_joints = [item for item in range(21) if item not in visible_joints]
        self.skeletons= list(chain(*self.sequences))
        self.hip_id=hip_id
        self.normalize_skeleton = normalize
        self.augment_skeleton=augmentation

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        clean_skeleton=self.skeletons[idx]
        modified_skeleton = skeleton_utils.centerSkeletonAroundHip(clean_skeleton,hip_id=self.hip_id)
        if self.augment_skeleton:
            modified_skeleton=skeleton_utils.augmentSkeleton(modified_skeleton)
        if self.normalize_skeleton:
            modified_skeleton,_=skeleton_utils.normalize_skeleton_joints_distance(skeleton_to_normalize=modified_skeleton)
        masked_skeleton=skeleton_utils.maskSkeleton(skeleton=modified_skeleton,mask_joints_id=self.masked_joints)
        return torch.tensor(masked_skeleton).double(),torch.tensor(modified_skeleton).double()

    def getJointsNumber(self):
        if len(self.skeletons)!=0:
            return len(self.skeletons[0])
        else:
            return None

    def getJointCardinality(self):
        if len(self.skeletons)!=0:
            return len(self.skeletons[0][0])
        else:
            return None


class SkeletonMaskingDataset(Dataset):
    def __init__(self, dataset_path,visible_joints,hip_id=0,exercise="squat",set="train",normalize=False,augmentation=False,crossValidation=None):
        """

        :param dataset_path:
        :param visible_joints:
        :param hip_id: Used to know from which joint rescale all the other joints. In both formats hip joint id is 0
        """
        self.visible_joints=visible_joints
        if "fit3d" in dataset_path:
            self.sequences=loadAnnotationsJSONS(dataset_path)
            self.masked_joints = [item for item in range(25) if item not in visible_joints]
        else:
            if crossValidation is None:
                self.sequences=loadAnnotationsPickle(dataset_path,exercise_to_get=exercise,set=set)
            else:
                self.sequences,_=loadAnnotationsPickleCrossValidation(dataset_path,exercise_to_get=exercise,set=set,crossValidation_id=crossValidation)
            self.masked_joints = [item for item in range(21) if item not in visible_joints]
        self.skeletons= list(chain(*self.sequences))
        self.hip_id=hip_id
        self.normalize_skeleton = normalize
        self.augment_skeleton=augmentation

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        clean_skeleton=self.skeletons[idx]
        modified_skeleton = skeleton_utils.centerSkeletonAroundHip(clean_skeleton,hip_id=self.hip_id)
        if self.augment_skeleton:
            modified_skeleton=skeleton_utils.augmentSkeleton(modified_skeleton)
        if self.normalize_skeleton:
            modified_skeleton,_=skeleton_utils.normalize_skeleton_joints_distance(skeleton_to_normalize=modified_skeleton)
        masked_skeleton=skeleton_utils.maskSkeleton(skeleton=modified_skeleton,mask_joints_id=self.masked_joints)
        return torch.tensor(masked_skeleton).double(),torch.tensor(modified_skeleton).double()

    def getJointsNumber(self):
        if len(self.skeletons)!=0:
            return len(self.skeletons[0])
        else:
            return None

    def getJointCardinality(self):
        if len(self.skeletons)!=0:
            return len(self.skeletons[0][0])
        else:
            return None


