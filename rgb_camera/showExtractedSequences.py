import plot_utils
import os
import pickle as pkl
import matplotlib.pyplot as plt
import skeleton_utils
import data_utils

sequence_dir="../data/pkl/rgb/final_takes"
take_list=os.listdir(sequence_dir)

for take_name in take_list:
    # load take
    with open(os.path.join(sequence_dir, take_name), 'rb') as file:
        data = pkl.load(file)
    data=skeleton_utils.normalize_sequence(clean_sequence=data,hip_index=33,head_index=0)
    plot_utils.plot_take(data, center_on_hips=True, title=take_name,source="MPOSE",skip=15)
    plt.close(1)
