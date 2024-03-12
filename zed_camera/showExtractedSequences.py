import plot_utils
import os
import pickle as pkl
import matplotlib.pyplot as plt

sequence_dir="../data/pkl/zed/final_takes/18 joints"
take_list=os.listdir(sequence_dir)

for take_name in take_list:
    # load take
    with open(os.path.join(sequence_dir, take_name), 'rb') as file:
        data = pkl.load(file)
    plot_utils.plot_take(data, center_on_hips=True, title=take_name,source="ZED",skip=10)
    plt.close(1)
