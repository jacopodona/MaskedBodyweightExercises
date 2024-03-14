import math
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
from scipy.signal import find_peaks,correlate,savgol_filter,medfilt
import plot_utils
import skeleton_utils

def count_pattern_occurrences(derivative_signal, threshold):
    # Find peaks (sharp increases) and troughs (sharp decreases)
    peaks, _ = find_peaks(derivative_signal, height=threshold)
    troughs, _ = find_peaks(-derivative_signal, height=threshold)

    # Combine peaks and troughs and sort the indices
    pattern_indices = sorted(np.concatenate((peaks, troughs)))

    # Count the number of occurrences
    occurrences = len(pattern_indices) - 1

    return occurrences, pattern_indices

def min_max_normalization(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


def filter_occurencses(points):
    # Define a distance threshold (adjust as needed)
    distance_threshold = 25

    # Create a new list to store filtered points
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if points[i] - filtered_points[-1] >= distance_threshold:
            filtered_points.append(points[i])
    return len(filtered_points),np.array(filtered_points)


def find_closest_or_zero_value(vector, start, end):
    subvector = vector[start:end + 1]  # Include the end index in the subvector

    # Check if zero is in the subvector
    if 0 in subvector:
        return start + np.where(subvector == 0)[0][0]

    # Find the index of the value closest to zero
    closest_index = start + np.argmin(np.abs(subvector - 0))

    return closest_index


def find_exercise_keypoints(values,derivative):
    exercise_couples = values.reshape(-1, 2)
    start_points=[]
    middle_points=[]
    end_points = []
    values2=np.insert(values.copy(),0,0)[:-1]
    start_couples=values2.reshape(-1, 2)
    values3 = np.insert(values.copy(), len(values),len(derivative))[1:]
    end_couples = values3.reshape(-1, 2)
    for rep_range in start_couples:
        start_points.append(find_closest_or_zero_value(derivative, rep_range[0], rep_range[1]))
    for rep_range in exercise_couples:
        middle_points.append(find_closest_or_zero_value(derivative,rep_range[0],rep_range[1]))
    for rep_range in end_couples:
        end_points.append(find_closest_or_zero_value(derivative,rep_range[0],rep_range[1]))
    return start_points,middle_points,end_points


def get_repetitions_plank(take, temporal_window_size, num_segments=10):
    array_length = len(take)
    segment_size = array_length // num_segments
    sequence_repetitions=[]
    start_points = [(i * segment_size)+ 5 for i in range(num_segments)]
    end_points = [(i + 1) * segment_size for i in range(num_segments)]
    end_points[-1] = array_length-1  # Make sure the last segment extends to the end
    assert len(start_points)==len(end_points)==num_segments
    for rep_idx in range(len(start_points)):
        start_point=start_points[rep_idx]
        end_point = end_points[rep_idx]
        repetition_indexes = np.linspace(start_point, end_point, temporal_window_size, dtype=int)
        repetition_values = take[repetition_indexes]
        sequence_repetitions.append(repetition_values)
    return sequence_repetitions

def plotSignal(signal,title="",y_value="Hip Y value (meters)"):
    plt.plot(range(len(signal)), signal)

    # Add labels and a legend
    plt.xlabel('Frames')
    plt.ylabel(y_value)
    plt.title(title)
    #plt.legend()

    # Show the plot
    plt.show()

def extract_repetitions_temporal_window(data_path= "data/pkl/optitrack/baseline_skeleton/",D=21,pelvis_index=0,window_size=201,derivative_threshold=0.002):
    """

    :param D: Output frames
    :param pelvis_index: Index of the pelvis joint in the skeleton
    :param window_size: Window size for the smoothing, must be odd (201 for optitrack, 61 for 60Hz acquisition on Zed)
    :return:
    """
    files = os.listdir(data_path)
    plank_debug=[]
    folder_name=data_path.split("/").pop()
    for f in files:
        file_path = os.path.join(data_path, f)
        exercise_name = f.split(".")[0]
        #output_folder = "data/pkl/single_repetitions/"
        output_folder = data_path.replace(folder_name,"single_repetitions")
        output_folder = os.path.join(output_folder, str(D) + "_frames", exercise_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(file_path, 'rb') as file:
            take = pkl.load(file)
        take_array = np.array(take)
        if "squat" in f or "lunge" in f:
            print(f)

            y_original = take_array[:, pelvis_index, 1]
            y_original=min_max_normalization(y_original)
            first_derivative_original = np.gradient(y_original)
            #if "squat" in f:
            #    plotSignal(y_original,"Hip Position During Squat Repetitions")
            #if "lunge" in f:
            #    plotSignal(y_original,"Hip Position During Lunge Repetitions")
            # Apply smoothing filter
            poly_order = 1
            y_smoothed = savgol_filter(y_original, window_size, poly_order)

            #y_smoothed=medfilt(y_original,kernel_size=window_size)
            first_derivative_smoothed= np.gradient(y_smoothed)
            #first_derivative_smoothed = savgol_filter(first_derivative_smoothed, 21, poly_order)
            first_derivative_smoothed=savgol_filter(first_derivative_smoothed, 31, poly_order)

            #if "squat" in f:
            #    plotSignal(first_derivative_smoothed, r"$\frac{dY}{dt}$ during Squat repetitions",y_value=r"$\frac{dY}{dt}$ value")
            #if "lunge" in f:
            #    plotSignal(first_derivative_smoothed, r"$\frac{dY}{dt}$ during Lunge repetitions",y_value=r"$\frac{dY}{dt}$ value")

            occurences,indexes=count_pattern_occurrences(first_derivative_smoothed,threshold=derivative_threshold)
            occurences,indexes=filter_occurencses(indexes)
            #print(f"Found {occurences} repetitions at indexes :{indexes}")
            print("\tOccurences",occurences)
            #assert occurences==20

            plt.scatter(indexes,first_derivative_smoothed[indexes],c="orange")
            #plotSignal(first_derivative_smoothed,title="Derivative peaks",y_value="$\\frac{dY}{dt}$ value")
            #plt.scatter(indexes, y_original[indexes], c="orange")
            #plotSignal(y_original, title="Original Signal with Derivative peaks", y_value="hip height (m)")
            if occurences%2==1:
                occurences-=1
                indexes=indexes.tolist()
                indexes.pop(14)
                plt.scatter(indexes, first_derivative_smoothed[indexes], c="red")
                #plotSignal(first_derivative_smoothed, title="Derivative peaks", y_value="$\\frac{dY}{dt}$ value")
                plt.scatter(indexes, y_original[indexes], c="orange")
                # plotSignal(y_original, title="Original Signal with Derivative peaks", y_value="hip height (m)")
                indexes=np.array(indexes)
                plt.show()
            start,middle,end=find_exercise_keypoints(indexes,first_derivative_smoothed)

            #Plots for thesis

            #plt.scatter(indexes,first_derivative_smoothed[indexes],c="orange")
            #plotSignal(first_derivative_smoothed,title="Derivative peaks",y_value="$\\frac{dY}{dt}$ value")
            #plt.scatter(indexes, y_original[indexes], c="orange")
            #plotSignal(y_original, title="Original Signal with Derivative peaks", y_value="hip height (m)")

            #plt.scatter(start, first_derivative_smoothed[start], c="green", label="Start")
            #plt.scatter(middle, first_derivative_smoothed[middle], c="blue", label="Middle")
            #plt.scatter(end, first_derivative_smoothed[end], facecolors='none', edgecolors='r', label="End")
            #plt.legend()
            #plotSignal(first_derivative_smoothed,title="Repetition Identifiers in Derivative Signal",y_value="$\\frac{dY}{dt}$ value")

            #plt.scatter(start, y_original[start], c="green", label="Start")
            #plt.scatter(middle, y_original[middle], c="blue", label="Middle")
            #plt.scatter(end, y_original[end], facecolors='none', edgecolors='r', label="End")
            #plt.legend()
            #plotSignal(y_original, title="Repetition Identifiers in Original Signal",y_value="hip height (m)")
            if len(start)==len(middle)==len(end):
                print(f"Sequence {f} OK. Saving repetitions in pickle file")
                assert len(start) == len(middle) == len(end)
                repetitions = extractRepetitionFromKeypoints(take_array, start, middle, end, temporal_window_size=D)
                for rep_idx in range(len(repetitions)):
                    repetition = repetitions[rep_idx].tolist()
                    output_path = os.path.join(output_folder,"rep_" + str(rep_idx)+".pkl")
                    with open(output_path, 'wb') as file:
                        pkl.dump(repetition, file)
                    #print(f"Data has been pickled and saved to {output_path}")
            else:
                print(f"Sequence {f} NOT OK")
        elif "plank" in f:
            repetitions=get_repetitions_plank(take_array,temporal_window_size=D)
            print(f"Sequence {f} OK. Saving repetitions in pickle file")
            for rep_idx in range(len(repetitions)):
                repetition = repetitions[rep_idx].tolist()
                output_path = os.path.join(output_folder, "rep_" + str(rep_idx) + ".pkl")
                with open(output_path, 'wb') as file:
                    pkl.dump(repetition, file)
                plank_debug.append(repetition)
    #print(plank_debug)

def extractRepetitionFromKeypoints(take, start_array, middle_array, end_array,temporal_window_size):
    d=int(math.ceil((temporal_window_size)/2))
    sequence_repetitions=[]
    value=1000000
    for rep_idx in range(len(start_array)):
        start_point=start_array[rep_idx]
        middle_point=middle_array[rep_idx]
        end_point = end_array[rep_idx]
        value=min(end_point - start_point, value)
        descent_indices = np.linspace(start_point, middle_point, d , dtype=int)
        ascent_indices= np.linspace(middle_point, end_point, d , dtype=int)
        repetition_indexes = sorted(list(set(descent_indices) | set(ascent_indices)))
        repetition_values = take[repetition_indexes]
        sequence_repetitions.append(repetition_values)
    print(f"\tMaximum extractable length: {value}")
    return sequence_repetitions


def presentation_plot(y_original,start,middle,end):
    fig,axes=plt.subplots(figsize=(16, 5),nrows=1,ncols=3)
    temporal_windows=[3,9,51]
    first_rep = y_original[start[4]:end[4]]
    middle_p = middle[4] - start[4]
    end_p = end[4] - start[4] - 1
    start_p = 0
    for i in range(0,3):
        axes[i].plot(range(len(first_rep)), first_rep, c="black")
        temporal_window=temporal_windows[i]
        d = int(math.ceil((temporal_window) / 2))
        descent_indices = np.linspace(start_p, middle_p, d, dtype=int)
        ascent_indices = np.linspace(middle_p, end_p, d, dtype=int)
        #xes[i].set_ylabel("pelvis height (m)")
        axes[i].set_xlabel("Frames")
        axes[i].set_title(f"D={temporal_window}")
        axes[i].scatter(descent_indices,first_rep[descent_indices],c="orange",label="Sampled Frame")
        axes[i].scatter(ascent_indices, first_rep[ascent_indices], c="orange")
        axes[i].scatter(start_p, first_rep[start_p], c="green", label="Start")
        axes[i].scatter(middle_p, first_rep[middle_p], c="blue", label="Middle")
        axes[i].scatter(end_p, first_rep[end_p], c="red", label="End")
        axes[i].legend()
    fig.supylabel("Pelvis height (m)")
    fig.suptitle("Squat Repetition Temporal Window Sampling")
    plt.show()


def window_example(take,start,middle,end):
    temporal_window=5
    fig,axes=plt.subplots(figsize=(20, 5),nrows=1,ncols=temporal_window,subplot_kw={'projection': '3d'})
    d = int(math.ceil((temporal_window) / 2))
    repetitions=extractRepetitionFromKeypoints(take=np.array(take),start_array=start,middle_array=middle,end_array=end,temporal_window_size=temporal_window)
    repetition=repetitions[5]
    for i in range(temporal_window):
        skeleton= skeleton_utils.centerSkeletonAroundHip(repetition[i])
        plot_utils.plot_3d_skeleton_OPT(skeleton,ax=axes[i],color="black")
        axes[i].set_title(f"Frame {i+1}/{temporal_window}")
        axes[i].view_init(azim=60)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(-1, 1)
        axes[i].set_zlim(-1, 1)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_zticklabels([])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_zlabel("")
        #xes[i].set_ylabel("pelvis height (m)")
    fig.suptitle("Extracted Squat Temporal Window $D=5$")
    fig.tight_layout()
    #plt.subplots_adjust(wspace=0)  # Adjust the value as needed
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.8, wspace=0, hspace=0)
    plt.show()

def window_example_plank():
    data_path = "data/pkl/optitrack/baseline_skeleton/"
    filename = "plank_good.pkl"
    file_path = os.path.join(data_path, filename)
    with open(file_path, 'rb') as file:
        take = pkl.load(file)
    take= np.array(take)
    temporal_window=5
    fig,axes=plt.subplots(figsize=(20, 5),nrows=1,ncols=temporal_window,subplot_kw={'projection': '3d'})
    repetitions=get_repetitions_plank(take=np.array(take),temporal_window_size=temporal_window)
    repetition=repetitions[5]
    for i in range(temporal_window):
        skeleton= skeleton_utils.centerSkeletonAroundHip(repetition[i])
        plot_utils.plot_3d_skeleton_OPT(skeleton,ax=axes[i],color="black")
        axes[i].set_title(f"Frame {i+1}/{temporal_window}")
        axes[i].view_init(azim=60)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(-1, 1)
        axes[i].set_zlim(-1, 1)
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_zticklabels([])
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_zlabel("")
        #xes[i].set_ylabel("pelvis height (m)")
    fig.suptitle("Extracted Plank Temporal Window $D=5$")
    fig.tight_layout()
    #plt.subplots_adjust(wspace=0)  # Adjust the value as needed
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.8, wspace=0, hspace=0)
    plt.show()

def main():
    data_path = "data/pkl/optitrack/baseline_skeleton/"
    filename="squat_good.pkl"
    file_path=os.path.join(data_path,filename)
    D=21 #temporal window output size, must be odd
    with open(file_path, 'rb') as file:
        take = pkl.load(file)
    take_array = np.array(take)
    # Assuming 'signal' is your 1D array representing the signal

    y_original = take_array[:, 0, 1]
    y_original = min_max_normalization(y_original)
    first_derivative_original = np.gradient(y_original)

    # Apply smoothing filter
    window_size = 201  # Must be an odd number
    poly_order = 1
    y_smoothed = savgol_filter(y_original, window_size, poly_order)
    # y_smoothed=medfilt(y_original,kernel_size=window_size)
    first_derivative_smoothed = np.gradient(y_smoothed)
    first_derivative_smoothed = savgol_filter(first_derivative_smoothed, 21, poly_order)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes[0, 0].plot(np.arange(len(y_original)), y_original, label='Original Signal')
    axes[0, 0].set_title("Original Signal")
    axes[1, 0].plot(np.arange(len(first_derivative_original)), first_derivative_original, label='First Derivative')
    axes[1, 0].set_title("First Derivative")

    axes[0, 1].plot(np.arange(len(y_smoothed)), y_smoothed, label='Smoothed Signal')
    axes[0, 1].set_title("Smoothed Signal")
    axes[1, 1].plot(np.arange(len(first_derivative_smoothed)), first_derivative_smoothed, label='First Derivative')
    axes[1, 1].set_title("First Derivative")

    occurences, indexes = count_pattern_occurrences(first_derivative_smoothed, threshold=0.002)
    occurences, indexes = filter_occurencses(indexes)
    # print(f"Found {occurences} repetitions at indexes :{indexes}")
    axes[0, 1].scatter(indexes, y_smoothed[indexes], c="orange")
    axes[1, 1].scatter(indexes, first_derivative_smoothed[indexes], c="orange")
    # assert occurences==20
    start, middle, end = find_exercise_keypoints(indexes, first_derivative_smoothed)
    axes[1, 1].scatter(start, first_derivative_smoothed[start], c="green", label="Start")
    axes[1, 1].scatter(middle, first_derivative_smoothed[middle], c="blue", label="Middle")
    axes[1, 1].scatter(end, first_derivative_smoothed[end], facecolors='none', edgecolors='r', label="End")
    axes[0, 1].scatter(start, y_smoothed[start], c="green", label="Start")
    axes[0, 1].scatter(middle, y_smoothed[middle], c="blue", label="Middle")
    axes[0, 1].scatter(end, y_smoothed[end], facecolors='none', edgecolors='r', label="End")
    axes[0, 1].legend()
    #plt.show()
    #presentation_plot(y_original,start,middle,end,temporal_window=21)
    #window_example(take,start,middle,end)
    window_example_plank()

def check_saved_sequences(path,frames):
    if ("optitrack" in path or "reconstructed" in path):
        source="OPT"
    elif "zed" in path:
        source="ZED"
    elif "rgb" in path:
        source="MPOSE"
    #path= f"data/pkl/optitrack/single_repetitions/{frames}_frames"
    path=os.path.join(path,f"{frames}_frames")
    if os.path.exists(path):
        exercise_list=os.listdir(path)
        exercise_list=["squat_back_forward","squat_too_high"]
        for exercise in exercise_list:
            folder_path=os.path.join(path,exercise)
            take_list=os.listdir(folder_path)
            print("Showing", exercise)
            for take_name in take_list:
                #load take
                with open(os.path.join(folder_path,take_name), 'rb') as file:
                    data = pkl.load(file)
                plot_utils.plot_take(data,skip=2,center_on_hips=True,title=take_name,source=source)
                #plt.close(1)
    else:
        print("Path does not exist")

if __name__ == '__main__':
    #main()
    D = 21

    #OPTITRACK
    datapath = "data/pkl/optitrack/baseline_skeleton"
    extract_repetitions_temporal_window(datapath,D,pelvis_index=0,window_size=201,derivative_threshold=0.002)
    check_saved_sequences(path="data/pkl/optitrack/single_repetitions",frames=D)

    # ZED
    #datapath = "data/pkl/zed/final_takes/18 joints"
    #extract_repetitions_temporal_window(datapath, D, pelvis_index=18, window_size=61, derivative_threshold=0.004)
    #check_saved_sequences(path="data/pkl/zed/final_takes/single_repetitions", frames=D)
    #check_saved_sequences(path="data/pkl/reconstructed_zed/single_repetitions", frames=D)

    # MPose
    #datapath = "data/pkl/rgb/30 gennaio lunge front higher"
    #extract_repetitions_temporal_window(datapath,D,pelvis_index=33,window_size=55,derivative_threshold=0.002)
    #check_saved_sequences(path="data/pkl/rgb/single_repetitions",frames=D)
    #check_saved_sequences(path="data/pkl/rgb/single_repetitions",frames=D)
