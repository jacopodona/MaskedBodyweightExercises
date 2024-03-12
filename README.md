# MaskedBodyweightExercises
This repository contains the code developed for the thesis work "Evaluation of bodyweight exercises
via masked autoencoders" by Jacopo Donà for the Master's Degree of Artificial Intelligence System of the University of Trento. 

Supervisor: Nicola Conci

Co-supervisor: Giulia Martinelli

This work employs Masked Autoencoders for performing Masked Pose Modeling, as described in the article 
[UNPOSED: an Ultra-wideband Network for Pose Estimation with Deep Learning](https://ieeexplore.ieee.org/abstract/document/10180019) by Martinelli et al.

The work consisted in acquiring 2 datasets to capture bodyweight exercises with both correct and finegrained mistakes, using two acquisition sources:
- Motion Capture Data
- Real World RGB Pose Estimation

The repetitions are extracted using the script `extract_repetition_from_sequence.py`, which automatically extracts the repetition aligning them along D frames, which is a settable parameters.

Once the repetitions are extracted, they are stored in `data/pkl/optitrack/single_repetitions/D_frames`

From there, you can train the network on each exercise using `single_frame/train_single_frame.py` or `single_frame/crossvalidation_single_frame_repetition.py`

Some example models are given in `nets/single_frame/sample_networks` for each exercise.

To visualize the network prediction and obtain metric result, insert the model path into `visualizeReconstructedSkeleton.py`, which evaluates the network on the validation set of repetition using D=21.
The reconstruction results are visualized through a plot, while the metric results are saved in a csv inside the model folder, and represent the Euclidean Error in meters.

In the `performance_analysis` folder, the algorithms for performing performance analysis for each acquisition source are available. With the `performance_analysisOPT.py`
the scores for optitrack full body, reconstructed optitrack, reconstructed Zed, reconstructed Mediapipe Pose Landmarker are available.

### Skeleton Data
For convenience, the data is already given in its post-processed and saved in its `pkl` format, for the whole sequence and in its repetition form.

If you want to run the Pose Estimation yourself, the scripts for extracting the skeleton data from the csv are in the `optitrack` folder for the motion capture data.

For the real world dataset, the scripts for extracting the pose are available in the `rgb_camera` and `zed_camera` folder.

## Requirements
For the following project, the necessary libraries are listed in the `requirements.txt` file.

The data is given in csv and pkl format for Optitrack, in json and pkl for the Zed camera, and in pkl format for the Mediapipe Pose Landmarker.

The original `svo/avi` data for the real world dataset is available at [this link](https://drive.google.com/file/d/1AdAoApxBWlPRum7dWsfsg7UtSeeGK6QW/view?usp=sharing)

#### Installing Zed SDK Libraries
To install the Zed camera python library, it is required to first download the Zed SDK.
A link to the installation guide followed for this project is available [at this link](https://www.stereolabs.com/docs/app-development/python/install)

#### Pytorch version
PyTorch with CUDA support is required to train the Pose Reconstruction Network. The code was developed using on python 3.9, pytorch 2.0.1, using CUDA 11.7.