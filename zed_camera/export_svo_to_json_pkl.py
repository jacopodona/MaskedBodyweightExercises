########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
NOTE: The following script was modified from original version available on ZED SDK.
"""

import cv2
import sys
import pyzed.sl as sl
import time
import ogl_viewer_exporter.viewer as gl
import numpy as np
import json
import os
import pickle
from tqdm import tqdm

def addIntoOutput(out, identifier, tab):
    out[identifier] = []
    for element in tab:
        out[identifier].append(element)
    return out

def serializeBodyData(body_data):
    """Serialize BodyData into a JSON like structure"""
    out = {}
    out["id"] = body_data.id
    #out["unique_object_id"] = str(body_data.unique_object_id)
    #out["tracking_state"] = str(body_data.tracking_state)
    #out["action_state"] = str(body_data.action_state)
    #addIntoOutput(out, "position", body_data.position)
    #addIntoOutput(out, "velocity", body_data.velocity)
    #addIntoOutput(out, "bounding_box_2d", body_data.bounding_box_2d)
    #out["confidence"] = body_data.confidence
    #addIntoOutput(out, "bounding_box", body_data.bounding_box)
    #addIntoOutput(out, "dimensions", body_data.dimensions)
    #addIntoOutput(out, "keypoint_2d", body_data.keypoint_2d)
    addIntoOutput(out, "keypoint", body_data.keypoint)
    #addIntoOutput(out, "keypoint_cov", body_data.keypoints_covariance)
    #addIntoOutput(out, "head_bounding_box_2d", body_data.head_bounding_box_2d)
    #addIntoOutput(out, "head_bounding_box", body_data.head_bounding_box)
    #addIntoOutput(out, "head_position", body_data.head_position)
    #addIntoOutput(out, "keypoint_confidence", body_data.keypoint_confidence)
    #addIntoOutput(out, "local_position_per_joint", body_data.local_position_per_joint)
    #addIntoOutput(out, "local_orientation_per_joint", body_data.local_orientation_per_joint)
    #addIntoOutput(out, "global_root_orientation", body_data.global_root_orientation)
    return out

def serializeBodies(bodies):
    """Serialize Bodies objects into a JSON like structure"""
    out = {}
    #out["is_new"] = bodies.is_new
    #out["is_tracked"] = bodies.is_tracked
    #out["timestamp"] = bodies.timestamp.data_ns
    out["body_list"] = []
    for sk in bodies.body_list:
        out["body_list"].append(serializeBodyData(sk))
    return out

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def extract_to_json(root_dir,take_name,body_joints=18):
    take_filepath=os.path.join(root_dir,take_name)
    output_json_name=take_name.replace(".svo",".json")
    output_dir=root_dir.replace("svo","json")
    output_dir=os.path.join(output_dir,f"{str(body_joints)} joints")



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_json_filepath=os.path.join(output_dir,output_json_name)

    # common parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
    init_params.set_from_svo_file(take_filepath)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    zed = sl.Camera()
    error_code = zed.open(init_params)
    if (error_code != sl.ERROR_CODE.SUCCESS):
        print("Can't open camera: ", error_code)

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True
    error_code = zed.enable_positional_tracking(positional_tracking_parameters)
    if (error_code != sl.ERROR_CODE.SUCCESS):
        print("Can't enable positionnal tracking: ", error_code)

    body_tracking_parameters = sl.BodyTrackingParameters()
    body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    if body_joints==18:
        body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
    elif body_joints==34:
        body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_34
    else:
        print("Inserted unsupported body joint type. Choose between 18 and 34")
        return
    body_tracking_parameters.enable_body_fitting = True
    body_tracking_parameters.enable_tracking = True

    body_runtime_params = sl.BodyTrackingRuntimeParameters()
    if "plank" in take_name:
        body_runtime_params.detection_confidence_threshold = 30
    else:
        body_runtime_params.detection_confidence_threshold = 80

    error_code = zed.enable_body_tracking(body_tracking_parameters)
    if (error_code != sl.ERROR_CODE.SUCCESS):
        print("Can't enable positionnal tracking: ", error_code)

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    #viewer = gl.GLViewer()
    #viewer.init()

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    # single_bodies = [sl.Bodies]

    skeleton_file_data = {}
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_bodies(bodies,body_runtime_params)
        skeleton_file_data[str(bodies.timestamp.get_milliseconds())] = serializeBodies(bodies)
    #while (viewer.is_available()):
    #    if zed.grab() == sl.ERROR_CODE.SUCCESS:
    #        zed.retrieve_bodies(bodies)
    #        skeleton_file_data[str(bodies.timestamp.get_milliseconds())] = serializeBodies(bodies)
    #        viewer.update_bodies(bodies)

    # Save data into JSON file:
    file_sk = open(output_json_filepath, 'w')
    file_sk.write(json.dumps(skeleton_file_data, cls=NumpyEncoder, indent=4))
    file_sk.close()
    print(f"Finished extracting poses from {take_name}")

    #viewer.exit()
    return output_json_filepath

def load_pose_sequence(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    sequence=[]
    dropped_frames=0
    for i, body in enumerate(data.values()):
        skeleton=[]
        if len(body['body_list'])!=0:
            for joint_coordinates in body['body_list'][0]["keypoint"]:
                skeleton.append(joint_coordinates)
            if len(skeleton)==18: #If 18 joints skeleton format, append pelvis joint created by interpolating the two hips, create 19 joint skeleton
                l_hip = np.array(skeleton[11])
                r_hip = np.array(skeleton[8])
                pelvis_position = np.mean([l_hip, r_hip], axis=0)
                skeleton.append(pelvis_position)
            skeleton=np.array(skeleton)
            if not np.isnan(skeleton).any(): #Remove partial skeletons
                sequence.append(skeleton)
            else:
                dropped_frames+=1
    print(f"Dropped {dropped_frames} frames due to missing joint positions")
    #plot_utils.plot_take(sequence,center_on_hips=True,skip=1,source="ZED")
    return np.array(sequence)

def save_pose_sequence_pickle(pose_sequence, output_path):
    with open(output_path, 'wb') as pickle_file:
        pickle.dump(pose_sequence, pickle_file)

if __name__ == "__main__":
    root_dir="../data/svo/final_takes"
    body_joints=18 #Choose between 18 and 34
    take_list=os.listdir(root_dir)
    for i in tqdm(range(len(take_list))):
        take_name=take_list[i]
        saved_path=extract_to_json(root_dir,take_name,body_joints)
        sequence=load_pose_sequence(saved_path)
        pkl_path = root_dir.replace("svo", "pkl/zed")
        pkl_path=os.path.join(pkl_path,f"{str(body_joints)} joints")
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)
        pkl_name=take_name.replace(".svo",".pkl")
        pkl_output_path=os.path.join(pkl_path,pkl_name)
        save_pose_sequence_pickle(sequence,pkl_output_path)