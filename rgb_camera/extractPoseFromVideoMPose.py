import os.path

import cv2
import mediapipe as mp
import time
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import plot_utils
import skeleton_utils


def computeExtraLandmarks(skeleton):
    l_hip = np.array(skeleton[23])
    r_hip = np.array(skeleton[24])
    pelvis_position = np.mean([l_hip, r_hip], axis=0)

    l_should = np.array(skeleton[11])
    r_should = np.array(skeleton[12])
    neck_position = np.mean([l_should, r_should], axis=0)

    return pelvis_position,neck_position

def make_video_from_image_list(image_list,output_video_path,fps=30):
    # Define the video codec, frames per second (fps), and video dimensions
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    width=image_list[0].shape[1]
    height=image_list[0].shape[0]

    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate through the list of images and add them to the video
    for frame_idx in range(len(image_list)):
        frame=image_list[frame_idx]
        video.write(frame)


    # Release the video writer
    video.release()

def main(video_path,show=True,dump=False):
    pkl_output_path=video_path.replace(".avi",".pkl")
    pkl_output_path=pkl_output_path.replace("rgb","pkl/rgb")
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(
        model_complexity=2 #0 for lite, 1 for Full, 2 for Heavy
    )
    mpDraw = mp.solutions.drawing_utils
    mpDraw._VISIBILITY_THRESHOLD = 0.2

    #cap = cv2.VideoCapture(0)
    print("Showing",video_path)
    cap = cv2.VideoCapture(video_path)
    pTime = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequence=[]
    sequence_images=[]

    out_video_folder = video_path.split("\\")[0]
    video_name = video_path.split("\\")[1]

    out_video_folder = out_video_folder.replace("rgb", "rgb_with_pose")
    if not os.path.exists(out_video_folder):
        os.makedirs(out_video_folder)
    out_video_path = os.path.join(out_video_folder, video_name)

    notFinished=True
    progress_bar = tqdm(total=total_frames, desc='Processing frames')
    i=0
    while notFinished:
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            #print(results.pose_landmarks)
            h, w, c = img.shape
            positions_3d = []
            if results.pose_landmarks:
                only_reference = img.copy()
                for id, lm in enumerate(results.pose_world_landmarks.landmark):
                    #positions_3d.append((cx,cy,lm.z))
                    positions_3d.append([lm.x,-lm.y,-lm.z])
                    #if id==26 or id==14:
                if show:
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h,w,c=img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # positions_3d.append((cx,cy,lm.z))
                        # if id==26 or id==14:
                        #cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

                pelvis_position,neck_position=computeExtraLandmarks(positions_3d)
                pcx, pcy = int(pelvis_position[0] * w), int(pelvis_position[1] * h)
                ncx, ncy = int(neck_position[0] * w), int(neck_position[1] * h)
                positions_3d.append(pelvis_position)
                positions_3d.append(neck_position)
                #if i==5:
                #    figure = plt.figure(figsize=(10, 8))
                #    ax = plt.axes(projection='3d')
                #    positions_3d=skeleton_utils.centerSkeletonAroundHip(positions_3d,hip_id=33)
                #    plot_utils.plot_3d_skeleton_MPOSE(positions_3d,ax,"black")
                #    plt.show()
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                #cv2.putText(img, str(int(frame_number)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                sequence_images.append(img)
                if show:
                    cv2.imshow("Frame", img)
                    key=cv2.waitKey(1)
                    #if key == 113: # for 'q' key
                    #    print("\tExiting...")
                    #    break
                sequence.append(positions_3d)
                progress_bar.update(1)
                i+=1
        else:
            notFinished=False
    if dump:
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(sequence, f)  # deserialize using load()

    #make_video_from_image_list(image_list=sequence_images,output_video_path=out_video_path,fps=60)

if __name__ == '__main__':
    video_dir="../data/rgb/final_takes"
    output_dir=video_dir.replace("rgb","pkl/rgb")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_list=os.listdir(video_dir)
    for video_name in video_list:
        video_path=os.path.join(video_dir,video_name)
        main(video_path,dump=False)