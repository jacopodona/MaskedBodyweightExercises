import pyzed.sl as sl
import cv2
import os
from tqdm import tqdm

def make_video_from_list(image_list,output_video_path,fps=60):
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


def extractVideoFromSVO(filepath):
    output_filepath=filepath.replace(".svo",".avi")
    output_filepath = output_filepath.replace("svo", "rgb")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.set_from_svo_file(filepath)


    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()


    # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    sequence=[]
    print("\tRetrieving images...")
    while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT)
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured

        image_ocv=image.get_data()[:,:,:3].copy()
        sequence.append(image_ocv)
        i = i + 1
        #if i % 25==0:
        #    cv2.imwrite(f"frames/{output_filepath.split('/').pop()}_{i}.png",image_ocv)
        #cv2.imshow("Prova",image_ocv)
        #cv2.waitKey(10)
    print("\tPreparing video...")
    make_video_from_list(sequence,output_filepath)

    # Close the camera
    zed.close()
    print("\tDone.")

if __name__ == "__main__":
    svo_path= "../data/svo/final_takes/"
    output_path=svo_path.replace("svo","rgb")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    takes=os.listdir(svo_path)
    for i in tqdm(range(len(takes))):
        takename=takes[i]
        take_path=os.path.join(svo_path,takename)
        print("Extracting video from",take_path)
        extractVideoFromSVO(take_path)