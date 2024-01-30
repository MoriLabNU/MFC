import argparse
import os
import cv2
import tqdm
import pandas as pd

global_frame_count = 1


def save_frame(frame, path):
    cv2.imwrite(path, frame)


def extract_frames(videofile, save_dir, step, ext, max_frame):
    global global_frame_count
    if not os.path.exists(save_dir):  # Create folder to save frames
        os.makedirs(save_dir)
    # Get name of the file without extention ex: /home/gohil/running.MOV -> running
    basename = os.path.splitext(os.path.basename(videofile))[0]
    print("BASENAME : ", basename)
    cap = cv2.VideoCapture(videofile)  # capture video frame
    orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_count = 0
    fps_count = 0

    save_step = step

    print("save directory: {}".format(save_dir))
    # extract only single frame
    pbar = tqdm.tqdm(
        total=int(max_frame / save_step),
        desc="extracting frames at step: {}".format(step),
    )
    # read all frames
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1
        fps_count += 1
        if frame_count >= max_frame:
            return
        if not success:
            print("Failed to read video")
            return
        # save frame
        if fps_count % save_step == 0:
            p = os.path.join(save_dir, f"frame_{frame_count}_endo" + "." + ext)
            global_frame_count += 1
            save_frame(frame, p)
            fps_count = 0
            pbar.update(1)
    pbar.close()
    cap.release()  # Release the video capture object


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="./data/cholec80"
    )
    parser.add_argument(
        "--cs8k_path", type=str, default="./data/CholecSeg8k"
    )
    parser.add_argument(
        "--save_path", type=str, default="./DATA/cholec/img/train"
    )
    parser.add_argument(
        "--img_ext", type=str, default="png",
        help="Save images with png or jpg or other extention",
    )
    parser.add_argument(
        "--step", type=float, default=25, help="Frame per second to capture in video"
    )
    args = parser.parse_args()
    print(args)

    frames = 0
    videos = 0

    if os.path.isdir(args.path):
        filenames = os.listdir(os.path.join(args.path, 'videos'))
        file_cs8k = os.listdir(args.cs8k_path)
        for filename in filenames:
            if filename.split('.')[0] not in file_cs8k and '.mp4' in filename:
                df = pd.read_csv(
                    os.path.join(args.path, 'phase_annotations', filename.replace('.mp4', '-phase.txt')),
                    sep="\t")
                preparation = df[df.Phase == "Preparation"]
                max_frame = len(preparation)
                video_file = os.path.abspath(os.path.join(args.path, 'videos', filename))
                print("Creating images for: {}".format(video_file))
                frames += max_frame
                videos += 1
                save_path = os.path.join(args.save_path, filename.split('.')[0])
                extract_frames(
                    video_file,
                    save_path,
                    args.step,
                    args.img_ext,
                    max_frame
                )
    else:
        print("Invalid path!!! Please check your --path value")

    print(videos, frames)
