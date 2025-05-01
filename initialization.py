'''
This file should generate the following files:
 - cameras.txt - (camera intrinsics)
 - images.txt - (camera poses)
 - points3D.txt - (3D points)

It will do so by loading data from a video file, or potentially a set of images.

For now, it will be run using:

python initialization.py video --video_path <path_to_video_file> --image_folder <path_to_image_folder> --output_folder <output_folder> --fps <fps>

OR 

python initialization.py images --image_folder <path_to_image_folder> --output_folder <output_folder>

'''
import argparse
import cv2
import os
import subprocess

def save_images_from_video(video_path, image_folder, fps):

    os.makedirs(image_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        read, frame = cap.read()
        if not read:
            break
        if frame_count % fps == 0:
            filename = os.path.join(image_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to {image_folder}")

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_colmap_pipeline(image_folder, output_folder, database_path = "database.db"):

    os.makedirs(output_folder, exist_ok=True)

    run_command([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_folder
    ])

    run_command([
        "colmap", "exhaustive_matcher",
        "--database_path", database_path
    ])

    run_command([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_folder,
        "--output_path", output_folder
    ])



def main():

    parser = argparse.ArgumentParser(description='Initialize 3D reconstruction from video or images.')

    subparsers = parser.add_subparsers(dest="command")

    #Video parser
    video_parser = subparsers.add_parser("video", help="Extract frames from a video, then use them for initialization")
    video_parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    video_parser.add_argument("--image_folder", type=str, required=True, help="Path to save images from video")
    video_parser.add_argument("--output_folder", type=str, required=True, help="Folder to save point cloud")
    video_parser.add_argument("--fps", type=int, default=3, help="Frames per second to extract")

    #Image parser
    image_parser = subparsers.add_parser("images", help="Use images for initialization")
    image_parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder")
    image_parser.add_argument("--output_folder", type=str, required=True, help="Folder to save point cloud")
    args = parser.parse_args()

    if args.command == "video":
        video_path = args.video_path
        image_folder = args.image_folder
        output_folder = args.output_folder
        fps = args.fps
    elif args.command == "images":
        image_folder = args.image_folder
        output_folder = args.output_folder

    mode = args.command
    print(f"Mode: {mode}")

    if mode == "video":
        save_images_from_video(video_path, image_folder, fps)
    
    run_colmap_pipeline(image_folder, output_folder)


if __name__ == "__main__":
    main()