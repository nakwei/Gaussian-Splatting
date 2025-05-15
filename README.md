An attempt at implementing Gaussian Splatting! 
By Jack Sullivan, Sam Bradley, & Nathan Kwei

How to run: 
If you are trying to create your own dataset from scratch (either images or a video), 
you must first create the necessary .bin files via Colmap. This can be done with the
following commands:

python3 initialization.py video --video_path <path_to_video_file> --image_folder <path_to_image_folder> --output_folder <output_folder> --fps <fps>
OR 
python3 initialization.py images --image_folder <path_to_image_folder> --output_folder <output_folder>

fps represents how many frames to extract from the video. For a higher quality dataset, 
individually captured photographs are ideal. This code will take some time to run. 
We are still struggling with perfecting our personal dataset, so it may be preferrable
to use one of the premade datasets and proceed from this point. 

Once sparse/0/ is created and all binary files are available, we can move onto to training. 
This is done with the command 

python3 train.py --path <directory_containing_sparse_and_images>

The directory must have the following structure

direc
├─── sparse
│    └─── 0
│         └───cameras.bin
│             points3D.bin
│             images.bin
└───images
│   └───IMG0001.jpg
│       IMG0002.jpg
│       ....

This training process also will take a while and can be finnicky, especially regarding
CUDA/GPU usage. Once the training is complete, there should be a new folder called 
data with a file called final.npy. This is the numpy version of our splat. The 
final step is then to convert into .ply, which is an easier format to visualize. 
To do so, run

python3 npy2ply.py <source_file.npy> <output_file.ply>

The output .ply file should now exist, and can be visualized with many different 
online viewers. My favorite is the following: https://niujinshuchong.github.io/mip-splatting-demo/
Just upload the file, mess with the camera, and voila!

