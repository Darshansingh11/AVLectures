# Steps to extract features

**Step 1:**

Execute the `create_feature_csv.py` program. To specify the path of the Datasubset use the `--base_dir` optional argument. The default path of the DataSubset is `/ssd_scratch/cvit/AVLectures/DataSubset`. After executing this program the following will be created inside `base_dir`.

a. *input_2d.csv*

b. *input_3d.csv*

c. Also empty directories called *features, features/2d/, features/3d/* will be created.

**Step 2:**

Once we have the 2d, 3d CSV files and empty directories to store 2d & 3d features, our next task is to extract the 2d and 3d features from the videos using the `extract.py` program.
First extract the 2d features using the following command:
```
python extract.py --csv=input_2d.csv --type=2d --batch_size=64 --num_decoding_thread=4
```
Then download the 3D ResNext-101 model as follows (for 3d feature extraction):

```
mkdir model
$ cd model
$ wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth
```
Now extract the 3d features using the following command:
```
$ python extract.py --csv=input_3d.csv --type=3d --batch_size=64 --num_decoding_thread=4
```

**Step 3:**

Now it is time to create the pickle file of our data. To do this execute the `create_pickle.py` program. To specify the path of the Datasubset use the `--base_dir` optional argument. The default path of the DataSubset is `/ssd_scratch/cvit/AVLectures/DataSubset`. After executing this program a pickle file called `avl.pkl` will be created inside the `base_dir`.

# Fast and Easy to use video feature extractor

This repo aims at providing an easy to use and efficient code for extracting
video features using deep CNN (2D or 3D).

It has been originally designed to extract video features for the large scale video dataset HowTo100M (https://www.di.ens.fr/willow/research/howto100m/) in an efficient manner.


Most of the time, extracting CNN features from video is cumbersome.
In fact, this usually requires dumping video frames into the disk, loading the dumped frames one
by one, pre processing them and use a CNN to extract features on chunks of videos.
This process is not efficient because of the dumping of frames on disk which is
slow and can use a lot of inodes when working with large dataset of videos.

To avoid having to do that, this repo provides a simple python script for that task: Just provide a list of raw videos and the script will take care of on the fly video decoding (with ffmpeg) and feature extraction using state-of-the-art models. While being fast, it also happen to be very convenient.

This script is also optimized for multi processing GPU feature extraction.


# Requirements
- Python 3
- PyTorch (>= 1.0)
- ffmpeg-python (https://github.com/kkroening/ffmpeg-python)

# How To Use ?

First of all you need to generate a csv containing the list of videos you
want to process. For instance, if you have video1.mp4 and video2.webm to process,
you will need to generate a csv of this form:

```
video_path,feature_path
absolute_path_video1.mp4,absolute_path_of_video1_features.npy
absolute_path_video2.webm,absolute_path_of_video2_features.npy
```

And then just simply run:

```sh
python extract.py --csv=input.csv --type=2d --batch_size=64 --num_decoding_thread=4
```
This command will extract 2d video feature for video1.mp4 (resp. video2.webm) at path_of_video1_features.npy (resp. path_of_video2_features.npy) in
a form of a numpy array.
To get feature from the 3d model instead, just change type argument 2d per 3d.
The parameter --num_decoding_thread will set how many parallel cpu thread are used for the decoding of the videos.

Please note that the script is intended to be run on ONE single GPU only.
if multiple gpu are available, please make sure that only one free GPU is set visible
by the script with the CUDA_VISIBLE_DEVICES variable environnement for example.

# Can I use multiple GPU to speed up feature extraction ?

Yes ! just run the same script with same input csv on another GPU (that can be from a different machine, provided that the disk to output the features is shared between the machines). The script will create a new feature extraction process that will only focus on processing the videos that have not been processed yet, without overlapping with the other extraction process already running.

# What models are implemented ?
So far, only one 2D and one 3D models can be used.

- The 2D model is the pytorch model zoo ResNet-152 pretrained on ImageNet. The 2D features are extracted at 1 feature per second at the resolution of 224.
- The 3D model is a ResNexT-101 16 frames (https://github.com/kenshohara/3D-ResNets-PyTorch) pretrained on Kinetics. The 3D features are extracted at 1.5 feature per second at the resolution of 112.

# Downloading pretrained models
This will download the pretrained 3D ResNext-101 model we used from: https://github.com/kenshohara/3D-ResNets-PyTorch 

```sh
mkdir model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth
```



# Acknowledgements
The code re-used code from https://github.com/kenshohara/3D-ResNets-PyTorch
for 3D CNN.
