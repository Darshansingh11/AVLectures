Step 1:
Execute the "create_feature_csv.py" program. To specify the path of the Datasubset use the '--base_dir' optional argument. The default path of the DataSubset is '/ssd_scratch/cvit/AVLectures/DataSubset'. After executing this program the following will be created inside base_dir.
a. input_2d.csv   
b. input_3d.csv
c. Also empty directories called "features", "features/2d/", "features/3d/" will be created.

Step 2:
Once we have the 2d, 3d CSV files and empty directories to store 2d & 3d features, our next task is to extract the 2d and 3d features from the videos using the "extract.py" program.
First extract the 2d features using the following command:
$ python extract.py --csv=input_2d.csv --type=2d --batch_size=64 --num_decoding_thread=4
Then download the 3D ResNext-101 model as follows (for 3d feature extraction):
$ mkdir model
$ cd model
$ wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth
Now extract the 3d features using the following command:
$ python extract.py --csv=input_3d.csv --type=3d --batch_size=64 --num_decoding_thread=4

Step 3:
Now it is time to create the pickle file of our data. To do this execute the "create_pickle.py" program. To specify the path of the Datasubset use the '--base_dir' optional argument. The default path of the DataSubset is '/ssd_scratch/cvit/AVLectures/DataSubset'. After executing this program a pickle file called 'avl.pkl' will be created inside base_dir.
