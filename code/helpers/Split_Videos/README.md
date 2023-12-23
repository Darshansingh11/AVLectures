# Split the videos using the following steps

**Step1**
Execute the `driver.py` program. To specify the path of the Datasubset use the `--base_dir` optional argument. The default path of the DataSubset is `/ssd_scratch/cvit/AVLectures/DataSubset`. To specify the minimum and maximum time of the splits use the `--min_time` and `--max_time` arguments respectively. By default `min_time=7 seconds` and `max_time= 15 seconds`. After executing this program the following will be created inside `base_dir`:
1. Inside each of the course directory a folder called `split_vids` would be created which contains the splits of all the videos.


