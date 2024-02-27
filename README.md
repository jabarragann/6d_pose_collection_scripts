# Synthetic data generation of surgical instruments' 6D pose datasets​ 

Package to generate 6D pose datasets of surgical instruments. The package is based on the [Surgical Robotic challenge][SRC-github] simulation environment and the [BOP toolkit][BOP-github].

<p align="center">
<img src="./docs/imgs/main_fig.png" width="700">
</p>


**Main functionalities:**

1. Replaying of pre-recorded trajectories (See [simple_replay.py](./scripts/simple_replay.py)) 
2. Recording of  6D pose datasets in the BOP format (See [collect_data.py](./scripts/collect_data.py)).
3. Reader class for datasets generated in BOP format (See [BopDatasetReader](./ambf6dpose/DataCollection/BOPSaver/BopReader.py)).

# Getting started

The following code base will require Ubuntu 20.04, ROS noetic, [AMBF][ambf-github], and the [surgical robotics challenge][SRC-github] simulation environment. After obtaining all dependencies, first install `ambf6dpose` package with:

```bash
pip install -e .
```

and then install additional dependencies with:

```bash
pip install -r requirements/requirements.txt
sudo apt install ros-noetic-ros-numpy
```

## Data generation  

To generated data, you will first need to open the surgical robotics challenge scene and then replay motions with the `simple_replay.py` script. While the motions are being replayed, you can collect data with the `collect_data.py` script. For more information about to each script refer see below.


**Replay instrumention motion**

```bash
python scripts/simple_replay.py single-replay --help
```

```
Usage: simple_replay.py single-replay [OPTIONS]

Options:
  --bag_path PATH            Path to bag file
  --percent_to_replay FLOAT  Path to bag file
  --ecm_pos FLOAT LIST       ECM joint position, specifid as a string of 4
                             float separated with a space,     e.g., '1.0 1.0
                             1.0 1.0'. If not provide current camera pose will
                             be used.
  -r                         Record images
  -o, --output_p DIRECTORY   Only required if record flag (-r) is set.
  --help                     Show this message and exit.
```

Sample trajectories are available upon request.

**Collection of image and pose data**
```bash
python scripts/collect_data.py --help
```

```bash
Usage: collect_data.py [OPTIONS]

  6D pose data collection script. Instructions: (1) Run ambf simulation (2)
  run recorded motions (3) run collection script.

Options:
  --path TEXT          Path to save dataset  [required]
  --sample_time FLOAT  Sample every n seconds
  --help               Show this message and exit.
```


## Troubleshooting:

<details>
<summary> Ros sync client did not received any data/timeout exceptions </summary>
<br>
Ros topic for images are hardcoded on the [Rostopics.py](./ambf6dpose/DataCollection/Rostopics.py). If you are using a different topic names, the ROS sync client will not generate any data to be saved. In particular, check if the toolpitchlink state for PSM1 and PSM2 are being published. These are not published by default on the simulation environment

```
/ambf/env/psm1/toolpitchlink/Command
/ambf/env/psm2/toolpitchlink/Command
```

To start publishing change the toolpitchlink BODY `passive flag` to `false` in the [PSM ADF files](https://github.com/surgical-robotics-ai/surgical_robotics_challenge/blob/eb82bdea8a10550b8dfad6fc9f8dd8002c6ad925/ADF/psm1.yaml#L415). You will have to do this for both PSMs.

</details>

<details>
<summary> Incompatibility issues with numpy greater than 1.20 </summary>

<br>
If you find incompatibility issues with your numpy version, you will probably need to modify some source files of <code>ros_numpy</code> to remove the numpy deprecated attributes. Replace <code>np.float</code> for <code>float</code> at line 224 of <code>point_cloud2.py</code>.  

<br>

``` bash
File "/opt/ros/noetic/lib/python3/dist-packages/ros_numpy/point_cloud2.py", li ne 224 , in def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
File "/home/jin/.local/lib/python3.8/site-packages/numpy/init.py", line 30 5 , in _getattr

raise AttributeError(former_attrs[attr])
AttributeError: module 'numpy' has no attribute 'float'.
'np.float" was a deprecated alias for the builtin 'float'. To avoid this error 
In existing code, use 'float' by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use 'np.float64" here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidan ce see the original release note at:
https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

</details>


# Citation
If you find this work useful, please cite it as:

```bibtex

```

[//]: # (Important resources)

[SRC-github]: https://github.com/surgical-robotics-ai/surgical_robotics_challenge
[BOP-github]: https://github.com/thodan/bop_toolkit 
[ambf-github]: https://github.com/WPI-AIM/ambf/tree/ambf-2.0
