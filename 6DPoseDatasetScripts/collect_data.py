from pathlib import Path
from DatasetBuilder import  SampleSaver
from SimulationInterface import SimulationInterface
import rospy
import time

# Config file
## path
## segmentation camera

if __name__ =="__main__":

    # path = input("Add path for dataset: ")
    path = "./test_d1"

    path = Path(path).resolve()

    ans = input(f"Saving dataset in {path}? (y/n) ")

    if ans != "y":
        print("exiting ...")
        exit()

    sim_interface = SimulationInterface() 
    saver = SampleSaver(root=path)

    input("Start rosbag and press enter to start recording data ... ")

    sample_every = 1.5
    last_time =time.time()+sample_every 

    while(not rospy.is_shutdown()):
        if time.time() - last_time > sample_every:
            sample = sim_interface.generate_dataset_sample()
            saver.save_sample(sample)
            print(f" Saved sample: {time.time()-last_time}")
            last_time = time.time()
    
    saver.close()