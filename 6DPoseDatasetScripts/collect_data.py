from pathlib import Path



if __name__ =="__main__":

    path = input("Add path for dataset: ")

    path = Path(path).resolve()

    ans = input(f"Saving dataset in {path}? (y/n) ")

    if ans != "y":
        print("exiting ...")
        exit()
    
    input("Start rosbag and press enter to start recording data ... ")