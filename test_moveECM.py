import time
import os
import sys
from glob import glob
import gc
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.utils import coordinate_frames
dynamic_path = os.path.abspath(__file__ + "/../")
# data_path = os.path.abspath(__file__+"/../../../../")
# print(dynamic_path)
sys.path.append(dynamic_path)


if __name__ == '__main__':
    simulation_manager = SimulationManager('ECM_move_test')
    time.sleep(0.5)
    w = simulation_manager.get_world_handle()
    time.sleep(0.2)
    w.reset_bodies()
    time.sleep(0.2)
    cam = ECM(simulation_manager, 'CameraFrame') # connect ECM
    # cam.servo_jp([0.0, 0.05, -0.01, 0.0])
    time.sleep(0.5)
    psm1 = PSM(simulation_manager, 'psm1', add_joint_errors=False) # connect PSM1
    time.sleep(0.5)
    psm2 = PSM(simulation_manager, 'psm2', add_joint_errors=False) # connect PSM2
    time.sleep(0.5)
    ### init psm poses
    psm1_jp = [0.28911492608790323, -0.22020479854735953, 0.14256010791068965, -1.3629036443829619, 0.591057459169165,
               -0.8305143685566997]

    psm2_jp = [-0.3287843223486435, -0.20381993113387006, 0.1382636372893194, -0.3342536922764675, 0.5989012506373477,
               -0.09461173139535281]
    psm1.servo_jp(psm1_jp)
    psm2.servo_jp(psm2_jp)

    ##### joint directions
    # x: left and right
    # y: up and down
    # z : rotation along the point of view axis

    # 1s joint: x movements, left -, right +
    # 2nd joint: y movements, down +, up -
    # 3rd joint: zoom in or out, out -, in +
    # 4th joint: z rotation, cw +, ccw -

    # cam.servo_jp([0.0, 0.05, -0.01, 0.0]) # the secret offset
    cam.servo_jp([0.0, 0.05, -0.01, 0.5])
    T = cam.measured_cp()
    print(T)
