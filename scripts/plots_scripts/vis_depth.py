import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from ambf6dpose.DataCollection.RosClients import SyncRosInterface
import cv2


if __name__ == "__main__":
    ros_client = SyncRosInterface()

    ros_client.wait_for_data()
    print("Data received")

    # p = "./000007.png"

    # img = Image.open(p)

    img_arr = ros_client.raw_data.camera_l_depth
    img_arr = img_arr.astype(np.uint8)
    print(img_arr.max())
    print(img_arr.min())
    print(img_arr.shape)

    # Vis depth

    # with opencv
    colormap = cv2.applyColorMap(img_arr, cv2.COLORMAP_PLASMA)
    cv2.imshow("Depth Map", colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    seg_img = ros_client.raw_data.camera_l_seg_img
    cv2.imshow("segmented img", seg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./test_record/sup_vid/rgb.png", ros_client.raw_data.camera_l_img)
    cv2.imwrite("./test_record/sup_vid/depth.png", colormap)
    cv2.imwrite("./test_record/sup_vid/seg.png", seg_img)

    # with matplotlib
    # plt.imshow(img_arr, cmap="plasma")
    # plt.show()
