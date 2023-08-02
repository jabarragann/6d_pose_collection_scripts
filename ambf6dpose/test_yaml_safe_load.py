import numpy as np
import yaml
from yaml import Loader, Dumper

# Notes: saving the yaml was harder than expected.
# pyyaml don't allow to easily save the array with a specific precision

matrix = np.random.rand(3,3)
print(matrix)

#Dump
with open('stack.yaml', 'w') as f:
    arr_1d = matrix.ravel()

    # option 1 
    # data = {"0000":[{ "cam_R_m2c": arr_1d.tolist()}]}

    # option 2
    # you need to remove square brackets to read the array with np.fromstring (arr[1:-1]])
    str_repr = np.array2string(arr_1d, separator=',',max_line_width=200, precision=8)[1:-1]
    print(str_repr,"hello")
    data = {"0000":[{ "cam_R_m2c": str_repr }]}
    str_repr = np.array2string(arr_1d+0.5, separator=',',max_line_width=200, precision=8)[1:-1]
    data["0001"] = [{ "cam_R_m2c":str_repr}]

    yaml.dump(data, f)

#Load
with open('stack.yaml') as f:
    loaded = yaml.load(f, Loader=Loader)
    # #option 1
    # loaded = loaded["0000"][0]["cam_R_m2c"]

    # Option 2
    loaded = loaded["0000"][0]["cam_R_m2c"]
    loaded = np.fromstring(loaded, sep=',')

loaded = loaded.reshape((3,3))
print(loaded)
print(type(loaded))
print(loaded - matrix)
