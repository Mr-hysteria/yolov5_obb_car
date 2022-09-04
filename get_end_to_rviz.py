from math import *

"""
两个角度需要对应,只需要确定一次即可，因为变化是固定的
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# 位置1------原点
euler_base_to_rviz = [30, 0, -90]  # rviz默认打开位置，只不过末端弧度为1.05
euler_base_to_end = [-90, 60, 0]  # end实际转到的位置

# 位置2
# euler_base_to_rviz = [-90, 0.000, 0]
# euler_base_to_end = [ 90,0,-90]

# 位置3(rviz end都转动60)
# euler_base_to_rviz = [-149.793, 0.000, -90.053]
# euler_base_to_end = [90, -60, 179.70]


ret1 = R.from_euler('xyz', euler_base_to_rviz, degrees=True)
ret11 = R.from_euler('xyz', euler_base_to_end, degrees=True)

A = ret1.as_matrix()
A = np.matrix(A)
B = ret11.as_matrix()
B = np.matrix(B)

print("end_to_rviz:", B.I @ A)
