import time

from elephant import elephant_command

robot = elephant_command()

# 初始位置
initA_g = [-166.083641, -81.562314, 108.679357, -206.630859, -13.886719, 59.765625]   # 第一次的位置(角度)---水平夹爪
robot.set_angles(initA_g, 1080)
# robot.set_angles(initA_g,1080)
print(robot.get_coords())
print(robot.get_angles())

# a = [-117.83900538980294, 238.797797, 460.97518662000573, 89.635103, -60.212089, -179.913677]
# robot.set_coords(a, 1000)
# print(robot.get_coords())

# while True:
#     print(robot.check_running())
#     time.sleep(1)