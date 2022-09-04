"""
本程序完成了三个步骤的内容：
1.目标识别
2.坐标转换
3.运动控制(第二个线程）
"""
import numbers
import socket
import time

from main_detect_lib import *
import threading
import os



# erobot.power_on()
# time.sleep(1)
# erobot.state_on()
# 
# localhost = '/home/nvidia/uds_socket'
# s = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
# s.settimeout(500)
# if os.path.exists(localhost):
#     os.unlink(localhost)
# s.bind(localhost)
# s.listen(1)

# while True:
#     print("waiting signal")
#     conn, addr = s.accept()
#     data =conn.recv(10000)
#     if data == b"begin":
#         break




erobot.set_angles(initA, 1080)
erobot.set_digital_out(1, 1)
erobot.set_digital_out(0, 0) 
# 1.先把机器人运行到初始位置,且关闭夹爪

# 2.yolo开始检测，并等待倒计时结束
print("------启动目标检测算法------\n")
t2 = threading.Thread(target=realsense_detect) # Press esc or 'q' to close the Thread
t2.start()
mytime = 16
while mytime > 0:
    print("校准倒计时：", mytime)
    mytime -=1
    time.sleep(1)


# 3.开始校准    
print("------机器人开始校准，ESC或者q可以退出校准------\n")
move_1()
t2.join()


# 4.旋转夹爪
print("------旋转夹爪-----")
move_3()

# 打开夹爪,先把两者电位复位，pin0高点平打开，pin1高电平关闭
time.sleep(2)
erobot.set_digital_out(1, 0)
erobot.set_digital_out(0, 1)
move_2()
time.sleep(2)  # 停稳

erobot.set_digital_out(0, 0)
erobot.set_digital_out(1, 1)
time.sleep(2)

#  路点1
pose_now = erobot.get_coords()
pose_now[2] = pose_now[2] + 20
erobot.set_coords(pose_now, 2000)
time.sleep(1)
erobot.wait_command_done()

# 路点2
erobot.set_angles([0, -30, -60, -90, -90, 60], 720)
time.sleep(1)
erobot.wait_command_done()

# 准备打开夹爪
mytime2 = 5
while mytime2>0:
    print("准备放下杯子，倒计时：",mytime2)
    mytime2-=1
    time.sleep(1)

erobot.set_digital_out(1, 0)
erobot.set_digital_out(0, 1)
time.sleep(2)




# 路点3
erobot.set_angles([0, -90, 0, -90, -90, 60],720)
erobot.wait_command_done()

# 发送socket
# conn.send(b"ok\r\n")
# conn.close()
# s.close()

# 下降路点
# a = [51.001237, 6.070218, 38.357388, -223.945312, 39.199219, 59.941406]
# erobot.set_angles(a,720)
# erobot.wait_command_done()
# time.sleep(2)
# erobot.set_digital_out(1,0)
# erobot.set_digital_out(0,1)
