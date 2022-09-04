#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 大象机器人Socket控制工具包
"""
踩坑:
1.python2与python3发送指令有所差别,在python3中使用时,可能需要在发送的指令后添加.encode
"""

import socket
import time

class elephant_command():
    def __init__(self):
        '''初始化，连接机械臂'''
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = ('192.168.3.5', 5001)  # 机械臂服务器的IP地址和端口
        print("start connect")
        self.sock.connect(self.server_address)
        print("connect success")

    def get_angles(self):
        '''获取当前六个关节角度(°)'''
        message = "get_angles()".encode()
        self.sock.sendall(message)
        angles_str = self.sock.recv(1024)
        # while not angles_str.startswith('get_angles'):
        #     self.sock.sendall(message)
        #     angles_str = self.sock.recv(1024)
        # str to list[float]
        angles = [float(p) for p in angles_str[12:-1].decode().split(',')]
        return angles

    def set_angles(self, angles_array, speed):
        '''设定六个关节的角度(°)和速度(°/min)'''
        ang_msg = "set_angles({},{})".format(','.join(['{:.3f}'.format(x) for x in angles_array]), speed).encode()
        self.sock.sendall(ang_msg)
        back_msg = self.sock.recv(1024)
        print(back_msg)
     
    def set_angle(self, joint, angle, speed):
        '''设定单个关节（joint,1~6）的角度(°)和速度(°/min)'''
        ang_msg = "set_angle(J{},{},{})".format(joint, angle, speed)
        self.sock.sendall(ang_msg)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def get_coords(self):
        '''获取当前末端位姿(mm)'''
        message = "get_coords()".encode()
        self.sock.sendall(message)
        coords_str = self.sock.recv(1024)
        while not coords_str.startswith(b'get_coords'):
            self.sock.sendall(message)
            coords_str = self.sock.recv(1024)
        # str to list[float]
        coords = [float(p) for p in coords_str[12:-1].split(b',')]
        return coords

    def set_coords(self, coords_array, speed):
        '''设定机械臂目标位姿(mm)和运动速度(mm/min)'''
        coords_msg = "set_coords({},{})".format(','.join(['{:.3f}'.format(x) for x in coords_array]), speed)
        # print(coords_msg)
        self.sock.sendall(coords_msg.encode())
        back_msg = self.sock.recv(1024)
        print(back_msg)
        
    def set_coord(self, axis, coord, speed):
        '''设定x,y,z,rx,ry,rz某一方向的坐标(mm)和速度(mm/min)'''
        coord_msg = "set_coord({},{:.3f},{})".format(axis, coord, speed)
        self.sock.sendall(coord_msg)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def jog_coord(self, axis, dirc, speed):
        '''让机械臂沿一轴(axis, x,y,z)方向(dirc, -1负方向,0停止,1正方向)以匀速(mm/min)运动'''
        coord_msg = "jog_coord({},{},{})".format(axis, dirc, speed)
        self.sock.sendall(coord_msg.encode())
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def jog_stop(self, axis):
        '''让机械臂沿一轴(axis, x,y,z,rx,ry,rz,j1~j6)运动停止'''
        coord_msg = "jog_stop({})".format(axis)
        self.sock.sendall(coord_msg.encode())
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def jog_angle(self, joint, dirc, speed):
        '''让机械臂某一关节(joint, 1~6)匀速( / )转动(dirc, -1负方向,0停止,1正方向)'''
        coord_msg = "jog_angle(J{},{},{})".format(joint, dirc, speed)
        self.sock.sendall(coord_msg.encode())
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def task_stop(self):
        '''停止当前任务'''
        message = "task_stop()"
        self.sock.sendall(message.encode())
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def wait(self, seconds):
        '''设定机械臂等待时间(s)'''
        message = "wait({})".format(seconds)
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)
    
    def power_on(self):
        '''给机械臂上电?'''
        message = "power_on()".encode()
        self.sock.sendall(message)
        time.sleep(20)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def power_off(self):
        '''给机械臂断电?'''
        message = "power_off()"
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def get_speed(self):
        '''获取机械臂(末端)速度(mm/s)'''
        message = "get_speed()"
        self.sock.sendall(message)
        speed = self.sock.recv(1024)
        return speed

    def state_check(self):
        '''检查机械臂状态(1正常,0不正常)'''
        message = b"state_check()"
        self.sock.sendall(message)
        state = self.sock.recv(1024)
        return state

    def check_running(self):
        '''检查机械臂是否运行(1正在运行,0不在运行)'''
        message = b"check_running()"
        self.sock.sendall(message)
        running_state = self.sock.recv(1024)
        # print(running_state)
        if running_state == b'check_running:1':
            return True
        else:
            return False

    def set_torque_limit(self, axis, torque):
        '''设置机械臂在x,y,z某一方向上的力矩限制(N)'''
        torque_limit = "set_torque_limit({},{})".format(axis, torque)
        self.sock.sendall(torque_limit)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def set_payload(self, payload):
        '''设置机械臂负载(kg)'''
        message = "set_payload({})".format(payload)
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def set_acceleration(self, acc):
        '''设置机械臂(末端)加速度(整数,mm/s^2)'''
        message = "set_acceleration({})".format(acc)
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def get_acceleration(self):
        '''获取机械臂(末端)加速度(mm/s^2)'''
        message = "get_acceleration()"
        self.sock.sendall(message)
        acc = self.sock.recv(1024)
        return acc

    def wait_command_done(self):
        '''等待命令执行完毕,（该命令会中断其他线程）'''
        message = "wait_command_done()".encode()
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)
        return back_msg

    def pause_program(self):
        '''暂停进程'''
        message = "pause_program()"
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def resume_program(self):
        '''重启已暂停的进程'''
        message = "resume_program()"
        self.sock.sendall(message)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def state_on(self):
        '''机器人使能（使可控）'''
        message = "state_on()".encode()
        self.sock.sendall(message)
        time.sleep(5)
        back_msg = self.sock.recv(1024)
        print(back_msg)
    
    def state_off(self):
        '''机器人去使能（使不可控）'''
        message = "state_off()"
        self.sock.sendall(message)
        time.sleep(5)
        back_msg = self.sock.recv(1024)
        print(back_msg)

    def set_digital_out(self, pin_number, signal):
        """ 设定数字输出端口电平，pin_number:0~15, signal:0低1高"""
        digital_signal = 'set_digital_out({},{})'.format(pin_number, signal)
        self.sock.sendall(digital_signal.encode())
        back_msg = self.sock.recv(1024)
        print(back_msg)

