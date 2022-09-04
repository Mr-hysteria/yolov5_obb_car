"""
这个文件是抓取两次，一次是倾斜，一次是平躺
"""

from concurrent.futures import thread
import os
import threading
import pyrealsense2 as rs
import random
import time
from scipy.spatial.transform import Rotation as R
from elephant import elephant_command
from global_v import *
import socket
import cv2
import numpy as np
import torch
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import torch.backends.cudnn as cudnn
import os
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords,
                           scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly

# 初始化重要全局变量
camera_coordinate_4d = [0, 0, 0, 1]  # 目标点在相机坐标系下的描述，单位是米
base_coordinate_4d = [0, 0, 0, 1]  # 齐次式
euler = euler_g
end_to_rviz = end_to_rviz_g
initP = initP_g
initA = initA_g
erobot = elephant_command()
key = 0
angle = 0
pianyi = 0


# socket new
def server():
    global mutex
    global send_flag1
    global send_flag2
    global send_flag3
    global grasp_flag
    global loose_flag
    global in_place_flag

    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind(('127.0.0.1', 8899))
    serv.listen(2)
    while True:
        conn, addr = serv.accept()
        print(conn, addr)
        while True:
            msg = conn.recv(1024)
            if msg:
                print(msg.decode())
                mutex.acquire()
                if msg.decode() == 'ready':
                    print('moving ...')
                    send_flag1 = True
                    mutex.release()
                    while True:
                        if in_place_flag == True:
                            mutex.acquire()
                            conn.send(bytes('in place', 'utf8'))
                            print('send: in place')
                            in_place_flag = False
                            mutex.release()
                            break
                elif msg.decode() == 'grasp':
                    print('grasping ...')
                    send_flag2 = True
                    mutex.release()
                    while True:
                        if grasp_flag == True:
                            mutex.acquire()
                            conn.send(bytes('succeed', 'utf8'))
                            print('send: succeed')
                            grasp_flag = False
                            mutex.release()
                            break
                elif msg.decode() == 'loose':
                    print('loosing ...')
                    send_flag3 = True
                    mutex.release()
                    while True:
                        if loose_flag == True:
                            mutex.acquire()
                            conn.send(bytes('placed', 'utf8'))
                            print('send: placed')
                            loose_flag = False
                            mutex.release()
                            break


#############################
mutex = threading.Lock()
send_flag1 = False
send_flag2 = False
send_flag3 = False
grasp_flag = False
loose_flag = False
in_place_flag = False
thread_server = threading.Thread(target=server)
thread_server.start()

##############################


# 初始化yolodetectLoad model
weights = '/home/nvidia/yolov5_obb_car/weights/best.pt'
device = select_device()
model = DetectMultiBackend(weights, device=device)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size((640, 640), s=stride)  # check image size
# Half
half = False
half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt or jit:
    model.model.half() if half else model.model.float()
# Run inference
model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup


# 进行平面移动,偏移量逐渐接近测得值，因为开始会误差大
def move_1():
    global camera_coordinate_4d
    global erobot
    # while True:
    #     if erobot.check_running():  # 检查是否还在运行，停稳后再运动防止跳变
    #         time.sleep(1)
    #         continue
    robot_pos = erobot.get_coords()
    A = camera_coordinate_4d
    # position = [robot_pos[0], robot_pos[1] + A[0], robot_pos[2] - A[1]-15, initP[3], initP[4], initP[5]]  # 桌子上
    # 用实时的末尾坐标，防止180度突变,停稳后一般不会突变
    # position = [robot_pos[0] + A[0], robot_pos[1], robot_pos[2] - A[1] - 10, robot_pos[3], robot_pos[4],
    #             robot_pos[5]]  # 小车上
    position = [robot_pos[0], robot_pos[1] - A[0], robot_pos[2] - A[1] - 5, robot_pos[3], robot_pos[4],
                robot_pos[5]]  # 小车上2

    # print('末端需要移动到的位置：\n', position)
    erobot.set_coords(position, 2000)
    # time.sleep(8)  # 此处不能用wait_command_done()，不然后台会被冻结。时间增加到8s，使机械臂完全停稳，防止跳变
    # time.sleep(4)
    # print("如果调整好则按ESC或者q")
    # global key
    # Press esc or 'q' to close the image window
    # if key & 0xFF == ord('q') or key == 27:
    global pianyi
    pianyi = angle - 89
    # cv2.destroyAllWindows()
    # break


def move_3():
    global erobot
    global pianyi
    robot_pos = erobot.get_angles()
    robot_pos[5] += pianyi  # 仅旋转关节6即可
    erobot.set_angles(robot_pos, 1080)
    time.sleep(2)
    erobot.wait_command_done()


def move_2():
    global base_coordinate_4d
    global erobot
    while True:
        A = base_coordinate_4d
        pos = erobot.get_coords()
        # position = [A[0] + 110, A[1], A[2], pos[3], pos[4], pos[5]]  # 桌子上
        # position = [A[0], A[1] - 110, A[2], pos[3], pos[4], pos[5]]  # 小车上
        if pos[5] > 70:
            position = [A[0] - 110, A[1], A[2], pos[3], pos[4], pos[5]]
        else:
            position = [A[0] - 110, A[1], A[2], pos[3], pos[4], pos[5]]  # 小车上2
        erobot.set_coords(position, 2000)
        judge = erobot.wait_command_done()
        time.sleep(1)
        if judge == b'wait_command_done:0':
            break


def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = [0]
    mid_pos = [(box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2]  # 确定中心点
    min_val = min(abs(box[0][0] - box[1][0]), abs(box[0][1] - box[3][1]))  # 以中心点为中心，确定深度搜索范围（方框宽度最小值）
    # randnum是为了多取一些值来取平均
    for i in range(randnum):
        bias = random.randint(-min_val // 8, min_val // 8)  # 随机偏差,控制被除数大小即可控制范围
        dist = depth_data.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))  # 单位为m
        # dist = depth_data[int(mid_pos[0] + bias), int(mid_pos[1] + bias)]

        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
    return np.mean(distance_list)


# 这个函数主要是在原图上画框,标出深度信息,此外用毫米看比较方便
def dectshow(org_img, boxs, depth_data, intrin, label):
    img = org_img
    global camera_coordinate_4d
    if label == 'no object':
        cv2.putText(img, 'no_object',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        for box in boxs:
            dist = get_mid_pos(org_img, box, depth_data, 24)  # dist单位m，后续统一表示为mm
            camera_coordinate_3d = rs.rs2_deproject_pixel_to_point(intrin=intrin,
                                                                   pixel=[(box[0][0] + box[2][0]) // 2,
                                                                          (box[0][1] + box[2][1]) // 2],
                                                                   depth=dist)  # 单位为m

            if np.isnan(camera_coordinate_3d[0]):
                camera_coordinate_4d = [0, 0, 0, 1]  # 需要使坐标为0,不然机械手会一直走
                cv2.putText(img, 'error_distance',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break
            camera_coordinate_4d = np.array(
                [camera_coordinate_3d[0] * 1000, camera_coordinate_3d[1] * 1000, camera_coordinate_3d[2] * 1000, 1])
            global base_coordinate_4d
            base_coordinate_4d1 = trans_base_to_cam(camera_coordinate_4d)
            base_coordinate_4d1 = np.array(base_coordinate_4d1)  # 需要array对象。有两层括号，因此需要提取[0]
            base_coordinate_4d = base_coordinate_4d1[0]
            global angle
            angle = np.rad2deg(np.arctan2(box[0][1] - box[3][1], box[0][0] - box[3][0]))
            if angle < 0:  # 统一到[0,180]
                angle += 180

            # print('像素坐标系：', [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2, dist * 1000])
            # print('相机坐标系：',
            #       [camera_coordinate_3d[0] * 1000, camera_coordinate_3d[1] * 1000, camera_coordinate_3d[2] * 1000])
            # global euler
            # print("rviz坐标系：", rviz_to_cam_f(euler) @ camera_coordinate_4d)
            # global end_to_rviz
            # print("end坐标系：", end_to_rviz @ rviz_to_cam_f(euler) @ camera_coordinate_4d)
            # global erobot
            # print("机器人末端位置：", erobot.get_coords())
            # print("base_coordinate_4d:", base_coordinate_4d)
            # print("坐标0",box[0])
            # print("坐标1",box[1])
            # print("坐标2",box[2])
            # print("坐标3",box[3])
            # print("----------------------------------\n")

            # text_pixel = str((int(box[0][0]) + int(box[2][0])) // 2) + ', ' + str(
            #     (int(box[0][1]) + int(box[2][1])) // 2) + '(pixel)'
            text_camera = str(camera_coordinate_3d[0] * 1000)[:4] + ', ' + str(camera_coordinate_3d[1] * 1000)[
                                                                           :4] + '(mm)'
            text_angel = str(angle)[:4] + 'degree'

            # rectangle画框，参数表示依次为：(图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细)
            # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # circle画圆心，参数表示依次为：(img, center, radius, color[, thickness]),thickness为负表示绘制实心圆
            center = [(box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2]
            cv2.circle(img, center, 8, (0, 255, 0), -1)
            # first_point = [int(box[0][0]), int(box[0][1])]
            # second_point = [int(box[1][0]), int(box[1][1])]
            # cv2.circle(img, first_point, 8, (0, 255, 0), -1)
            # cv2.circle(img, second_point, 8, (0, 0, 255), -1)
            cv2.drawContours(img, contours=[box], contourIdx=-1, color=(0, 255, 0), thickness=2)

            # putText各参数依次是：图片，添加的文字(标签+深度-单位m)，左上角坐标，字体，字体大小，颜色，字体粗细
            cv2.putText(img, 'cup_distance:' + str(dist * 1000)[:4] + 'mm',
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(img, 'cup_position_in_pixel:' + text_pixel,
            #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, 'cup_position_in_camera:' + text_camera,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, 'angle_in_camera:' + text_angel,
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(img, 'cup_position_in_base:' + text_base, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('dec_img', img)


def realsense_detect():  # 进行目标识别，显示目标识别效果，返回相机坐标系下的值
    # 配置yolov5
    # model = torch.hub.load('/home/desktop-tjj/yolov5', 'custom', path='/home/desktop-tjj/yolov5_obb/weights/best.pt',
    #                        source='local')  # 加载模型
    # model.eval()
    # 配置摄像机参数，用opencv的时候，颜色通道是bgr
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 深度通道的分辨率最大为1280x720
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # 开始采集
    pipeline.start(config)
    # 深度与彩色图像对齐
    alignIt = rs.align(rs.stream.color)
    num = 1
    many_time = 1
    try:
        while True:
            many_time += 1
            # 获取深度图以及彩色图像
            frames = pipeline.wait_for_frames()

            # 获取相机内参
            if num == 1:
                color_frame = frames.get_color_frame()
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                num += 1

            aligned_frame = alignIt.process(frames)  # 获取对齐数据
            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转化成numpy格式

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 需要输入BGR格式
            boxs, label = yolo_run(color_image)
            dectshow(color_image, boxs, depth_frame, intr, label)  # 用的是depth_frame(没有转化成np格式的)，因为要调用get_distance
            if many_time % 15 == 0 or many_time == 2:
                print("capture_frame:", many_time)
                move_1()
            # 当获取为nan的时候，虽然跳过了，但是还是会报错

            # # 可选，展示彩色图像和深度图
            # # 在深度图像上应用colormap(图像必须先转换为每像素8位)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # # 水平堆叠深度图和彩色图
            # images = np.hstack((color_image, depth_colormap))
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            global key
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                # if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


def rviz_to_cam_f(euler):  # 把欧拉角转化成4x4变换矩阵(rviz to cam)
    x = euler[0] * 1000
    y = euler[1] * 1000
    z = euler[2] * 1000
    r = [euler[3], euler[4], euler[5]]
    # 欧拉角转旋转矩阵
    r4 = R.from_euler('xyz', r, degrees=True)
    r4 = r4.as_matrix()
    t = np.array([[x], [y], [z]])
    RT1 = np.column_stack([r4, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    RT1 = np.matrix(RT1)  # array转化成matrix，方便求逆矩阵
    return RT1


def trans_base_to_end():
    global erobot
    end_pose = erobot.get_coords()
    # print("目前机械臂坐标位置(mm)：\n", end_pose)
    # 得到base to end的转化矩阵
    end = [end_pose[3], end_pose[4], end_pose[5]]
    ret = R.from_euler('xyz', end, degrees=True)
    r = ret.as_matrix()
    t = [end_pose[0], end_pose[1], end_pose[2]]
    rt = np.column_stack([r, t])  # 列合并
    rt = np.row_stack((rt, np.array([0, 0, 0, 1])))
    rt = np.matrix(rt)  # array转化成matrix，方便求逆矩阵
    return rt


def trans_base_to_cam(camera_coordinate):  # 把相机坐标系下的坐标转化为base坐标系下的坐标，根据链式法则求
    global euler
    rviz_to_cam = rviz_to_cam_f(euler)  # 得到矩阵1
    global end_to_rviz
    end_to_rviz1 = np.matrix(end_to_rviz)
    base_to_end = trans_base_to_end()
    return base_to_end @ end_to_rviz1 @ rviz_to_cam @ camera_coordinate


@torch.no_grad()
def yolo_run(imy,  # 不进行初始化
             # weights=ROOT / 'weights/best.pt',  # model.pt path(s)
             weights='/home/desktop-tjj/yolov5_obb/weights/best.pt',  # model.pt path(s)
             # imgsz=(640, 640),  # inference size (height, width)
             conf_thres=0.3,  # confidence threshold
             iou_thres=0.4,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             # device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             view_img=False,  # show results
             save_txt=False,  # save results to *.txt
             save_conf=False,  # save confidences in --save-txt labels
             save_crop=False,  # save cropped prediction boxes
             nosave=False,  # do not save images/videos
             classes=None,  # filter by class: --class 0, or --class 0 2 3
             agnostic_nms=False,  # class-agnostic NMS
             augment=False,  # augmented inference
             visualize=False,  # visualize features
             update=False,  # update all models
             name='exp',  # save results to project/name
             exist_ok=False,  # existing project/name ok, do not increment
             line_thickness=3,  # bounding box thickness (pixels)
             hide_labels=False,  # hide labels
             hide_conf=False,  # hide confidences
             # half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             ):
    # Load model
    # device = select_device(device)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn)
    # stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    # imgsz = check_img_size(imgsz, s=stride)  # check image size

    # # Half
    # half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    # if pt or jit:
    #     model.model.half() if half else model.model.float()

    # # Run inference
    # model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # for遍历图片
    imy_copy = imy
    im = letterbox(imy_copy)[0]
    im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    # pred: list*(n, [cxcylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
    pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True,
                                   max_det=max_det)
    dt[2] += time_sync() - t3

    # Process predictions
    for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5])  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        seen += 1
        annotator = Annotator(im, line_width=line_thickness, example=str(names))

        if len(det):
            # Rescale polys from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            pred_poly = scale_polys(im.shape[2:], pred_poly, imy_copy.shape)
            det = torch.cat((pred_poly, det[:, -2:]), dim=1)  # (n, [poly conf cls])

            # Write results
            polygon_list_all = []
            for *poly, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                poly = [x.cpu().numpy() for x in poly]
                polygon_list = np.array([(poly[0], poly[1]), (poly[2], poly[3]), \
                                         (poly[4], poly[5]), (poly[6], poly[7])], np.int32)
                # polygon_list = [(poly[0], poly[1]), (poly[2], poly[3]), \
                #                          (poly[4], poly[5]), (poly[6], poly[7])]
                polygon_list_all.append(polygon_list)
                # annotator.box_label(xyxy, label, color=colors(c, True))
                # result = annotator.poly_label_return(poly, label, color=colors(c, True))
            return polygon_list_all, label
            break
        else:
            aa = np.array([(0, 0), (0, 0), (0, 0), (0, 0)], np.int32)
            polygon_list_all = []
            polygon_list_all.append(aa)
            return polygon_list_all, 'no object'


# erobot.power_on()
# time.sleep(2)
# erobot.state_on()
'''
#socket进行监听
localhost = '/home/nvidia/uds_socket'
s = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
s.settimeout(500)
if os.path.exists(localhost):
    os.unlink(localhost)
s.bind(localhost)
s.listen(1)

while True:
    print("waiting signal")
    conn, addr = s.accept()
    data =conn.recv(10000)
    if data == b"begin":
        break
'''

numbers1 = 2
while numbers1 >= 1:
    # 1.先把机器人运行到初始位置,且关闭夹爪
    print('waiting for notification ...')
    while True:
        if send_flag1 == True:
            break
    mutex.acquire()
    erobot.set_angles(initA, 1080)
    erobot.set_digital_out(1, 1)
    erobot.set_digital_out(0, 0)
    mytime = 10
    while mytime > 0:
        print("校准倒计时：", mytime)
        mytime -= 1
        time.sleep(1)
    in_place_flag = True
    send_flag1 = False
    mutex.release()
    # 2.yolo开始检测，并等待倒计时结束
    while True:
        if send_flag2 == True:
            break
    mutex.acquire()
    print("------启动目标检测算法------\n")
    realsense_detect()

    # 4.旋转夹爪
    print("------旋转夹爪-----")
    move_3()

    # 打开夹爪,先把两者电位复位，pin0高点平打开，pin1高电平关闭
    erobot.set_digital_out(1, 0)
    erobot.set_digital_out(0, 1)
    move_2()
    time.sleep(2)  # 停稳

    erobot.set_digital_out(0, 0)
    erobot.set_digital_out(1, 1)
    time.sleep(2)

    #  路点1
    pose_now = erobot.get_coords()
    pose_now[2] = pose_now[2] + 15
    erobot.set_coords(pose_now, 2000)
    time.sleep(1)
    erobot.wait_command_done()

    # 路点2
    # erobot.set_angles([0, -30, -60, -90, -90, 60], 1080) yuanlaide
    erobot.set_angles([0, -90, 0, -90, -90, 60], 1080)
    time.sleep(1)
    erobot.wait_command_done()

    grasp_flag = True
    send_flag2 = False
    mutex.release()

    # 准备打开夹爪
    print('准备打开夹爪')
    while True:
        if send_flag3 == True:
            break
    mutex.acquire()

    # fangzhi
    if numbers1 == 2:
        erobot.set_angles([0, -114.4, 90, -155.3, -90, 60], 1080)
        time.sleep(1)
        erobot.wait_command_done()
    elif numbers1 == 1:
        erobot.set_angles([-82, -113.2, 90, -152.8, -7.2, 56.6], 1080)
        time.sleep(1)
        erobot.wait_command_done()

    mytime2 = 2
    while mytime2 > 0:
        print("准备放下杯子，倒计时：", mytime2)
        mytime2 -= 1
        time.sleep(1)

    erobot.set_digital_out(1, 0)
    erobot.set_digital_out(0, 1)
    time.sleep(2)
    numbers1 -= 1

    erobot.set_angles([0, -90, 0, -90, -90, 60], 1080)
    time.sleep(1)
    erobot.wait_command_done()

    loose_flag = True
    send_flag3 = False
    mutex.release()

print("ESC退出程序")
# 路点3
erobot.set_angles([0, -90, 0, -90, -90, 60], 720)
erobot.wait_command_done()
thread_server.join()
# 发送socket
# conn.send(b"ok\r\n")
# conn.close()
# s.close()
