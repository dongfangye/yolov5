# 鼠标跟踪
import cv2
import torch
import pyautogui
import numpy as np
import time

# 加载本地训练好的YOLOv5模型
model_path = 'weights/yolov5s.pt'  # 替换为你的本地模型路径
model = torch.hub.load('E:/Software/PythonProject/yolo/yolov5-6.0', 'custom', path=model_path, source='local')

# 打开视频流（0表示摄像头）
cap = cv2.VideoCapture(0)

def get_face_center(box):
    """获取边界框的中心点"""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy

# 获取视频窗口的初始位置
cv2.namedWindow("Video")
cv2.moveWindow("Video", 100, 100)  # 假设视频窗口位于屏幕左上角 (100, 100)
time.sleep(1)  # 等待视频窗口位置稳定

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv5进行检测
    results = model(frame)

    # 解析检测结果
    boxes = results.xyxy[0].cpu().numpy()
    faces = []
    confidences = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0:  # 仅处理类别为0（人类）的检测
            faces.append((x1, y1, x2, y2))
            confidences.append(conf)

    # 如果检测到人脸，找到置信度最高的人脸
    if faces:
        max_conf_idx = np.argmax(confidences)
        best_face = faces[max_conf_idx]
        cx, cy = get_face_center(best_face)

        # 获取视频窗口在屏幕上的位置
        window_name = "Video"
        rect = cv2.getWindowImageRect(window_name)
        win_x, win_y, win_w, win_h = rect

        # 计算鼠标应该移动到的屏幕坐标
        screen_x = win_x + int(cx * win_w / frame.shape[1])
        screen_y = win_y + int(cy * win_h / frame.shape[0])

        pyautogui.moveTo(screen_x, screen_y)

        # 在视频帧中绘制边界框和中心点
        x1, y1, x2, y2 = map(int, best_face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # 显示视频帧
    cv2.imshow(window_name, frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
