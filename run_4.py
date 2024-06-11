# 函数目的：yolov5 + opencv + 人脸识别对比
# 预处理对比库图片特征信息，避免重复提取
# 加入多线程，避免阻塞
import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pickle
import cv2
import threading
import queue

# 定义图像特征提取模型（如 ResNet50）
feature_extractor = models.resnet50(pretrained=False)
feature_extractor.load_state_dict(torch.load('resnet50.pth'))
feature_extractor.eval()

# 定义图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image).numpy().flatten()
    return features

# 文件夹路径
folder_path = 'mydata'  # 替换为你的文件夹路径

# 提取并保存特征信息
folder_features = {}
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert('RGB')
        features = extract_features(image)
        folder_features[filename] = features

# 保存特征信息到文件
with open('folder_features.pkl', 'wb') as f:
    pickle.dump(folder_features, f)

# 加载本地训练好的YOLOv5模型
model_path = 'weights/yolov5s.pt'  # 替换为你的本地模型路径
model = torch.hub.load('E:/Software/PythonProject/yolo/yolov5-6.0', 'custom', path=model_path, source='local')

# 设定目标标签
target_label = 'person'  # 替换为你想检测的标签
target_label_index = model.names.index(target_label)  # 获取目标标签对应的索引

# 获取摄像头
cap = cv2.VideoCapture(0)

def compare_images(image1_features, image2_features):
    return np.linalg.norm(image1_features - image2_features)

def compare_features_thread(cropped_features, result_queue):
    for filename, features in folder_features.items():
        similarity = compare_images(cropped_features, features)
        if similarity < 0.5:  # 自定义相似度阈值
            result_queue.put(filename)
            return
    result_queue.put(None)

# 加载保存的特征信息
with open('folder_features.pkl', 'rb') as f:
    folder_features = pickle.load(f)

result_queue = queue.Queue()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将帧转换为RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 使用YOLOv5模型进行对象检测
    results = model(img)
    
    # 解析检测结果，并将张量从GPU复制到CPU，再转换为NumPy数组
    labels = results.xyxyn[0][:, -1].cpu().numpy()
    cords = results.xyxyn[0][:, :-1].cpu().numpy()
    
    # 绘制检测结果
    n = len(labels)
    for i in range(n):
        if int(labels[i]) == target_label_index:  # 只处理目标标签
            row = cords[i]
            if row[4] >= 0.2:  # 设置置信度阈值
                x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, model.names[int(labels[i])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                
                # 裁剪检测到的对象
                cropped_img = frame[y1:y2, x1:x2]
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                cropped_features = extract_features(cropped_pil)
                
                # 启动特征对比线程
                threading.Thread(target=compare_features_thread, args=(cropped_features, result_queue)).start()
    
    # 检查对比结果
    try:
        result = result_queue.get_nowait()
        if result:
            cv2.putText(frame, f'Similar to {result}', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    except queue.Empty:
        pass
    
    # 显示结果帧
    cv2.imshow('YOLOv5 Detection', frame)
    
    # 按'Q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
