import cv2
import mediapipe as mp
import numpy as np

# 初始化 Mediapipe 人脸检测与 Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 读取图像
image = cv2.imread("face.png")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 运行 Face Mesh 进行特征提取
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(rgb_image)

# 画出面部关键点
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for idx, landmark in enumerate(face_landmarks.landmark):
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# 显示结果
cv2.imshow("Face Features", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
