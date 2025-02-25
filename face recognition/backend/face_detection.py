import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def detect_faces(frame):
    face_boxes = []
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w, h = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                              int(bboxC.width * w), int(bboxC.height * h))

                face_boxes.append({"x": x, "y": y, "w": w, "h": h})
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, face_boxes
