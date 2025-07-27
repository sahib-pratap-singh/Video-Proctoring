import mediapipe as mp
import numpy as np
import cv2

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                faces.append({
                    'box': bbox,
                    'confidence': confidence
                })
        return faces

    def get_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

    def estimate_head_pose(self, landmarks):
        # Dummy implementation: returns 0,0,0
        return {'roll': 0, 'pitch': 0, 'yaw': 0}

    def draw_annotations(self, frame, faces, landmarks):
        # Draw face boxes and landmarks
        return frame
