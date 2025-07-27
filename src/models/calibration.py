import numpy as np

class CalibrationModel:
    def __init__(self):
        self.data = []
        self.transformation_matrix = None

    def add_calibration_point(self, screen_pos, gaze_pos):
        self.data.append((screen_pos, gaze_pos))

    def calculate_transformation_matrix(self):
        self.transformation_matrix = np.eye(2)
        return self.transformation_matrix

    def apply_calibration(self, gaze_vector):
        if self.transformation_matrix is not None:
            return np.dot(self.transformation_matrix, gaze_vector)
        return gaze_vector

    def validate_calibration(self):
        return True

    def save_calibration(self, filename):
        pass
