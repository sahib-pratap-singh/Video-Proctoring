import unittest
from src.core.face_detector import FaceDetector
from src.core.eye_detector import EyeDetector
from src.core.movement_tracker import MovementTracker

class TestDetection(unittest.TestCase):
    def test_face_detector(self):
        detector = FaceDetector()
        self.assertIsNotNone(detector)

    def test_eye_detector(self):
        detector = EyeDetector()
        self.assertIsNotNone(detector)

    def test_movement_tracker(self):
        tracker = MovementTracker()
        self.assertIsNotNone(tracker)

if __name__ == "__main__":
    unittest.main()
