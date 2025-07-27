import sys
from src.utils.video_handler import VideoHandler
from src.core.face_detector import FaceDetector
from src.core.eye_detector import EyeDetector
from src.core.movement_tracker import MovementTracker
from src.gui.main_window import ProctoringMainWindow


def main():
    video_handler = VideoHandler()
    face_detector = FaceDetector()
    eye_detector = EyeDetector()
    movement_tracker = MovementTracker()
    app = ProctoringMainWindow()
    app.setup_components(video_handler, face_detector, eye_detector, movement_tracker)
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)