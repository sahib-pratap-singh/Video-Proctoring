import cv2
import numpy as np
import time
from collections import deque
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

@dataclass
class EyeMetrics:
    """Container for eye tracking metrics."""
    ear: float  # Eye Aspect Ratio
    gaze_direction: Tuple[float, float]
    pupil_center: Tuple[int, int]
    blink_detected: bool
    movement_magnitude: float
    attention_score: float

class EyeDetector:
    """Enhanced eye detection for proctoring applications."""
    
    # MediaPipe face mesh landmarks for eyes
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Key points for EAR calculation
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # outer, inner corners and vertical points
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    
    def __init__(self, ear_threshold=0.25, blink_frames=3, movement_threshold=10):
        """
        Initialize eye detector with proctoring-specific parameters.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for blink detection
            blink_frames: Consecutive frames below threshold to confirm blink
            movement_threshold: Pixel threshold for significant eye movement
        """
        self.ear_threshold = ear_threshold
        self.blink_frames = blink_frames
        self.movement_threshold = movement_threshold
        
        # Tracking variables
        self.prev_pupils = {'left': None, 'right': None}
        self.blink_counter = 0
        self.total_blinks = 0
        self.ear_history = deque(maxlen=30)  # Store last 1 second at 30fps
        self.gaze_history = deque(maxlen=60)  # Store last 2 seconds
        self.movement_history = deque(maxlen=90)  # Store last 3 seconds
        
        # Proctoring flags
        self.looking_away = False
        self.excessive_blinking = False
        self.suspicious_movement = False
        self.last_blink_time = time.time()
        
        # Calibration data
        self.gaze_center_calibrated = False  # Fix attribute name
        self.gaze_center = (0, 0)
        self.gaze_bounds = {'left': -50, 'right': 50, 'up': -30, 'down': 30}

    def extract_eye_regions(self, landmarks, frame):
        """Extract eye regions with improved preprocessing for pupil detection."""
        h, w = frame.shape[:2]
        eye_regions = {}
        
        for eye_name, eye_points in [('left', self.LEFT_EYE), ('right', self.RIGHT_EYE)]:
            # Get eye landmarks
            eye_landmarks = []
            for point_idx in eye_points:
                if hasattr(landmarks[point_idx], 'x') and hasattr(landmarks[point_idx], 'y'):
                    x = int(landmarks[point_idx].x * w)
                    y = int(landmarks[point_idx].y * h)
                    eye_landmarks.append((x, y))
            
            if len(eye_landmarks) < 6:
                continue
                
            # Create bounding box with padding
            x_coords = [p[0] for p in eye_landmarks]
            y_coords = [p[1] for p in eye_landmarks]
            
            x_min, x_max = min(x_coords) - 10, max(x_coords) + 10
            y_min, y_max = min(y_coords) - 5, max(y_coords) + 5
            
            # Ensure bounds are within frame
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            # Ensure we have 4 values for bbox
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Extract and preprocess eye region
            eye_region = frame[y_min:y_max, x_min:x_max]
            if eye_region.size > 0:
                # Enhanced preprocessing for better pupil detection
                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
                eye_gray = cv2.GaussianBlur(eye_gray, (5, 5), 0)
                eye_gray = cv2.equalizeHist(eye_gray)  # Improve contrast
                
                eye_regions[eye_name] = {
                    'region': eye_gray,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'landmarks': eye_landmarks
                }
        
        return eye_regions

    def detect_pupils(self, eye_region_data):
        """Enhanced pupil detection using multiple methods."""
        eye_img = eye_region_data['region']
        bbox = eye_region_data['bbox']
        
        if eye_img.size == 0:
            return None
            
        # Method 1: HoughCircles for circular pupil detection
        circles = cv2.HoughCircles(
            eye_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=25
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Take the most central circle
            center_x, center_y = eye_img.shape[1] // 2, eye_img.shape[0] // 2
            best_circle = min(circles, key=lambda c: np.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2))
            
            # Convert to frame coordinates
            pupil_x = bbox[0] + best_circle[0]
            pupil_y = bbox[1] + best_circle[1]
            return (pupil_x, pupil_y)
        
        # Method 2: Contour-based detection (fallback)
        _, thresh = cv2.threshold(eye_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)  # Invert to make pupil white
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the pupil)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 20:  # Minimum area threshold
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert to frame coordinates
                    pupil_x = bbox[0] + cx
                    pupil_y = bbox[1] + cy
                    return (pupil_x, pupil_y)
        
        return None

    def calculate_ear(self, eye_landmarks_indices, landmarks, frame_shape):
        """Calculate Eye Aspect Ratio for blink detection."""
        h, w = frame_shape[:2]
        
        points = []
        for idx in eye_landmarks_indices:
            if hasattr(landmarks[idx], 'x') and hasattr(landmarks[idx], 'y'):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                points.append((x, y))
        
        if len(points) < 6:
            return 0.0
        
        # Calculate vertical distances
        vertical_1 = self.calculate_distance(points[1], points[5])
        vertical_2 = self.calculate_distance(points[2], points[4])
        
        # Calculate horizontal distance
        horizontal = self.calculate_distance(points[0], points[3])
        
        # Calculate EAR
        if horizontal > 0:
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        
        return 0.0

    def detect_blinks(self, landmarks, frame_shape):
        """Enhanced blink detection with temporal smoothing."""
        left_ear = self.calculate_ear(self.LEFT_EYE_EAR, landmarks, frame_shape)
        right_ear = self.calculate_ear(self.RIGHT_EYE_EAR, landmarks, frame_shape)
        
        # Average EAR for both eyes
        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_history.append(avg_ear)
        
        blink_detected = False
        
        # Check if current EAR is below threshold
        if avg_ear < self.ear_threshold:
            self.blink_counter += 1
        else:
            # If eyes were closed for enough frames, count as blink
            if self.blink_counter >= self.blink_frames:
                self.total_blinks += 1
                blink_detected = True
                self.last_blink_time = time.time()
            self.blink_counter = 0
        
        # Check for excessive blinking (proctoring flag)
        current_time = time.time()
        if len(self.ear_history) >= 30:  # 1 second of data
            recent_blinks = sum(1 for ear in list(self.ear_history)[-30:] if ear < self.ear_threshold)
            self.excessive_blinking = recent_blinks > 15  # More than 50% of frames
        
        return {
            'blink_detected': blink_detected,
            'ear': avg_ear,
            'blink_rate': self.total_blinks / max(1, (current_time - self.last_blink_time) / 60),
            'excessive_blinking': self.excessive_blinking
        }

    def calculate_gaze_direction(self, eye_landmarks, pupil_center, eye_type='left'):
        """Calculate gaze direction relative to eye center."""
        if not pupil_center or not eye_landmarks:
            return (0, 0)
        
        # Calculate eye center from landmarks
        eye_center_x = sum(p[0] for p in eye_landmarks) / len(eye_landmarks)
        eye_center_y = sum(p[1] for p in eye_landmarks) / len(eye_landmarks)
        
        # Calculate gaze vector
        gaze_x = pupil_center[0] - eye_center_x
        gaze_y = pupil_center[1] - eye_center_y
        
        # Normalize based on eye size
        eye_width = max(p[0] for p in eye_landmarks) - min(p[0] for p in eye_landmarks)
        eye_height = max(p[1] for p in eye_landmarks) - min(p[1] for p in eye_landmarks)
        
        if eye_width > 0 and eye_height > 0:
            normalized_x = (gaze_x / eye_width) * 100
            normalized_y = (gaze_y / eye_height) * 100
            
            return (normalized_x, normalized_y)
        
        return (0, 0)

    def calibrate_center_gaze(self, gaze_direction):
        """Calibrate center gaze position (call when user looks at center)."""
        self.gaze_center = gaze_direction
        self.gaze_center_calibrated = True

    def is_looking_away(self, gaze_direction):
        """Determine if user is looking away from screen."""
        if not self.gaze_center_calibrated:
            return False
        
        gaze_x, gaze_y = gaze_direction
        center_x, center_y = self.gaze_center
        
        # Calculate deviation from center
        dev_x = abs(gaze_x - center_x)
        dev_y = abs(gaze_y - center_y)
        
        # Check if outside acceptable bounds
        looking_away = (dev_x > 40 or dev_y > 30)  # Configurable thresholds
        self.looking_away = looking_away
        
        return looking_away

    def track_eye_movement(self, curr_pupils):
        """Track eye movement patterns for suspicious behavior detection."""
        if not self.prev_pupils['left'] or not self.prev_pupils['right']:
            self.prev_pupils = curr_pupils.copy()
            return {'movement_magnitude': 0, 'suspicious': False}
        
        # Calculate movement magnitude
        total_movement = 0
        valid_measurements = 0
        
        for eye in ['left', 'right']:
            if curr_pupils[eye] and self.prev_pupils[eye]:
                movement = self.calculate_distance(curr_pupils[eye], self.prev_pupils[eye])
                total_movement += movement
                valid_measurements += 1
        
        if valid_measurements > 0:
            avg_movement = total_movement / valid_measurements
            self.movement_history.append(avg_movement)
            
            # Detect suspicious patterns
            if len(self.movement_history) >= 30:  # 1 second of data
                recent_movements = list(self.movement_history)[-30:]
                avg_recent_movement = sum(recent_movements) / len(recent_movements)
                
                # Flag rapid, erratic movements
                self.suspicious_movement = avg_recent_movement > self.movement_threshold * 2
            
            self.prev_pupils = curr_pupils.copy()
            
            return {
                'movement_magnitude': avg_movement,
                'suspicious': self.suspicious_movement
            }
        
        return {'movement_magnitude': 0, 'suspicious': False}

    def calculate_attention_score(self, blink_data, gaze_direction, movement_data):
        """Calculate overall attention score for proctoring."""
        score = 100.0
        
        # Deduct for looking away
        if self.is_looking_away(gaze_direction):
            score -= 30
        
        # Deduct for excessive blinking
        if blink_data['excessive_blinking']:
            score -= 20
        
        # Deduct for suspicious movement
        if movement_data['suspicious']:
            score -= 25
        
        # Deduct for abnormal blink rate
        if blink_data['blink_rate'] > 30 or blink_data['blink_rate'] < 5:  # blinks per minute
            score -= 15
        
        return max(0, score)

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_frame(self, landmarks, frame):
        """
        Main processing function for each frame.
        Returns comprehensive eye tracking data for proctoring.
        """
        results = {
            'left_eye': None,
            'right_eye': None,
            'blink_data': None,
            'attention_score': 0,
            'flags': {
                'looking_away': False,
                'excessive_blinking': False,
                'suspicious_movement': False
            }
        }
        
        try:
            # Extract eye regions
            eye_regions = self.extract_eye_regions(landmarks, frame)
            
            # Process each eye
            curr_pupils = {'left': None, 'right': None}
            eye_metrics = {}
            
            for eye_name in ['left', 'right']:
                if eye_name in eye_regions:
                    # Detect pupil
                    pupil = self.detect_pupils(eye_regions[eye_name])
                    curr_pupils[eye_name] = pupil
                    
                    # Calculate gaze direction
                    gaze_direction = self.calculate_gaze_direction(eye_regions[eye_name]['landmarks'], pupil, eye_name)
                    
                    ear = self.calculate_ear(self.LEFT_EYE_EAR if eye_name == 'left' else self.RIGHT_EYE_EAR, landmarks, frame.shape)
                    
                    eye_metrics[eye_name] = EyeMetrics(
                        ear=ear,
                        gaze_direction=gaze_direction,
                        pupil_center=pupil if pupil else (0, 0),
                        blink_detected=False,
                        movement_magnitude=0,
                        attention_score=0
                    )
            
            # Detect blinks
            blink_data = self.detect_blinks(landmarks, frame.shape)
            
            # Track movement
            movement_data = self.track_eye_movement(curr_pupils)
            
            # Calculate average gaze direction
            avg_gaze = (0, 0)
            if eye_metrics:
                gaze_vectors = [metrics.gaze_direction for metrics in eye_metrics.values() if metrics.gaze_direction]
                if gaze_vectors:
                    avg_gaze = tuple(np.mean(gaze_vectors, axis=0))
            
            # Calculate attention score
            attention_score = self.calculate_attention_score(blink_data, avg_gaze, movement_data)
            
            # Update results
            results.update({
                'left_eye': eye_metrics.get('left'),
                'right_eye': eye_metrics.get('right'),
                'blink_data': blink_data,
                'attention_score': attention_score,
                'gaze_direction': avg_gaze,
                'movement_data': movement_data,
                'flags': {
                    'looking_away': self.looking_away,
                    'excessive_blinking': self.excessive_blinking,
                    'suspicious_movement': self.suspicious_movement
                }
            })
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return results

    def get_proctoring_summary(self):
        """Get summary of proctoring session data."""
        return {
            'total_blinks': self.total_blinks,
            'average_ear': sum(self.ear_history) / len(self.ear_history) if self.ear_history else 0,
            'movement_pattern': list(self.movement_history)[-30:] if self.movement_history else [],
            'gaze_pattern': list(self.gaze_history)[-30:] if self.gaze_history else [],
            'flags_triggered': {
                'looking_away': self.looking_away,
                'excessive_blinking': self.excessive_blinking,
                'suspicious_movement': self.suspicious_movement
            }
        }