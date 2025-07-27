import tkinter as tk
import threading
import time
import cv2
from PIL import Image, ImageTk
from datetime import datetime
from collections import deque
from tkinter import messagebox
import json

class ProctoringMainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Eye and Face Movement Detection System - Proctoring Mode")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # System state
        self.running = False
        self.proctoring_mode = False
        self.calibration_mode = False
        
        # Component references
        self.video_handler = None
        self.face_detector = None
        self.eye_detector = None
        self.movement_tracker = None
        
        # Proctoring data
        self.session_start_time = None
        self.violation_log = []
        self.attention_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.face_detection_count = 0
        self.eye_detection_count = 0
        self.total_frames = 0
        self.alert_threshold = 60  # Attention score threshold
        self.consecutive_violations = 0
        self.max_violations = 5
        
        # Alert system
        self.current_alerts = set()
        self.alert_sounds_enabled = True
        
        # Dashboard update tracking
        self.last_dashboard_update = 0
        self.dashboard_update_interval = 0.5  # Update every 500ms
        
        # Current metrics for dynamic updates
        self.current_attention_score = 0
        self.current_blink_rate = 0
        self.current_gaze_direction = (0, 0)
        self.current_violations_count = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container
        main_container = tk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create left frame for video
        video_frame = tk.Frame(main_container)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
        
        # Main canvas for video display
        self.canvas = tk.Canvas(video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create right frame for dashboard
        self.dashboard_container = tk.Frame(main_container, bg='#34495e', width=350)
        self.dashboard_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)
        self.dashboard_container.pack_propagate(False)
        
        # Setup dashboard
        self.setup_dashboard()

        # Control panel frame
        control_frame = tk.Frame(self, bg='#2c3e50', height=120)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        control_frame.pack_propagate(False)

        # Status and control buttons
        self.setup_controls(control_frame)

        # Alert overlay (initially hidden)
        self.setup_alert_overlay()
        
        # Start dashboard update thread
        self.start_dashboard_updater()

    def setup_controls(self, parent):
        """Setup control buttons and status display."""
        button_frame = tk.Frame(parent, bg='#2c3e50')
        button_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Control buttons
        self.start_button = tk.Button(
            button_frame, text="Start Detection", command=self.start_detection,
            font=("Arial", 12, "bold"), bg='#27ae60', fg='white', width=15
        )
        self.start_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.stop_button = tk.Button(
            button_frame, text="Stop Detection", command=self.stop_detection,
            font=("Arial", 12, "bold"), bg='#e74c3c', fg='white', width=15
        )
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.proctoring_button = tk.Button(
            button_frame, text="Enable Proctoring", command=self.toggle_proctoring,
            font=("Arial", 12, "bold"), bg='#f39c12', fg='white', width=15
        )
        self.proctoring_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.calibrate_button = tk.Button(
            button_frame, text="Calibrate Gaze", command=self.start_calibration,
            font=("Arial", 12, "bold"), bg='#9b59b6', fg='white', width=15
        )
        self.calibrate_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Status display
        self.status_label = tk.Label(
            button_frame, text="Status: Ready", font=("Arial", 14, "bold"),
            bg='#34495e', fg='white', width=50
        )
        self.status_label.grid(row=1, column=0, columnspan=4, pady=10, sticky='ew')
        
    def setup_dashboard(self):
        """Setup proctoring dashboard with metrics."""
        # Clear existing dashboard content
        for widget in self.dashboard_container.winfo_children():
            widget.destroy()
            
        # Dashboard title
        title_label = tk.Label(
            self.dashboard_container, text="PROCTORING DASHBOARD", 
            font=("Arial", 14, "bold"), bg='#34495e', fg='#ecf0f1'
        )
        title_label.pack(pady=(15, 10))

        # Create scrollable frame for metrics
        canvas_scroll = tk.Canvas(self.dashboard_container, bg='#34495e', highlightthickness=0)
        scrollbar = tk.Scrollbar(self.dashboard_container, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = tk.Frame(canvas_scroll, bg='#34495e')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )

        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y")

        # Main metrics section
        self.create_metrics_section(scrollable_frame)
        
        # Alert indicators section
        self.create_alert_section(scrollable_frame)
        
        # Session statistics section
        self.create_session_stats_section(scrollable_frame)
        
    def create_metrics_section(self, parent):
        """Create the main metrics section."""
        metrics_frame = tk.LabelFrame(
            parent, text="Real-time Metrics", font=("Arial", 11, "bold"),
            bg='#34495e', fg='#ecf0f1', padx=10, pady=10
        )
        metrics_frame.pack(fill='x', padx=10, pady=5)
        
        # Attention Score (most prominent)
        self.attention_label = tk.Label(
            metrics_frame, text="Attention Score: --", 
            font=("Arial", 14, "bold"), bg='#34495e', fg='#2ecc71'
        )
        self.attention_label.pack(anchor='w', pady=3)
        
        # Attention progress bar
        self.attention_progress_frame = tk.Frame(metrics_frame, bg='#34495e')
        self.attention_progress_frame.pack(fill='x', pady=(0, 5))
        
        self.attention_progress = tk.Canvas(
            self.attention_progress_frame, height=20, bg='#2c3e50', highlightthickness=1
        )
        self.attention_progress.pack(fill='x')
        
        # Current detection status
        self.detection_status_label = tk.Label(
            metrics_frame, text="Detection: Inactive", 
            font=("Arial", 11, "bold"), bg='#34495e', fg='#95a5a6'
        )
        self.detection_status_label.pack(anchor='w', pady=2)
        
        # Gaze Direction
        self.gaze_label = tk.Label(
            metrics_frame, text="Gaze Direction: --", 
            font=("Arial", 10), bg='#34495e', fg='#ecf0f1'
        )
        self.gaze_label.pack(anchor='w', pady=1)
        
        # Blink Rate
        self.blink_label = tk.Label(
            metrics_frame, text="Blink Rate: -- bpm", 
            font=("Arial", 10), bg='#34495e', fg='#ecf0f1'
        )
        self.blink_label.pack(anchor='w', pady=1)
        
    def create_alert_section(self, parent):
        """Create alert indicators section."""
        alert_frame = tk.LabelFrame(
            parent, text="Alert Status", font=("Arial", 11, "bold"),
            bg='#34495e', fg='#ecf0f1', padx=10, pady=10
        )
        alert_frame.pack(fill='x', padx=10, pady=5)
        
        # Violations count
        self.violation_label = tk.Label(
            alert_frame, text="Total Violations: 0", 
            font=("Arial", 12, "bold"), bg='#34495e', fg='#e74c3c'
        )
        self.violation_label.pack(anchor='w', pady=2)
        
        # Alert indicators grid
        indicators_frame = tk.Frame(alert_frame, bg='#34495e')
        indicators_frame.pack(fill='x', pady=(5, 0))
        
        self.alert_indicators = {}
        alert_types = ['Looking Away', 'Excessive Blinking', 'Suspicious Movement', 'No Face Detected']
        
        for i, alert_type in enumerate(alert_types):
            indicator = tk.Label(
                indicators_frame, text=f"● {alert_type}", 
                font=("Arial", 9), bg='#34495e', fg='#7f8c8d'
            )
            indicator.grid(row=i//2, column=i%2, sticky='w', padx=5, pady=2)
            self.alert_indicators[alert_type] = indicator
            
    def create_session_stats_section(self, parent):
        """Create session statistics section."""
        stats_frame = tk.LabelFrame(
            parent, text="Session Statistics", font=("Arial", 11, "bold"),
            bg='#34495e', fg='#ecf0f1', padx=10, pady=10
        )
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        # Session Time
        self.session_label = tk.Label(
            stats_frame, text="Session Duration: 00:00:00", 
            font=("Arial", 11, "bold"), bg='#34495e', fg='#3498db'
        )
        self.session_label.pack(anchor='w', pady=2)
        
        # Accuracy metrics
        accuracy_frame = tk.Frame(stats_frame, bg='#34495e')
        accuracy_frame.pack(fill='x', pady=5)
        
        # Face Detection Accuracy
        self.face_accuracy_label = tk.Label(
            accuracy_frame, text="Face Detection: --%", 
            font=("Arial", 10), bg='#34495e', fg='#3498db'
        )
        self.face_accuracy_label.pack(anchor='w', pady=1)
        
        # Eye Detection Accuracy  
        self.eye_accuracy_label = tk.Label(
            accuracy_frame, text="Eye Detection: --%", 
            font=("Arial", 10), bg='#34495e', fg='#9b59b6'
        )
        self.eye_accuracy_label.pack(anchor='w', pady=1)
        
        # Frame processing rate
        self.fps_label = tk.Label(
            accuracy_frame, text="Processing Rate: -- fps", 
            font=("Arial", 10), bg='#34495e', fg='#95a5a6'
        )
        self.fps_label.pack(anchor='w', pady=1)
        
    def start_dashboard_updater(self):
        """Start the dashboard update thread."""
        def update_dashboard():
            while True:
                try:
                    if self.running:
                        self.update_dashboard_display()
                    time.sleep(self.dashboard_update_interval)
                except Exception as e:
                    print(f"Dashboard update error: {e}")
                    
        threading.Thread(target=update_dashboard, daemon=True).start()
        
    def update_dashboard_display(self):
        """Update dashboard display with current data."""
        try:
            # Clamp attention score to 100
            score = min(self.current_attention_score, 100)
            score_color = '#2ecc71' if score > 80 else '#f39c12' if score > 60 else '#e74c3c'
            self.attention_label.config(
                text=f"Attention Score: {score:.1f}%",
                fg=score_color
            )
            self.update_progress_bar(score)

            # Update detection status
            status_text = "Detection: Active" if self.running else "Detection: Inactive"
            status_color = '#2ecc71' if self.running else '#95a5a6'
            self.detection_status_label.config(text=status_text, fg=status_color)

            # Update gaze direction
            gaze_x, gaze_y = self.current_gaze_direction
            gaze_desc = self.describe_gaze_direction(gaze_x, gaze_y)
            self.gaze_label.config(text=f"Gaze Direction: {gaze_desc}")

            # Update blink rate
            self.blink_label.config(text=f"Blink Rate: {self.current_blink_rate:.1f} bpm")

            # Update violations count with color coding
            violation_color = '#e74c3c' if len(self.violation_log) > 0 else '#95a5a6'
            self.violation_label.config(
                text=f"Total Violations: {len(self.violation_log)}",
                fg=violation_color
            )

            # Get face and eye accuracy from detectors if available
            face_accuracy = None
            eye_accuracy = None
            if hasattr(self.face_detector, 'get_accuracy'):
                face_accuracy = self.face_detector.get_accuracy()
            if hasattr(self.eye_detector, 'get_accuracy'):
                eye_accuracy = self.eye_detector.get_accuracy()
            # Fallback to counters if not available
            if face_accuracy is None:
                face_accuracy = (self.face_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0
            if eye_accuracy is None:
                eye_accuracy = (self.eye_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0
            self.face_accuracy_label.config(text=f"Face Detection: {face_accuracy:.1f}%")
            self.eye_accuracy_label.config(text=f"Eye Detection: {eye_accuracy:.1f}%")

            # Calculate and display FPS
            if hasattr(self, 'frame_times') and len(self.frame_times) > 1:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                self.fps_label.config(text=f"Processing Rate: {fps:.1f} fps")
        except Exception as e:
            print(f"Error updating dashboard: {e}")
            
    def update_progress_bar(self, value):
        """Update the attention score progress bar."""
        try:
            self.attention_progress.delete("all")
            width = self.attention_progress.winfo_width()
            height = self.attention_progress.winfo_height()
            
            if width > 1 and height > 1:
                # Background
                self.attention_progress.create_rectangle(0, 0, width, height, fill='#2c3e50', outline='')
                
                # Progress fill
                fill_width = (value / 100) * width
                color = '#2ecc71' if value > 80 else '#f39c12' if value > 60 else '#e74c3c'
                self.attention_progress.create_rectangle(0, 0, fill_width, height, fill=color, outline='')
                
                # Text overlay
                self.attention_progress.create_text(
                    width/2, height/2, text=f"{value:.1f}%", 
                    fill='white', font=("Arial", 8, "bold")
                )
        except Exception as e:
            print(f"Error updating progress bar: {e}")
        
    def setup_alert_overlay(self):
        """Setup alert overlay for critical violations."""
        self.alert_overlay = tk.Frame(self, bg='#e74c3c', height=80)
        self.alert_overlay.place(relx=0.5, rely=0.1, anchor=tk.CENTER, relwidth=0.6)
        self.alert_overlay.place_forget()  # Hide initially
        
        self.alert_text = tk.Label(
            self.alert_overlay, text="ALERT: Suspicious Activity Detected!",
            font=("Arial", 16, "bold"), bg='#e74c3c', fg='white'
        )
        self.alert_text.pack(expand=True)
        
    def setup_components(self, video_handler, face_detector, eye_detector, movement_tracker):
        """Setup component references."""
        self.video_handler = video_handler
        self.face_detector = face_detector
        self.eye_detector = eye_detector
        self.movement_tracker = movement_tracker
        
        # Initialize frame timing for FPS calculation
        self.frame_times = deque(maxlen=30)  # Keep last 30 frame times
        self.last_frame_time = time.time()
        
    def start_detection(self):
        """Start the detection system."""
        if not self.running:
            # Reinitialize video handler if needed
            if self.video_handler and not self.video_handler.is_opened():
                self.video_handler = self.video_handler.__class__()
            
            self.running = True
            self.session_start_time = time.time()
            
            # Reset counters
            self.face_detection_count = 0
            self.eye_detection_count = 0
            self.total_frames = 0
            
            threading.Thread(target=self.process_video, daemon=True).start()
            threading.Thread(target=self.update_session_timer, daemon=True).start()
            
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
    def stop_detection(self):
        """Stop the detection system."""
        self.running = False
        if self.video_handler:
            self.video_handler.release()
            
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Generate session report if proctoring was enabled
        if self.proctoring_mode:
            self.generate_session_report()
            
    def toggle_proctoring(self):
        """Toggle proctoring mode."""
        self.proctoring_mode = not self.proctoring_mode
        
        if self.proctoring_mode:
            self.proctoring_button.config(text="Disable Proctoring", bg='#e74c3c')
            self.violation_log.clear()
            self.consecutive_violations = 0
            messagebox.showinfo("Proctoring Enabled", 
                              "Proctoring mode is now active. Your behavior will be monitored.")
        else:
            self.proctoring_button.config(text="Enable Proctoring", bg='#f39c12')
            self.hide_alert()
            
    def start_calibration(self):
        """Start gaze calibration process."""
        if not self.running:
            messagebox.showwarning("Calibration", "Please start detection first.")
            return
            
        self.calibration_mode = True
        self.calibrate_button.config(text="Calibrating...", state='disabled')
        
        # Show calibration instructions
        messagebox.showinfo("Gaze Calibration", 
                          "Look directly at the center of the screen and press OK.")
        
        # The calibration will be handled in the next few frames
        self.after(3000, self.finish_calibration)  # Auto-finish after 3 seconds
        
    def finish_calibration(self):
        """Finish the calibration process."""
        self.calibration_mode = False
        self.calibrate_button.config(text="Calibrate Gaze", state='normal')
        messagebox.showinfo("Calibration Complete", "Gaze calibration completed successfully!")
        
    def process_video(self):
        """Main video processing loop with proctoring integration."""
        while self.running:
            try:
                frame_start_time = time.time()
                ret, frame = self.video_handler.read_frame()
                if not ret:
                    self.update_status("Camera error")
                    continue
                self.total_frames += 1
                # Detect faces and landmarks
                faces = self.face_detector.detect_faces(frame)
                landmarks = self.face_detector.get_landmarks(frame)
                # Initialize status messages
                face_status = "No face detected"
                eye_status = "Eyes not detected"
                proctoring_data = None
                if faces and landmarks:
                    face_status = f"Face detected ({len(faces)})"
                    self.face_detection_count += 1
                    proctoring_data = self.eye_detector.process_frame(landmarks, frame)
                    # Update current metrics for dashboard
                    if proctoring_data:
                        # Penalize attention score if looking away or face is not centered
                        flags = proctoring_data.get('flags', {})
                        attention_score = proctoring_data.get('attention_score', 0)
                        if flags.get('looking_away') or flags.get('suspicious_movement'):
                            attention_score = min(attention_score, 70)  # Reduce score if not focused
                        self.current_attention_score = attention_score
                        self.current_gaze_direction = proctoring_data.get('gaze_direction', (0, 0))
                        blink_data = proctoring_data.get('blink_data', {})
                        self.current_blink_rate = blink_data.get('blink_rate', 0)
                else:
                    # No face detected, set attention score to 0
                    self.current_attention_score = 0
                    self.current_gaze_direction = (0, 0)
                    self.current_blink_rate = 0
                
                # Eye status logic
                left_eye = proctoring_data.get('left_eye') if proctoring_data else None
                right_eye = proctoring_data.get('right_eye') if proctoring_data else None
                flags = proctoring_data.get('flags', {}) if proctoring_data else {}
                
                if left_eye and right_eye:
                    if left_eye.pupil_center == (0, 0) or right_eye.pupil_center == (0, 0):
                        eye_status = "Please look into the screen"
                    else:
                        gaze_x, gaze_y = proctoring_data.get('gaze_direction', (0, 0))
                        if flags.get('looking_away'):
                            eye_status = "Looking away from screen"
                        elif abs(gaze_x) < 20 and abs(gaze_y) < 20:
                            eye_status = "Eyes focused - Good attention"
                            self.eye_detection_count += 1
                        else:
                            direction = []
                            if gaze_x < -20:
                                direction.append("Left")
                            elif gaze_x > 20:
                                direction.append("Right")
                            if gaze_y < -20:
                                direction.append("Up")
                            elif gaze_y > 20:
                                direction.append("Down")
                            eye_status = "Looking " + " ".join(direction) if direction else "Eyes detected"
                            self.eye_detection_count += 1
                else:
                    eye_status = "Eyes not detected"
                    
                # Handle calibration
                if self.calibration_mode and proctoring_data and proctoring_data.get('gaze_direction'):
                    self.eye_detector.calibrate_center_gaze(proctoring_data['gaze_direction'])
                
                # Process proctoring data
                if self.proctoring_mode and proctoring_data:
                    self.process_proctoring_data(proctoring_data)
                    
                # Update display
                self.update_video_display(frame, faces, landmarks, proctoring_data)
                self.update_status(f"{face_status} | {eye_status}")
                frame_end_time = time.time()
                frame_time = frame_end_time - frame_start_time
                self.frame_times.append(frame_time)
            except Exception as e:
                print(f"Error in video processing: {e}")
                self.update_status(f"Error: {str(e)}")
            time.sleep(0.033)  # ~30 FPS
            
    def process_proctoring_data(self, data):
        """Process proctoring data and handle violations."""
        attention_score = data.get('attention_score', 0)
        flags = data.get('flags', {})
        
        # Store attention history
        self.attention_history.append(attention_score)
        
        # Check for violations
        current_violations = []
        
        if flags.get('looking_away'):
            current_violations.append('Looking Away')
            
        if flags.get('excessive_blinking'):
            current_violations.append('Excessive Blinking')
            
        if flags.get('suspicious_movement'):
            current_violations.append('Suspicious Movement')
            
        if attention_score < self.alert_threshold:
            current_violations.append('Low Attention')
            
        # Update alert indicators
        self.update_alert_indicators(current_violations)
        
        # Log violations
        if current_violations:
            self.consecutive_violations += 1
            
            violation_entry = {
                'timestamp': datetime.now().isoformat(),
                'violations': current_violations,
                'attention_score': attention_score,
                'gaze_direction': data.get('gaze_direction', (0, 0))
            }
            self.violation_log.append(violation_entry)
            
            # Only show critical alert if accuracy is below 80% and threshold is reached
            face_accuracy = (self.face_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0
            eye_accuracy = (self.eye_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0
            if self.consecutive_violations >= self.max_violations and (face_accuracy < 80 or eye_accuracy < 80):
                self.show_critical_alert()
            else:
                self.hide_alert()
        else:
            self.consecutive_violations = max(0, self.consecutive_violations - 1)
            self.hide_alert()
            
    def update_alert_indicators(self, active_violations):
        """Update visual alert indicators."""
        # Add "No Face Detected" to violations if no face is detected
        if self.total_frames > 0:
            recent_face_ratio = self.face_detection_count / self.total_frames
            if recent_face_ratio < 0.8:
                if 'No Face Detected' not in active_violations:
                    active_violations.append('No Face Detected')
        # Add "Looking Away" if gaze is not centered
        gaze_x, gaze_y = self.current_gaze_direction
        if abs(gaze_x) > 20 or abs(gaze_y) > 20:
            if 'Looking Away' not in active_violations:
                active_violations.append('Looking Away')
        # Update indicator colors
        for alert_type, indicator in self.alert_indicators.items():
            if alert_type in active_violations:
                indicator.config(fg='#e74c3c')
            else:
                indicator.config(fg='#7f8c8d')
                
    def show_critical_alert(self):
        """Show critical violation alert."""
        self.alert_overlay.place(relx=0.5, rely=0.1, anchor=tk.CENTER, relwidth=0.6)
        self.alert_text.config(text="⚠️ CRITICAL: Multiple Violations Detected! ⚠️")
        
        # Flash the alert
        self.flash_alert()
        
    def flash_alert(self):
        """Flash the alert overlay."""
        current_bg = self.alert_overlay.cget('bg')
        new_bg = '#c0392b' if current_bg == '#e74c3c' else '#e74c3c'
        
        self.alert_overlay.config(bg=new_bg)
        self.alert_text.config(bg=new_bg)
        
        if self.consecutive_violations >= self.max_violations:
            self.after(500, self.flash_alert)
            
    def hide_alert(self):
        """Hide the alert overlay."""
        self.alert_overlay.place_forget()
        
    def describe_gaze_direction(self, x, y):
        """Describe gaze direction in human-readable format."""
        threshold = 20
        
        if abs(x) < threshold and abs(y) < threshold:
            return "Center"
        
        desc = []
        if y < -threshold:
            desc.append("Up")
        elif y > threshold:
            desc.append("Down")
            
        if x < -threshold:
            desc.append("Left")
        elif x > threshold:
            desc.append("Right")
            
        return " ".join(desc) if desc else "Center"
        
    def update_session_timer(self):
        """Update session timer display."""
        while self.running and self.session_start_time:
            try:
                elapsed = time.time() - self.session_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                
                self.session_label.config(text=f"Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
                time.sleep(1)
            except Exception as e:
                print(f"Timer update error: {e}")
                break
            
    def update_video_display(self, frame, faces, landmarks, proctoring_data):
        """Update video display with annotations."""
        display_frame = frame.copy()

        # Draw face rectangles
        for face in faces:
            bbox = face.get('box')
            if bbox:
                h, w = display_frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                cv2.rectangle(display_frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

        # Draw eye regions and pupils if available
        if proctoring_data:
            for eye_name in ['left_eye', 'right_eye']:
                eye_data = proctoring_data.get(eye_name)
                if eye_data and eye_data.pupil_center != (0, 0):
                    cv2.circle(display_frame, eye_data.pupil_center, 3, (0, 0, 255), -1)
                    
        # Add attention score overlay if proctoring is active
        if self.proctoring_mode and proctoring_data:
            attention_score = proctoring_data.get('attention_score', 0)
            score = min(attention_score, 100)
            score_color = (0, 255, 0) if score > 80 else (0, 165, 255) if score > 60 else (0, 0, 255)
            
            cv2.putText(display_frame, f"Attention: {score:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)
                       
        # Convert and display
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img = img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk  # Keep a reference
        
    def update_status(self, message):
        """Update status label thread-safely."""
        try:
            self.status_label.config(text=f"Status: {message}")
        except:
            pass  # Ignore if window is closing
            
    def generate_session_report(self):
        """Generate and save proctoring session report."""
        if not self.violation_log:
            messagebox.showinfo("Session Report", "No violations detected during this session.")
            return
            
        report = {
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'total_violations': len(self.violation_log),
            'violation_details': self.violation_log,
            'average_attention': sum(self.attention_history) / len(self.attention_history) if self.attention_history else 0,
            'face_accuracy': (self.face_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0,
            'eye_accuracy': (self.eye_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0,
            'total_frames_processed': self.total_frames,
            'session_statistics': {
                'max_attention_score': max(self.attention_history) if self.attention_history else 0,
                'min_attention_score': min(self.attention_history) if self.attention_history else 0,
                'attention_variance': self.calculate_attention_variance(),
                'most_common_violations': self.get_most_common_violations()
            },
            'report_generated': datetime.now().isoformat()
        }
        
        # Save report to file
        filename = f"proctoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            messagebox.showinfo("Session Report", f"Report saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")
            
    def calculate_attention_variance(self):
        """Calculate variance in attention scores."""
        if len(self.attention_history) < 2:
            return 0
            
        mean_attention = sum(self.attention_history) / len(self.attention_history)
        variance = sum((x - mean_attention) ** 2 for x in self.attention_history) / len(self.attention_history)
        return variance
        
    def get_most_common_violations(self):
        """Get the most common violation types."""
        violation_counts = {}
        for entry in self.violation_log:
            for violation in entry['violations']:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        # Sort by count, descending
        sorted_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_violations[:5])  # Return top 5
            
    def on_close(self):
        """Handle window close event."""
        if self.proctoring_mode and self.violation_log:
            result = messagebox.askyesno("Save Report", 
                                       "Save proctoring report before closing?")
            if result:
                self.generate_session_report()
                
        self.running = False
        if self.video_handler:
            self.video_handler.release()
        self.destroy()