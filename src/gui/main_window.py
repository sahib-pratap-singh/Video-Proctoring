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
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface with proctoring controls."""
        # Main canvas for video display
        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel frame
        control_frame = tk.Frame(self, bg='#2c3e50', height=120)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        control_frame.pack_propagate(False)
        
        # Status and control buttons
        self.setup_controls(control_frame)
        
        # Proctoring dashboard
        self.setup_dashboard(control_frame)
        
        # Alert overlay (initially hidden)
        self.setup_alert_overlay()
        
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
        
    def setup_dashboard(self, parent):
        """Setup proctoring dashboard with metrics."""
        dashboard_frame = tk.Frame(parent, bg='#34495e', width=400)
        dashboard_frame.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.Y)
        dashboard_frame.pack_propagate(False)
        
        # Dashboard title
        tk.Label(
            dashboard_frame, text="PROCTORING DASHBOARD", 
            font=("Arial", 14, "bold"), bg='#34495e', fg='#ecf0f1'
        ).pack(pady=(10, 5))
        
        # Metrics frame
        metrics_frame = tk.Frame(dashboard_frame, bg='#34495e')
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Attention Score
        self.attention_label = tk.Label(
            metrics_frame, text="Attention Score: --", 
            font=("Arial", 12, "bold"), bg='#34495e', fg='#2ecc71'
        )
        self.attention_label.pack(anchor='w', pady=2)
        
        # Blink Rate
        self.blink_label = tk.Label(
            metrics_frame, text="Blink Rate: --", 
            font=("Arial", 10), bg='#34495e', fg='#ecf0f1'
        )
        self.blink_label.pack(anchor='w', pady=1)
        
        # Gaze Direction
        self.gaze_label = tk.Label(
            metrics_frame, text="Gaze: --", 
            font=("Arial", 10), bg='#34495e', fg='#ecf0f1'
        )
        self.gaze_label.pack(anchor='w', pady=1)
        
        # Violations Count
        self.violation_label = tk.Label(
            metrics_frame, text="Violations: 0", 
            font=("Arial", 10), bg='#34495e', fg='#e74c3c'
        )
        self.violation_label.pack(anchor='w', pady=1)
        
        # Session Time
        self.session_label = tk.Label(
            metrics_frame, text="Session: 00:00:00", 
            font=("Arial", 10), bg='#34495e', fg='#ecf0f1'
        )
        self.session_label.pack(anchor='w', pady=1)
        
        # Face Accuracy
        self.face_accuracy_label = tk.Label(
            metrics_frame, text="Face Accuracy: --%", 
            font=("Arial", 10), bg='#34495e', fg='#3498db'
        )
        self.face_accuracy_label.pack(anchor='w', pady=1)
        
        # Eye Accuracy
        self.eye_accuracy_label = tk.Label(
            metrics_frame, text="Eye Accuracy: --%", 
            font=("Arial", 10), bg='#34495e', fg='#9b59b6'
        )
        self.eye_accuracy_label.pack(anchor='w', pady=1)
        
        # Alert indicators
        self.alert_frame = tk.Frame(metrics_frame, bg='#34495e')
        self.alert_frame.pack(fill='x', pady=(10, 0))
        
        self.alert_indicators = {}
        alert_types = ['Looking Away', 'Excessive Blinking', 'Suspicious Movement', 'No Face']
        
        for i, alert_type in enumerate(alert_types):
            indicator = tk.Label(
                self.alert_frame, text=f"● {alert_type}", 
                font=("Arial", 9), bg='#34495e', fg='#7f8c8d'
            )
            indicator.grid(row=i//2, column=i%2, sticky='w', padx=5, pady=1)
            self.alert_indicators[alert_type] = indicator
        
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
        
    def start_detection(self):
        """Start the detection system."""
        if not self.running:
            # Reinitialize video handler if needed
            if self.video_handler and not self.video_handler.is_opened():
                self.video_handler = self.video_handler.__class__()
            
            self.running = True
            self.session_start_time = time.time()
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
                    
                    # Process with enhanced eye detector
                    proctoring_data = self.eye_detector.process_frame(landmarks, frame)
                    
                    # Eye status logic
                    left_eye = proctoring_data.get('left_eye')
                    right_eye = proctoring_data.get('right_eye')
                    flags = proctoring_data.get('flags', {})
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
                    if self.calibration_mode and proctoring_data.get('gaze_direction'):
                        self.eye_detector.calibrate_center_gaze(proctoring_data['gaze_direction'])
                    
                    # Process proctoring data
                    if self.proctoring_mode:
                        self.process_proctoring_data(proctoring_data)
                        
                # Update display
                self.update_video_display(frame, faces, landmarks, proctoring_data)
                self.update_status(f"{face_status} | {eye_status}")
                
            except Exception as e:
                print(f"Error in video processing: {e}")
                self.update_status(f"Error: {str(e)}")
                
            time.sleep(0.033)  # ~30 FPS
            
    def format_eye_status(self, proctoring_data):
        """Format eye status message from proctoring data."""
        if not proctoring_data:
            return "Eyes not detected"
            
        attention_score = proctoring_data.get('attention_score', 0)
        flags = proctoring_data.get('flags', {})
        
        if flags.get('looking_away'):
            return "Looking away from screen"
        elif flags.get('excessive_blinking'):
            return "Excessive blinking detected"
        elif flags.get('suspicious_movement'):
            return "Suspicious eye movement"
        elif attention_score > 80:
            return "Eyes focused - Good attention"
        elif attention_score > 60:
            return "Eyes detected - Fair attention"
        else:
            return "Poor attention detected"
            
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
            
            # Show critical alert if too many consecutive violations
            if self.consecutive_violations >= self.max_violations:
                self.show_critical_alert()
        else:
            self.consecutive_violations = max(0, self.consecutive_violations - 1)
            self.hide_alert()
            
    def update_alert_indicators(self, active_violations):
        """Update visual alert indicators."""
        for alert_type, indicator in self.alert_indicators.items():
            if alert_type in active_violations:
                indicator.config(fg='#e74c3c')  # Red for active
            else:
                indicator.config(fg='#7f8c8d')  # Gray for inactive
                
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
        
    def update_dashboard(self, proctoring_data):
        """Update the proctoring dashboard with current metrics."""
        attention_score = proctoring_data.get('attention_score', 0)
        blink_data = proctoring_data.get('blink_data', {})
        gaze_direction = proctoring_data.get('gaze_direction', (0, 0))
        
        # Update attention score with color coding
        score_color = '#2ecc71' if attention_score > 80 else '#f39c12' if attention_score > 60 else '#e74c3c'
        self.attention_label.config(
            text=f"Attention Score: {attention_score:.1f}%",
            fg=score_color
        )
        
        # Update blink rate
        blink_rate = blink_data.get('blink_rate', 0)
        self.blink_label.config(text=f"Blink Rate: {blink_rate:.1f}/min")
        
        # Update gaze direction
        gaze_x, gaze_y = gaze_direction
        gaze_desc = self.describe_gaze_direction(gaze_x, gaze_y)
        self.gaze_label.config(text=f"Gaze: {gaze_desc}")
        
        # Update violations count
        self.violation_label.config(text=f"Violations: {len(self.violation_log)}")
        
        # Update face and eye accuracy
        face_accuracy = (self.face_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        eye_accuracy = (self.eye_detection_count / self.total_frames) * 100 if self.total_frames > 0 else 0
        self.face_accuracy_label.config(text=f"Face Accuracy: {face_accuracy:.1f}%")
        self.eye_accuracy_label.config(text=f"Eye Accuracy: {eye_accuracy:.1f}%")
        
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
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            self.session_label.config(text=f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
            time.sleep(1)
            
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
            score_color = (0, 255, 0) if attention_score > 80 else (0, 165, 255) if attention_score > 60 else (0, 0, 255)
            
            cv2.putText(display_frame, f"Attention: {attention_score:.1f}%", 
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