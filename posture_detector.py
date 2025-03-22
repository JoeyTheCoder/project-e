import cv2
import mediapipe as mp
import time
import numpy as np
import math  # Add this for angle calculations
from utils import *
from config import *  # This will import DEBUG_MODE from config.py
import matplotlib.pyplot as plt

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def detect_pose(self, image):
        """Process an image and detect pose landmarks"""
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results
        
    def draw_landmarks(self, image, results):
        """Draw the pose landmarks on the image"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image

class PostureAnalyzer:
    def __init__(self):
        # Track posture status over time
        self.posture_history = []
        self.max_history = 10
    
    def analyze_posture(self, landmarks):
        """Analyze posture based on landmark positions"""
        if landmarks is None:
            return "No pose detected"
        
        # Get key landmarks
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Only analyze if upper body is visible with high confidence
        if (left_shoulder.visibility < 0.7 or right_shoulder.visibility < 0.7 or
            left_ear.visibility < 0.7 or right_ear.visibility < 0.7 or
            nose.visibility < 0.7):
            return "Incomplete pose detected"
        
        # Check for posture issues
        issues = []
        
        # 1. Check shoulder alignment (horizontal)
        shoulder_diff = calculate_vertical_difference(left_shoulder, right_shoulder)
        if shoulder_diff > SHOULDER_ALIGNMENT_THRESHOLD:
            issues.append("Uneven shoulders")
        
        # 2. Calculate midpoint of shoulders for other checks
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 3. Check for forward head posture - CORRECTED
        # In the image coordinate system, x increases from left to right
        # For forward head, nose.x will be SMALLER than mid_shoulder_x
        head_forward_distance = nose.x - mid_shoulder_x
        
        # 4. Check for slouching (vertical alignment) - CORRECTED
        # In the image coordinate system, y increases from top to bottom
        # So for good posture, ear_y should be LESS than shoulder_y
        mid_ear_y = (left_ear.y + right_ear.y) / 2
        # Positive value indicates ear is above shoulder (good)
        # Negative value indicates ear is below shoulder (slouching)
        slouch_value = mid_shoulder_y - mid_ear_y  
        
        # Debug output to understand the actual measurements
        if DEBUG_MODE:
            print(f"Slouch value: {slouch_value:.3f}, Threshold: {SLOUCH_THRESHOLD}")
            print(f"Head forward value: {head_forward_distance:.3f}, Threshold: {-HEAD_FORWARD_THRESHOLD}")
            print(f"Shoulder diff: {shoulder_diff:.3f}, Threshold: {SHOULDER_ALIGNMENT_THRESHOLD}")
            print("--------------------")
        
        # CORRECTED slouch detection
        # When slouching, ears will be lower relative to shoulders, so slouch_value will be smaller or negative
        if slouch_value < SLOUCH_THRESHOLD:
            issues.append("Slouching")
        
        # CORRECTED forward head detection
        # When head is forward, nose.x will be smaller than mid_shoulder_x, so head_forward_distance is negative
        if head_forward_distance < -HEAD_FORWARD_THRESHOLD:
            issues.append("Forward head")
        
        # Determine overall posture status
        if issues:
            return f"Bad posture: {', '.join(issues)}"
        else:
            return "Good posture"
    
    def update_history(self, status):
        """Update posture history with new status"""
        self.posture_history.append(status)
        if len(self.posture_history) > self.max_history:
            self.posture_history.pop(0)
    
    def get_persistent_status(self):
        """Get the most persistent posture status in recent history"""
        if not self.posture_history:
            return "Unknown"
        
        # Count occurrences of each status
        status_counts = {}
        for status in self.posture_history:
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1
        
        # Return most common status
        return max(status_counts, key=status_counts.get)

class PostureFeedback:
    def __init__(self):
        self.last_alert_time = time.time()
        self.bad_posture_counter = 0
    
    def should_alert(self, posture_status):
        """Determine if we should alert the user"""
        current_time = time.time()
        
        # Reset counter if posture is good
        if posture_status.startswith("Good"):
            self.bad_posture_counter = 0
            return False
        
        # Increment counter for bad posture
        if posture_status.startswith("Bad"):
            self.bad_posture_counter += 1
        
        # Alert if bad posture persists and cooldown has passed
        if (self.bad_posture_counter >= BAD_POSTURE_THRESHOLD and 
            current_time - self.last_alert_time > ALERT_COOLDOWN):
            self.last_alert_time = current_time
            return True
        
        return False
    
    def alert(self, message):
        """Alert the user to correct their posture"""
        print(f"\nPOSTURE ALERT: {message}\n")
        # You could add sound alert here
        # import winsound
        # winsound.Beep(1000, 500)  # frequency, duration in ms
        return

def setup_camera(camera_id=CAMERA_ID):
    """Initialize camera with specified settings"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        raise Exception("Failed to open camera")
    return cap

def main():
    # Initialize components
    cap = setup_camera()
    detector = PoseDetector()
    analyzer = PostureAnalyzer()
    analyzer.mp_pose = mp.solutions.pose  # Pass mp_pose reference to analyzer
    feedback = PostureFeedback()
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Detect pose
        results = detector.detect_pose(frame)
        
        # Analyze posture
        posture_status = "Unknown"
        if results.pose_landmarks:
            posture_status = analyzer.analyze_posture(results.pose_landmarks)
            analyzer.update_history(posture_status)
            
            # Get persistent status for more stable feedback
            persistent_status = analyzer.get_persistent_status()
            
            # Check if we should alert the user
            if feedback.should_alert(persistent_status):
                feedback.alert(persistent_status)
        
        # Draw landmarks on frame
        frame = detector.draw_landmarks(frame, results)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        # Display information on the frame
        draw_text_with_background(frame, f"FPS: {fps:.1f}", (10, 30))
        draw_text_with_background(frame, f"Status: {posture_status}", (10, 60), 
                               bg_color=(0, 0, 255) if posture_status.startswith("Bad") else (0, 255, 0))
        
        # Display the frame
        cv2.imshow('Posture Detector', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
