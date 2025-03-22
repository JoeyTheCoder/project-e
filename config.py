# Camera settings
CAMERA_ID = 0  # 0 for default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Pose detection settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Posture analysis parameters
# These thresholds will need to be calibrated based on testing
SHOULDER_ALIGNMENT_THRESHOLD = 0.1  # Max vertical difference between shoulders
HEAD_FORWARD_THRESHOLD = 0.2  # Max distance nose can be in front of shoulders
SLOUCH_THRESHOLD = 0.3  # Increased from 0.1 to 0.3 to be less sensitive

# Feedback settings
ALERT_COOLDOWN = 5  # Seconds between alerts
BAD_POSTURE_THRESHOLD = 3  # Consecutive bad posture detections before alerting

# Debug mode for calibration
DEBUG_MODE = True  # Set to False after calibration
