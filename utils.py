import math
import cv2
import numpy as np
import time

def calculate_angle(a, b, c):
    """Calculate angle between three points (in radians)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return angle

def calculate_distance(a, b):
    """Calculate Euclidean distance between two landmarks"""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def calculate_vertical_difference(a, b):
    """Calculate vertical (y-axis) difference between two landmarks"""
    return abs(a.y - b.y)

def calculate_horizontal_difference(a, b):
    """Calculate horizontal (x-axis) difference between two landmarks"""
    return abs(a.x - b.x)

def draw_text_with_background(image, text, position, font_scale=0.7, thickness=1, 
                             text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with background on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Draw background rectangle
    cv2.rectangle(image, position, 
                 (position[0] + text_w, position[1] + text_h + 5), 
                 bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, 
               (position[0], position[1] + text_h + 1), 
               font, font_scale, text_color, thickness)
    
    return image
