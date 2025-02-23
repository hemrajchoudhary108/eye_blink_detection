import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def get_eye_landmarks(landmarks):
    eye_landmarks = []
    for i in range(36, 48):
        x = landmarks[i].x
        y = landmarks[i].y
        eye_landmarks.append((x, y))
    return eye_landmarks

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def ear(points):
    A = calculate_distance(points[1], points[5])
    B = calculate_distance(points[2], points[4])
    C = calculate_distance(points[0], points[3])
    return (A + B) / (2.0 * C)

def calculate_ear(eye_landmarks):
    # Landmarks contains both the left and right eye landmarks.
    n = len(eye_landmarks)
    left_side = eye_landmarks[:n//2]
    right_side = eye_landmarks[n//2:]
    left_ear = ear(left_side)
    right_ear = ear(right_side)
    return (left_ear + right_ear) / 2.0
def draw_rectangle_around_eye(frame, eye_landmarks, start):
    x1 = eye_landmarks[start][0]
    y1 = int(np.mean([eye_landmarks[start + 1][1], eye_landmarks[start + 2][1]]))
    x2 = eye_landmarks[start + 3][0]
    y2 = int(np.mean([eye_landmarks[start + 4][1], eye_landmarks[start + 5][1]]))
    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2)
    
def draw_eye_landmarks(frame, eye_landmarks):
    draw_rectangle_around_eye(frame, eye_landmarks, 0)
    draw_rectangle_around_eye(frame, eye_landmarks, 6)

class Plotter:
    def __init__(self):
        # Initialize figure and axis
        self.fig, self.ax = plt.subplots()
        self.x_data, self.ear_values = deque([]), deque([])
        self.line, = self.ax.plot([], [], lw=2)
        
        # Set axis limits
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 2)
        self.ax.set_ylabel("Ear Value")
        self.ax.set_title("EAR Trend")
        self.ax.xaxis.set_visible(False)  # Hide x-axis

    # Function to update the graph
    def update(self, frame, ear):
        self.x_data.append(frame)
        self.ear_values.append(ear)  # Simulated price data

        if len(self.x_data) > 100:
            self.x_data.popleft()
            self.ear_values.popleft()

        self.line.set_data(self.x_data, self.ear_values)
        self.ax.set_xlim(self.x_data[0], self.x_data[-1])
        plt.pause(0.05)
        return self.line
