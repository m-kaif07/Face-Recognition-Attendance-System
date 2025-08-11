import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hand Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Canvas setup
paintWindow = np.ones((471, 636, 3)) * 255
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0  # Default color is blue
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_index = green_index = red_index = yellow_index = 0

# Setup for buttons
cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Start video capture
cap = cv2.VideoCapture(1)

drawing = False  # Flag to track whether drawing should occur

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a more intuitive experience

    # Process the frame using Mediapipe hand tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Initialize center point
    index_finger_y = None

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger and thumb (landmarks 8 and 4)
            index_finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the distance between the index finger tip and thumb tip
            distance = calculate_distance(index_finger_tip, thumb_tip)

            # Threshold for pinch detection
            if distance < 0.05:
                drawing = True  # Start drawing
                index_finger_x, index_finger_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)  # Green circle for the finger tip

                # Handle drawing
                if index_finger_y <= 65:
                    if 40 <= index_finger_x <= 140:  # Clear Button
                        bpoints = [deque(maxlen=1024)]
                        gpoints = [deque(maxlen=1024)]
                        rpoints = [deque(maxlen=1024)]
                        ypoints = [deque(maxlen=1024)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :, :] = 255  # Clear canvas
                    elif 160 <= index_finger_x <= 255:  # Blue
                        colorIndex = 0
                    elif 275 <= index_finger_x <= 370:  # Green
                        colorIndex = 1
                    elif 390 <= index_finger_x <= 485:  # Red
                        colorIndex = 2
                    elif 505 <= index_finger_x <= 600:  # Yellow
                        colorIndex = 3
                else:
                    # Add the current point to the respective color's points list
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft((index_finger_x, index_finger_y))
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft((index_finger_x, index_finger_y))
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft((index_finger_x, index_finger_y))
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft((index_finger_x, index_finger_y))
            else:
                drawing = False  # Stop drawing

    # Add new deque when no finger is detected
    if not drawing:
        bpoints.append(deque(maxlen=1024))
        blue_index += 1
        gpoints.append(deque(maxlen=1024))
        green_index += 1
        rpoints.append(deque(maxlen=1024))
        red_index += 1
        ypoints.append(deque(maxlen=1024))
        yellow_index += 1

    # Draw the lines for all colors on the frame and paint canvas
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
