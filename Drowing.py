import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import datetime
import os
import time

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Setup webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1024)  # width
cap.set(4, 576)   # height

canvas = None
prev_point = None
brush_color = (2, 2, 2)  # Default color (black)
brush_thickness = 5

# Recording variables
recording = False
video_writer = None

# Message display timer
saved_time = None

# Create output folders
os.makedirs("screenshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

# Color palette (BGR)
colors = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255),
    "Black": (2, 2, 2),
    "Orange": (0, 165, 255),
    "Purple": (128, 0, 128),
    "Eraser": None  # Eraser option
}

palette_x = None
palette_w = 80
palette_h = 40

# Buffer to smooth finger states
finger_state_buffer = deque(maxlen=5)

def draw_palette(frame):
    """Draw horizontal color palette at the top."""
    global palette_x
    h, w, _ = frame.shape
    palette_x = 10
    y1 = 10
    for i, (name, color) in enumerate(colors.items()):
        x1 = i * palette_w + 10
        x2 = x1 + palette_w
        cv2.rectangle(frame, (x1, y1), (x2, y1 + palette_h), color, -1)
        text_color = (0, 0, 0) if name != "Black" else (255, 255, 255)
        if name == "Eraser":
            text_color = (255, 255, 255)
        cv2.putText(frame, name, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return frame

def check_palette_hover(x, y):
    """Check if finger is over a palette color."""
    y1 = 10
    if y < y1 or y > y1 + palette_h:
        return None
    index = (x - 10) // palette_w
    if 0 <= index < len(colors):
        name, value = list(colors.items())[index]
        return name, value
    return None

def fingers_up(landmarks):
    """Return booleans [thumb, index, middle, ring, pinky]."""
    if landmarks is None:
        return [False]*5
    fingers = []
    fingers.append(landmarks[4].x < landmarks[3].x)  # Thumb
    for tip_id, pip_id in zip([8,12,16,20],[6,10,14,18]):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)
    return fingers

print("Press 'q' to quit, 'c' to clear, 's' for screenshot, 'r' to record.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    finger_states = [False]*5

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        finger_states = fingers_up(landmarks)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)
        cv2.circle(frame, (index_x, index_y), 8, (0, 0, 255), -1)

        finger_state_buffer.append(finger_states)

        hovered = check_palette_hover(index_x, index_y)
        if hovered:
            name, value = hovered
            if name == "Eraser":
                brush_color = (0, 0, 0)
                brush_thickness = 30
            else:
                brush_color = value
                brush_thickness = 5

    else:
        finger_state_buffer.append([False]*5)
        prev_point = None

    if len(finger_state_buffer) == finger_state_buffer.maxlen:
        counts = np.sum(finger_state_buffer, axis=0)
        smoothed_states = counts > (finger_state_buffer.maxlen // 2)
    else:
        smoothed_states = finger_states

    # Draw when only index finger is up
    if smoothed_states[1] and not smoothed_states[0] and not smoothed_states[2] and not smoothed_states[3] and not smoothed_states[4]:
        if prev_point:
            cv2.line(canvas, prev_point, (index_x, index_y), brush_color, brush_thickness)
        prev_point = (index_x, index_y)
    else:
        prev_point = None

    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_inv))
    fg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
    output = cv2.add(bg, fg)

    output = draw_palette(output)

    # Show "REC" while recording
    if recording:
        cv2.putText(output, "REC", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show "Saved!" after screenshot
    if saved_time and time.time() - saved_time < 2:
        cv2.putText(output, "Saved!", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write video if recording
    if recording and video_writer is not None:
        video_writer.write(output)

    cv2.imshow("Hand Drawing", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        filename = f"screenshots/screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, output)
        saved_time = time.time()
        print(f"Screenshot saved: {filename}")
    elif key == ord('r'):
        if not recording:
            filename = f"recordings/record_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            recording = True
            print(f"Recording started: {filename}")
        else:
            recording = False
            if video_writer is not None:
                video_writer.release()
            print("Recording stopped.")

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()