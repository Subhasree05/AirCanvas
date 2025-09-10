import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import datetime
import os
import pickle

# Ensure the session save directory exists
os.makedirs("sessions", exist_ok=True)

# ----------- Helper Functions -----------

def convert_points_to_list(points):
    return [[list(p) for p in color] for color in points]

def convert_list_to_points(saved_list):
    return [deque([tuple(p) for p in color], maxlen=1024) for color in saved_list]

def serialize_history(history):
    return [(color, idx, list(stroke.copy())) for color, idx, stroke in history]

def deserialize_history(history):
    return [(color, idx, deque(stroke, maxlen=1024)) for color, idx in history]

# ----------- Initialization -----------

# Stroke storage
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = green_index = red_index = yellow_index = 0

# History
draw_history = []
redo_stack = []
cleared_history = []

 
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR: Blue, Green, Red, Yellow
colorIndex = 0

paintWindow = np.ones((520, 680, 3), dtype=np.uint8) * 255
button_height = 60

# UI Color Selection Buttons: (x, y, color)
color_circles = [
    (230, 30, (0, 0, 255)),     # Red
    (290, 30, (0, 255, 0)),     # Green
    (350, 30, (255, 0, 0)),     # Blue
    (410, 30, (0, 255, 255))# Yellow
]

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 560)

# ----------- Main Loop -----------
ret = True
while ret:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.resize(frame, (720, 560))
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw Toolbar
    cv2.rectangle(frame, (0, 0), (720, button_height), (35, 45, 65), -1)
    cv2.putText(frame, "AirCanvas", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, (x, y, color) in enumerate(color_circles):
        cv2.circle(frame, (x, y), 25, color, -1)
        if i == colorIndex:
            cv2.circle(frame, (x, y), 27, (255, 255, 255), 2)

    cv2.rectangle(frame, (470, 10), (550, 40), (150, 150, 150), -1)
    cv2.putText(frame, "Undo", (475, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.rectangle(frame, (560, 10), (640, 40), (150, 150, 150), -1)
    cv2.putText(frame, "Redo", (565, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.line(frame, (0, button_height), (720, button_height), (0, 0, 0), 2)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 720)
                lmy = int(lm.y * 560)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        middle_finger = (landmarks[12][0], landmarks[12][1])

        ring_tip = landmarks[16]
        ring_base = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_base = landmarks[18]

        is_ring_folded = ring_tip[1] > ring_base[1]
        is_pinky_folded = pinky_tip[1] > pinky_base[1]

        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        peace_distance = np.linalg.norm(np.array(fore_finger) - np.array(middle_finger))
        if peace_distance < 30 and is_ring_folded and is_pinky_folded:
            cleared_history = draw_history.copy()
            draw_history.clear()
            redo_stack.clear()
            bpoints = [deque(maxlen=1024)]; blue_index = 0
            gpoints = [deque(maxlen=1024)]; green_index = 0
            rpoints = [deque(maxlen=1024)]; red_index = 0
            ypoints = [deque(maxlen=1024)]; yellow_index = 0
            paintWindow = np.ones((560, 720, 3), dtype=np.uint8) * 255

        elif (thumb[1] - center[1] < 30):
            bpoints.append(deque(maxlen=1024)); blue_index += 1
            gpoints.append(deque(maxlen=1024)); green_index += 1
            rpoints.append(deque(maxlen=1024)); red_index += 1
            ypoints.append(deque(maxlen=1024)); yellow_index += 1

        elif center[1] <= button_height:
            if 470 <= center[0] <= 550:
                if draw_history:
                    color, idx, stroke = draw_history.pop()
                    redo_stack.append((color, idx, stroke))
                    if color == 0 and idx < len(bpoints): bpoints[idx].clear()
                    elif color == 1 and idx < len(gpoints): gpoints[idx].clear()
                    elif color == 2 and idx < len(rpoints): rpoints[idx].clear()
                    elif color == 3 and idx < len(ypoints): ypoints[idx].clear()

            elif 560 <= center[0] <= 640:
                if redo_stack:
                    color, idx, stroke = redo_stack.pop()
                    draw_history.append((color, idx, stroke))
                    target_list = [bpoints, gpoints, rpoints, ypoints][color]
                    while len(target_list) <= idx:
                        target_list.append(deque(maxlen=1024))
                    target_list[idx] = deque(stroke, maxlen=1024)

            else:
                for i, (x, y, _) in enumerate(color_circles):
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= 25 ** 2:
                        colorIndex = i
                        break

        else:
            if colorIndex == 0:
                rpoints[red_index].appendleft(center)
                draw_history.append((2, red_index, deque(rpoints[red_index], maxlen=1024)))
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
                draw_history.append((1, green_index, deque(gpoints[green_index], maxlen=1024)))
            elif colorIndex == 2:
                bpoints[blue_index].appendleft(center)
                draw_history.append((0, blue_index, deque(bpoints[blue_index], maxlen=1024)))
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
                draw_history.append((3, yellow_index, deque(ypoints[yellow_index], maxlen=1024)))

    else:
        bpoints.append(deque(maxlen=1024)); blue_index += 1
        gpoints.append(deque(maxlen=1024)); green_index += 1
        rpoints.append(deque(maxlen=1024)); red_index += 1
        ypoints.append(deque(maxlen=1024)); yellow_index += 1

    for i, color_points in enumerate([bpoints, gpoints, rpoints, ypoints]):
        for stroke in color_points:
            for k in range(1, len(stroke)):
                if stroke[k - 1] is None or stroke[k] is None:
                    continue
                cv2.line(frame, stroke[k - 1], stroke[k], colors[i], 2)
                cv2.line(paintWindow, stroke[k - 1], stroke[k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        canvas_only = paintWindow[button_height+2:, :, :]
        filename = f"AirCanvas_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas_only)
        print(f"Canvas saved as {filename}")
    elif key == ord('x'):
        ...  # Add save session logic
    elif key == ord('l'):
        ...  # Add load session logic

cap.release()
cv2.destroyAllWindows()
