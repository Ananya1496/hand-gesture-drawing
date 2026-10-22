import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize the webcam and the hand detector
cap = cv2.VideoCapture(0)
# Lowering detection confidence slightly to make tracking smoother while drawing
detector = HandDetector(detectionCon=0.75, maxHands=1)

# Variables for drawing
canvas = None
xp, yp = 0, 0  # Previous coordinates

# Color definitions (OpenCV uses BGR format: Blue, Green, Red)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
ERASER = (0, 0, 0) # Drawing black on the canvas clears it!

# Default startup tools
brush_color = RED
brush_thickness = 5

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip the frame horizontally for a natural mirror experience
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Initialize the black canvas if it hasn't been created yet
    if canvas is None:
        canvas = np.zeros_like(frame)

    # --- DRAW THE COLOR PALETTE INTERFACE AT THE TOP ---
    # Red Box
    cv2.rectangle(frame, (0, 0), (160, 60), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "RED", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blue Box
    cv2.rectangle(frame, (160, 0), (320, 60), (255, 0, 0), cv2.FILLED)
    cv2.putText(frame, "BLUE", (210, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Green Box
    cv2.rectangle(frame, (320, 0), (480, 60), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, "GREEN", (365, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Eraser Box
    cv2.rectangle(frame, (480, 0), (w, 60), (150, 150, 150), cv2.FILLED)
    cv2.putText(frame, "ERASER", (530, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Find the hand
    hands, frame = detector.findHands(frame, draw=False)

    if hands:
        hand = hands[0]
        lmList = hand['lmList']

        # Get coordinates of Index finger tip (8) and Middle finger tip (12)
        idx_x, idx_y = lmList[8][0], lmList[8][1]
        mid_x, mid_y = lmList[12][0], lmList[12][1]

        # Check which fingers are up
        fingers = detector.fingersUp(hand)

        # 1. SELECTION MODE: Both Index and Middle fingers are up (Use this to choose color)
        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0  # Reset line anchor
            
            # Draw a white pointer circle
            cv2.circle(frame, (idx_x, idx_y), 15, (255, 255, 255), cv2.FILLED)
            
            # Check if your pointer is inside the top menu bar (y < 60)
            if idx_y < 60:
                if 0 < idx_x < 160:
                    brush_color = RED
                    brush_thickness = 5
                elif 160 < idx_x < 320:
                    brush_color = BLUE
                    brush_thickness = 5
                elif 320 < idx_x < 480:
                    brush_color = GREEN
                    brush_thickness = 5
                elif 480 < idx_x < w:
                    brush_color = ERASER
                    brush_thickness = 50 # Make eraser thicker so it cleans faster!

            cv2.putText(frame, "Selecting Color...", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. DRAWING MODE: Only Index finger is up
        elif fingers[1] == 1 and fingers[2] == 0:
            # Draw a circle showing current active color on tip of finger
            cv2.circle(frame, (idx_x, idx_y), 10, brush_color, cv2.FILLED)
            
            if brush_color == ERASER:
                cv2.putText(frame, "Erasing Mode", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Drawing Mode", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, brush_color, 2)
            
            if xp == 0 and yp == 0:
                xp, yp = idx_x, idx_y

            # Draw onto our persistent canvas
            cv2.line(canvas, (xp, yp), (idx_x, idx_y), brush_color, brush_thickness)
            xp, yp = idx_x, idx_y
        else:
            xp, yp = 0, 0
    else:
        xp, yp = 0, 0

    # Combine canvas drawings smoothly with the live webcam stream
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    # Show the canvas window
    cv2.imshow("Multi-Color Hand Gesture Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()