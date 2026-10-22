import cv2
import numpy as np
import pygame
from cvzone.HandTrackingModule import HandDetector

# Initialize Pygame Audio Mixer
pygame.mixer.init()

# --- AUDIO GENERATOR HELPER ---
# Automatically creates piano sound waves on the fly
def generate_sine_wave(frequency, duration=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * 32767
    wave = wave.astype(np.int16)
    # Duplicate for stereo layout
    stereo_wave = np.repeat(wave[:, np.newaxis], 2, axis=1)
    # FIX: Changed pygame.sound.Sound to pygame.mixer.Sound
    return pygame.mixer.Sound(buffer=stereo_wave)

# Map notes to standard piano frequencies (C4, D4, E4, F4, G4)
NOTE_FREQ = {'C': 261.63, 'D': 293.66, 'E': 329.63, 'F': 349.23, 'G': 392.00}
sounds = {note: generate_sine_wave(freq) for note, freq in NOTE_FREQ.items()}

# Initialize Webcam and Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Tracks which key was pressed in the previous frame to prevent echo loops
last_pressed_key = None 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Define our 5 Piano Keys dynamically based on screen width
    key_width = w // 5
    keys = [
        {'name': 'C', 'x1': 0, 'x2': key_width, 'color': (240, 240, 240)},
        {'name': 'D', 'x1': key_width, 'x2': key_width * 2, 'color': (255, 255, 255)},
        {'name': 'E', 'x1': key_width * 2, 'x2': key_width * 3, 'color': (240, 240, 240)},
        {'name': 'F', 'x1': key_width * 3, 'x2': key_width * 4, 'color': (255, 255, 255)},
        {'name': 'G', 'x1': key_width * 4, 'x2': w, 'color': (240, 240, 240)}
    ]

    # --- DRAW THE PIANO KEYS ---
    for key in keys:
        cv2.rectangle(frame, (key['x1'], 0), (key['x2'], 200), key['color'], cv2.FILLED)
        cv2.rectangle(frame, (key['x1'], 0), (key['x2'], 200), (0, 0, 0), 3)
        cv2.putText(frame, key['name'], (key['x1'] + (key_width // 2) - 15, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # Find Hand Tracking Landmarks
    hands, frame = detector.findHands(frame, draw=False)

    current_key_detected = None

    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        
        # Get Index Finger Tip coordinates
        idx_x, idx_y = lmList[8][0], lmList[8][1]
        fingers = detector.fingersUp(hand)

        # TRIGGER ACTION: Hand is in "Pointer Mode" (Index finger up, others down)
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(frame, (idx_x, idx_y), 15, (0, 0, 255), cv2.FILLED)

            # Check if your finger tip is touching the piano key area (Y-axis less than 200)
            if idx_y < 200:
                for key in keys:
                    if key['x1'] < idx_x < key['x2']:
                        current_key_detected = key['name']
                        # Highlight the pressed key on screen
                        cv2.rectangle(frame, (key['x1'], 0), (key['x2'], 200), (0, 255, 255), cv2.FILLED)
                        cv2.putText(frame, key['name'], (key['x1'] + (key_width // 2) - 15, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # --- PLAY AUDIO ON NEW KEY PRESS ---
    if current_key_detected and current_key_detected != last_pressed_key:
        sounds[current_key_detected].play()
        last_pressed_key = current_key_detected
    elif not current_key_detected:
        last_pressed_key = None

    # Show Application window
    cv2.imshow("Air-Piano Studio", frame)

    # Press 'q' to close the piano
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()