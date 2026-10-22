import cv2
import numpy as np
import pygame
from cvzone.HandTrackingModule import HandDetector

# Initialize Pygame Audio Mixer
pygame.mixer.init()

# --- AUDIO GENERATOR HELPER ---
# Creates clean musical sound waves dynamically
def generate_sine_wave(frequency, duration=0.6, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * 32767
    wave = wave.astype(np.int16)
    stereo_wave = np.repeat(wave[:, np.newaxis], 2, axis=1)
    return pygame.mixer.Sound(buffer=stereo_wave)

# --- THE COMPLETE SAPTAK FREQUENCIES (Middle C / C4 Scale) ---
# Sa = C4, Re = D4, Ga = E4, Ma = F4, Pa = G4, Dha = A4, Ni = B4, High Sa = C5
SAPTAK_FREQ = {
    'Sa': 261.63,   # Lower Sa (C4)
    'Re': 293.66,   # D4
    'Ga': 329.63,   # E4
    'Ma': 349.23,   # F4
    'Pa': 392.00,   # G4
    'Dha': 440.00,  # A4
    'Ni': 493.88,   # B4
    "Sa'": 523.25   # Higher Sa (C5)
}

# Generate sounds for all 8 notes automatically
sounds = {note: generate_sine_wave(freq) for note, freq in SAPTAK_FREQ.items()}

# Initialize Webcam and Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Tracks the last played note to prevent glitchy repeats
last_pressed_note = None 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Calculate width dynamically for 8 keys
    key_width = w // 8
    
    # Define our 8 Saptak Notes
    note_names = ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni', "Sa'"]
    
    # Build the keys configuration layout
    keys = []
    for i, name in enumerate(note_names):
        # Alternate background colors slightly so you can see separate keys clearly
        bg_color = (245, 245, 245) if i % 2 == 0 else (255, 255, 255)
        keys.append({
            'name': name,
            'x1': i * key_width,
            'x2': (i + 1) * key_width if i < 7 else w, # Last key stretches to screen edge
            'color': bg_color
        })

    # --- DRAW THE SAPTAK KEYBOARD ---
    for key in keys:
        # Draw the key block (top 200 pixels of the screen)
        cv2.rectangle(frame, (key['x1'], 0), (key['x2'], 200), key['color'], cv2.FILLED)
        # Draw clean black border lines between keys
        cv2.rectangle(frame, (key['x1'], 0), (key['x2'], 200), (0, 0, 0), 2)
        # Text Label for Swaras
        cv2.putText(frame, key['name'], (key['x1'] + (key_width // 2) - 15, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Find Hand Tracking Landmarks
    hands, frame = detector.findHands(frame, draw=False)

    current_note_detected = None

    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        
        # Get Index Finger Tip coordinates
        idx_x, idx_y = lmList[8][0], lmList[8][1]
        fingers = detector.fingersUp(hand)

        # TRIGGER ACTION: Only Index finger up ☝️
        if fingers[1] == 1 and fingers[2] == 0:
            # Draw your active virtual musical pointer mallet
            cv2.circle(frame, (idx_x, idx_y), 12, (0, 0, 255), cv2.FILLED)

            # Check if your finger tip is touching the piano key area (y < 200)
            if idx_y < 200:
                for key in keys:
                    if key['x1'] < idx_x < key['x2']:
                        current_note_detected = key['name']
                        # Flash the pressed key Yellow
                        cv2.rectangle(frame, (key['x1'], 0), (key['x2'], 200), (0, 255, 255), cv2.FILLED)
                        cv2.putText(frame, key['name'], (key['x1'] + (key_width // 2) - 15, 140), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # --- PLAY AUDIO ON NEW NOTE PRESS ---
    if current_note_detected and current_note_detected != last_pressed_note:
        sounds[current_note_detected].play()
        last_pressed_note = current_note_detected
    elif not current_note_detected:
        last_pressed_note = None

    # Show Application window
    cv2.imshow("Air-Saptak Indian Harmonium Studio", frame)

    # Press 'q' to close the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()