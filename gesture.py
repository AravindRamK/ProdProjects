import os
import cv2
import mediapipe as mp
from glob import glob

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to process a single image
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Process with MediaPipe (using BGR directly)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image)

        # Draw landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Finger count logic
                finger_status = []
                # Thumb: Compare landmark 4 and landmark 3
                finger_status.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
                # Other fingers: Compare tip and PIP landmarks
                finger_status.extend([
                    hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,   # Index
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, # Middle
                    hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, # Ring
                    hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Pinky
                ])
                finger_count = sum(finger_status)
                print(f"Processed {image_path}: {finger_count} fingers detected")
        else:
            print(f"No hands detected in {image_path}")

    # Save the annotated image (optional)
    output_path = os.path.join("output", os.path.basename(image_path))
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, image)

# Main directory path
base_dir = "C:/Users/karav_867vu4n/OneDrive/Desktop/Handgesture/leapGestRecog"  # Replace with your dataset folder path

# Recursively find all images in subfolders
image_paths = glob(os.path.join(base_dir, "**", "*.png"), recursive=True) + \
              glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True) + \
              glob(os.path.join(base_dir, "**", "*.jpeg"), recursive=True)

# Process each image
for image_path in image_paths:
    process_image(image_path)

print("Processing complete!")
