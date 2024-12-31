import os
import cv2
import mediapipe as mp
from glob import glob


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def process_image(image_path):
 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image)

    
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
             
                finger_status = []
              
                finger_status.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
     
                finger_status.extend([
                    hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y, 
                    hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y, 
                    hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  
                ])
                finger_count = sum(finger_status)
                print(f"Processed {image_path}: {finger_count} fingers detected")
        else:
            print(f"No hands detected in {image_path}")

   
    output_path = os.path.join("output", os.path.basename(image_path))
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, image)


base_dir = "C:/Users/karav_867vu4n/OneDrive/Desktop/Handgesture/leapGestRecog" 


image_paths = glob(os.path.join(base_dir, "**", "*.png"), recursive=True) + \
              glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True) + \
              glob(os.path.join(base_dir, "**", "*.jpeg"), recursive=True)


for image_path in image_paths:
    process_image(image_path)

print("Processing complete!")
