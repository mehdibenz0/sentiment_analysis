import cv2
import os

# Ask the user for the label of the object (e.g. 'pen', 'glasses')
label = input("Enter the name of the object you are collecting images for: ").strip().lower()
save_dir = os.path.join("data", label)

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Open the webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to save an image. Press ESC to exit.")

count = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the current frame
    cv2.imshow("Image Collector - " + label, frame)
    key = cv2.waitKey(1)

    # If SPACE is pressed, save the image
    if key == 32:  # SPACE
        filename = f"{label}_{count}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved {filepath}")
        count += 1

    # If ESC is pressed, exit
    elif key == 27:  # ESC
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
