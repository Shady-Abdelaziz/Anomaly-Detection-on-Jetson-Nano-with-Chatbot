import cv2

# Open the camera
cap = cv2.VideoCapture("rtsp://admin:admin@192.168.251.3/video.mjpg")

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Capture frames in a loop for live display
while True:
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame")
        break

    # Display the live video
    cv2.imshow('Live Camera Feed', frame)

    # Capture and save an image when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('captured_image.jpg', frame)
        print("Image saved as captured_image.jpg")

    # Exit the loop when 'q' is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

