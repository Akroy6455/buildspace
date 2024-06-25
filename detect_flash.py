import cv2
import numpy as np

def detect_brightest_pixel():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the maximum pixel value and its location
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # Draw a circle around the brightest pixel
        cv2.circle(frame, maxLoc, 10, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Brightest Pixel Detection', frame)

        # Print the brightest pixel location and value
        print(f"Brightest pixel location: {maxLoc}, Value: {maxVal}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function
detect_brightest_pixel()