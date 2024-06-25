import cv2
import numpy as np

last_face_center = None
last_face_size = None

def detect_facial_features(frame):
    global last_face_center, last_face_size
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Create a region of interest (ROI) that's 80% of the frame
    roi_width = int(width * 0.8)
    roi_height = int(height * 0.8)
    roi_x = (width - roi_width) // 2
    roi_y = (height - roi_height) // 2
    roi = gray[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    
    feature_map = np.zeros((roi_height, roi_width))

    # Edge detection
    edges = cv2.Canny(roi, 50, 150)
    
    # Detect circular shapes (potential eyes)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=30, minRadius=15, maxRadius=40)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(feature_map, (i[0], i[1]), i[2], 0.2, -1)
    
    # Detect horizontal lines (potential mouth)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 15:  # Nearly horizontal lines
                cv2.line(feature_map, (x1, y1), (x2, y2), 0.2, 2)
    
    # Detect vertical gradients (potential nose)
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    vertical_edges = np.where((magnitude > 50) & (abs(direction) > 70), 1, 0).astype(np.uint8)
    feature_map += vertical_edges * 0.2
    
    # Detect skin color
    hsv = cv2.cvtColor(frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width], cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    feature_map += skin_mask * 0.2 / 255

    # Find the region with the highest combined weight
    kernel = np.ones((40, 40), np.float32) / 1600  # 40x40 averaging filter
    conv_result = cv2.filter2D(feature_map, -1, kernel)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(conv_result)

    face_center_x, face_center_y = max_loc
    face_size = min(roi_width, roi_height) // 3  # Initial estimate

    # Apply smoothing to reduce jitter
    if last_face_center is not None and last_face_size is not None:
        face_center_x = int(0.7 * last_face_center[0] + 0.3 * face_center_x)
        face_center_y = int(0.7 * last_face_center[1] + 0.3 * face_center_y)
        face_size = int(0.7 * last_face_size + 0.3 * face_size)

    last_face_center = (face_center_x, face_center_y)
    last_face_size = face_size

    # Adjust coordinates to full frame
    face_center_x += roi_x
    face_center_y += roi_y

    # Draw rectangle around the detected face region
    top_left = (face_center_x - face_size//2, face_center_y - face_size//2)
    bottom_right = (face_center_x + face_size//2, face_center_y + face_size//2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Draw ROI for visualization
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)

    return frame

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame to detect face
    result = detect_facial_features(frame)
    
    # Display result
    cv2.imshow('Face Detection', result)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()