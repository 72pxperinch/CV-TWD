import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe Pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

# Define the pose connections (joint connections) from MediaPipe
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Convert the frame back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw the pose landmarks and ellipses around each connection if detected
    if results.pose_landmarks:
        # Loop through the connections and draw ellipses around each connected landmark
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            
            # Get the coordinates of the start and end points
            start_x, start_y = int(start_landmark.x * frame_bgr.shape[1]), int(start_landmark.y * frame_bgr.shape[0])
            end_x, end_y = int(end_landmark.x * frame_bgr.shape[1]), int(end_landmark.y * frame_bgr.shape[0])
            
            # Calculate the center and axes of the ellipse
            center = ((start_x + end_x) // 2, (start_y + end_y) // 2)
            
            # Calculate the distance between the two landmarks
            dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            
            # Semi-major and semi-minor axes (ellipse size)
            axes = (int(dist / 2), 15)  # Adjust 15 for vertical size

            # Calculate angle for rotation (in degrees)
            angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))

            # Draw the ellipse with the calculated parameters
            cv2.ellipse(frame_bgr, center, axes, angle, 0, 360, (0, 255, 0), 2)

    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame_bgr, 1)

    # Show the flipped frame with the body outline and ellipses around joint connections
    cv2.imshow('Body Outline with Ellipses Around Connections', flipped_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
