import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Pose solution setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Cubic Bezier function
def bezier_curve(p0, p1, p2, p3, num_points=100):
    curve_points = []
    for t in np.linspace(0, 1, num_points):
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        curve_points.append((int(x), int(y)))
    return np.array(curve_points, dtype=np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Convert the frame back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw the pose landmarks and Bezier curves around each connection if detected
    if results.pose_landmarks:
        # Loop through the connections and draw Bezier curves around each connection
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            
            # Get the coordinates of the start and end points
            start_x, start_y = int(start_landmark.x * frame_bgr.shape[1]), int(start_landmark.y * frame_bgr.shape[0])
            end_x, end_y = int(end_landmark.x * frame_bgr.shape[1]), int(end_landmark.y * frame_bgr.shape[0])
            
            # Control points for the cubic Bezier curve (example: use mid-point and control point above/below the line)
            control_point1 = (int((start_x + end_x) / 2), int((start_y + end_y) / 2) - 30)  # above the line
            control_point2 = (int((start_x + end_x) / 2), int((start_y + end_y) / 2) + 30)  # below the line
            
            # Define the Bezier curve
            bezier_points = bezier_curve((start_x, start_y), control_point1, control_point2, (end_x, end_y))
            
            # Draw the Bezier curve on the frame
            cv2.polylines(frame_bgr, [bezier_points], isClosed=False, color=(0, 255, 0), thickness=2)

            # Calculate the center and axes of the ellipse
            center = ((start_x + end_x) // 2, (start_y + end_y) // 2)
            
            # Calculate the distance between the two landmarks
            dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            
            # Semi-major and semi-minor axes (ellipse size)
            axes = (int(dist / 2), 50)  # Adjust 15 for vertical size

            # Calculate angle for rotation (in degrees)
            angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))

            # Draw the ellipse with the calculated parameters
            cv2.ellipse(frame_bgr, center, axes, angle, 0, 360, (0, 255, 0), 2)

            for landmark in results.pose_landmarks.landmark:
            # Get the coordinates of the landmark
                x, y = int(landmark.x * frame_bgr.shape[1]), int(landmark.y * frame_bgr.shape[0])
                # Draw a circle at the landmark's position (radius of 5 pixels, green color)
                cv2.circle(frame_bgr, (x, y), 20, (0, 255, 0), -1)  # -1 means filled circle
    
    
    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame_bgr, 1)

    # Show the flipped frame with the body outline and Bezier curves
    cv2.imshow('Body Outline with Bezier Curves', flipped_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
