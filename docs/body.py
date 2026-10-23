import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

min_radius = 15

# Set dynamic radius for pose landmarks
def generate_pose_radius(constant):
    radius_table = {
        0: int(constant*2.5),  # Nose
        11: int(constant),     # Left shoulder
        12: int(constant),     # Right shoulder
        13: int(constant*0.9), # Left elbow
        14: int(constant*0.9), # Right elbow
        15: int(constant*0.8), # Left wrist
        16: int(constant*0.8), # Right wrist
        23: int(constant),     # Left hip
        24: int(constant),     # Right hip
        25: int(constant*0.9), # Left knee
        26: int(constant*0.9), # Right knee
        27: int(constant*0.8), # Left ankle
        28: int(constant*0.8)  # Right ankle
    }
    return radius_table

def calculate_tangents(circle1, circle2, base_angle):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Distance between centers
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    tangents = []
    for sign in [-1, 1]:
        tangent_angle = base_angle + -1 * sign * np.pi/2 + sign * np.arcsin(abs(r1 - r2) / dist)
        tangents.append((np.cos(tangent_angle), np.sin(tangent_angle)))

    return tangents

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)
    canvas = np.zeros_like(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect pose landmarks
    result = pose.process(rgb_frame)

    # Draw landmarks and tangents if pose landmarks are detected
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Calculate dynamic constant based on frame size
        constant_value = max(min_radius, int(frame.shape[1] / 50))
        pose_radius = generate_pose_radius(constant_value)

        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = landmarks[start_idx]
            end_landmark = landmarks[end_idx]

            x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
            x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
            base_angle = np.arctan2(y2 - y1, x2 - x1)

            # Draw ellipses for specific landmarks
            if end_idx in {11, 12, 13, 14, 23, 24}:  # Key joints
                cv2.ellipse(frame, (x2, y2), (pose_radius.get(end_idx, min_radius), pose_radius.get(end_idx, min_radius)),
                            base_angle * 180 / np.pi, -90, 90, (0, 255, 0), 2)

            # Calculate tangents
            tangent_points = calculate_tangents(
                (x1, y1, pose_radius.get(start_idx, min_radius)),
                (x2, y2, pose_radius.get(end_idx, min_radius)),
                base_angle
            )

            for t in tangent_points:
                try:
                    line_x1 = int(x1 + t[0] * pose_radius.get(start_idx, min_radius))
                    line_y1 = int(y1 + t[1] * pose_radius.get(start_idx, min_radius))
                    line_x2 = int(x2 + t[0] * pose_radius.get(end_idx, min_radius))
                    line_y2 = int(y2 + t[1] * pose_radius.get(end_idx))
                except (ValueError, TypeError):
                    line_x1, line_y1, line_x2, line_y2 = x1, y1, x2, y2

                cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (0, 255, 0), 2)

    # Show the frame with pose landmarks and tangents
    cv2.imshow("Pose Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
