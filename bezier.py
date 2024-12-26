import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate 4 common tangents between two circles
def calculate_tangents(circle1, circle2):
    # Circle1 and Circle2 are tuples of (x, y, radius)
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Distance between the centers of the circles
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # If the circles are too close or one circle is inside the other, no tangents can be drawn
    if dist < abs(r1 - r2) or dist < r1 + r2:
        return []

    # Calculate the angle between the centers of the circles
    angle = np.arctan2(y2 - y1, x2 - x1)

    # External tangents
    external_tangents = []
    for sign in [-1, 1]:
        angle1 = angle + sign * np.arcsin((r1 - r2) / dist)
        external_tangents.append((np.cos(angle1), np.sin(angle1)))

    # Internal tangents
    internal_tangents = []
    for sign in [-1, 1]:
        angle1 = angle + sign * np.arcsin((r1 + r2) / dist)
        internal_tangents.append((np.cos(angle1), np.sin(angle1)))

    # Combine external and internal tangents
    tangents = external_tangents + internal_tangents
    return tangents

# Set the radius for each landmark manually (adjust the values as needed)
def generate_landmark_radius(constant):
    """
    Generates a dictionary of radii for hand landmarks based on a constant value.
    The function adjusts the radii based on the landmark's position.

    :param constant: The constant value that affects the radius for each landmark.
    :return: A dictionary mapping landmark indices to radii.
    """
    # Define the base radii adjustment based on the constant
    radius_table = {
        0: constant,  # Wrist
        1: constant,  # Thumb CMC
        2: constant - 5,  # Thumb MCP
        3: constant - 10, # Thumb IP
        4: constant - 15, # Thumb tip
        5: constant,  # Index finger MCP
        6: constant - 2,  # Index finger PIP
        7: constant - 4,  # Index finger DIP
        8: constant - 6,  # Index finger tip
        9: constant,  # Middle finger MCP
        10: constant - 2,  # Middle finger PIP
        11: constant - 4,  # Middle finger DIP
        12: constant - 6,  # Middle finger tip
        13: constant,  # Ring finger MCP
        14: constant - 2,  # Ring finger PIP
        15: constant - 4,  # Ring finger DIP
        16: constant - 6,  # Ring finger tip
        17: constant,  # Pinky MCP
        18: constant - 2,  # Pinky PIP
        19: constant - 4,  # Pinky DIP
        20: constant - 6   # Pinky tip
    }
    return radius_table

# Example usage
constant_value = 20
generate_landmark_radius(constant_value)



# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    result = hands.process(rgb_frame)

    # Draw landmarks and tangents if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw circles at each landmark with customized radius
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                radius = landmark_radius.get(idx, 20)  # Default radius is 20 if not specified
                cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)  # Draw circle

            # Draw tangents between connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]

                x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])

                # Calculate tangents for the two circles
                tangent_points = calculate_tangents((x1, y1, radius), (x2, y2, radius))

                # Draw tangent lines
                for t in tangent_points:
                    # Extend the line a bit to simulate tangent
                    line_x1 = int(x1 + t[0] * radius)
                    line_y1 = int(y1 + t[1] * radius)
                    line_x2 = int(x2 + t[0] * radius)
                    line_y2 = int(y2 + t[1] * radius)
                    cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 2)

    # Show the frame with hand landmarks and tangents
    cv2.imshow("Hand Pose", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
