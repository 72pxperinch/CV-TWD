import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

def calculate_tangents(circle1, circle2, base_angle):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Distance between centers
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    tangents = []
    for sign in [-1, 1]:
        # Adjust angle for tangent
        tangent_angle = base_angle + -1 * sign * np.pi/2 + sign * np.arcsin(abs(r1 - r2) / dist)

        # Store tangent directions as unit vectors
        tangents.append((np.cos(tangent_angle), np.sin(tangent_angle)))

    return tangents

        
    # internal_tangents = []
    # for sign in [-1, 1]:
    #     angle2 = angle + sign * np.arcsin((r1 + r2) / dist)
    #     external_tangents.append((np.cos(angle2), np.sin(angle2)))


    # tangents = external_tangents + internal_tangents
    # tangents = external_tangents
    # return tangents



min_radius = 15

# Set the radius for each landmark dynamically
def generate_landmark_radius(constant):
    radius_table = {
        0: int(constant*2.5),  # Wrist
        1: int(constant*1.2),  # Thumb CMC
        2: int(constant*1.1),  # Thumb MCP
        3: int(constant*0.95), # Thumb IP
        4: int(constant*0.85), # Thumb tip
        5: int(constant),  # Index finger MCP
        6: int(constant*0.85),  # Index finger PIP
        7: int(constant*0.8),  # Index finger DIP
        8: int(constant*0.7),  # Index finger tip
        9: int(constant),  # Middle finger MCP
        10: int(constant*0.85),  # Middle finger PIP
        11: int(constant*0.8),  # Middle finger DIP
        12: int(constant*0.75),  # Middle finger tip
        13: int(constant),  # Ring finger MCP
        14: int(constant*0.85),  # Ring finger PIP
        15: int(constant*0.8),  # Ring finger DIP
        16: int(constant*0.7),  # Ring finger tip
        17: int(constant),  # Pinky MCP
        18: int(constant*0.75),  # Pinky PIP
        19: int(constant*0.7),  # Pinky DIP
        20: int(constant*0.6)   # Pinky tip
    }
    return radius_table
# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for better user experience
    frame = cv2.flip(frame, 1)
    canvas = np.zeros_like(frame)
    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    result = hands.process(rgb_frame)

    # Draw landmarks and tangents if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_idx , hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_type = result.multi_handedness[hand_idx].classification[0].label
            if hand_type == "Right":
                line_color = (255, 0, 0)
            else :
                line_color = (0, 255, 0)

            # Get the screen coordinates for index MCP and middle finger MCP
            idx_mcp = hand_landmarks.landmark[5]
            mid_mcp = hand_landmarks.landmark[9]

            idx_x, idx_y = int(idx_mcp.x * frame.shape[1]), int(idx_mcp.y * frame.shape[0])
            mid_x, mid_y = int(mid_mcp.x * frame.shape[1]), int(mid_mcp.y * frame.shape[0])

            # Calculate the dynamic constant as half the distance
            distance = np.sqrt((mid_x - idx_x) ** 2 + (mid_y - idx_y) ** 2)
            constant_value = max(min_radius, int(distance / 2))  # Ensure a minimum value of 1
            landmark_radius = generate_landmark_radius(constant_value)


            # Draw tangents between connections
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]

                x1, y1 = int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0])
                x2, y2 = int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0])
                base_angle = np.arctan2(y2 - y1, x2 - x1)
                
                if end_idx in { 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20}:
                    # cv2.ellipse(canvas, (x2,y2), (landmark_radius.get(end_idx, min_radius),landmark_radius.get(end_idx, min_radius)), base_angle*180/np.pi, -90, 90, line_color, 2)
                    cv2.ellipse(frame, (x2,y2), (landmark_radius.get(end_idx, min_radius),landmark_radius.get(end_idx, min_radius)), base_angle*180/np.pi, -90, 90, line_color, 2)
                
                # Calculate tangents for the two circles
                tangent_points = calculate_tangents((x1, y1, landmark_radius.get(start_idx, min_radius)), (x2, y2, landmark_radius.get(end_idx, min_radius)), base_angle)
                # print(tangent_points)
                # Draw tangent lines
                for t in tangent_points:
                    # Extend the line a bit to simulate tangent
                    try:
                        line_x1 = int(x1 + t[0] * landmark_radius.get(start_idx, min_radius))
                        line_y1 = int(y1 + t[1] * landmark_radius.get(start_idx, min_radius))
                        line_x2 = int(x2 + t[0] * landmark_radius.get(end_idx, min_radius))
                        line_y2 = int(y2 + t[1] * landmark_radius.get(end_idx, min_radius))
                    except (ValueError, TypeError) as e:
                        
                        line_x1 = int(x1)
                        line_y1 = int(y1)
                        line_x2 = int(x2)
                        line_y2 = int(y2)
                    # print( start_idx,  end_idx,   x1,y1,  line_x1,line_y1,   x2,y2,  line_x2,line_y2, t[0],t[1])
                    # cv2.line(canvas, (line_x1, line_y1), (line_x2, line_y2), line_color, 2)
                    cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), line_color, 2)
                    # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # print(constant_value, landmark_radius.get(start_idx, min_radius) )

                    
                
                
                    

    # Show the frame with hand landmarks and tangents
    cv2.imshow("Hand Pose", frame)
    # cv2.imshow("Hand Pose", canvas)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
