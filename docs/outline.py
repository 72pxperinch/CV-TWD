import cv2
import numpy as np

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges_canny = cv2.Canny(blurred, 50, 150)
        
        # Sobel filters for gradient-based edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal gradients
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical gradients
        sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
        sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
        
        # Apply Laplacian filter for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = cv2.convertScaleAbs(laplacian)

        # Prewitt kernels for edge detection
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(gray, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray, -1, kernel_y)
        prewitt_edges = cv2.add(prewitt_x, prewitt_y)

        # Dilation followed by erosion for morphological edge detection
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        eroded = cv2.erode(gray, kernel, iterations=1)
        morph_edges = cv2.absdiff(dilated, eroded)

        # Resize the images for smaller window size
        width, height = 640, 360
        edges_canny_resized = cv2.resize(edges_canny, (width, height))
        sobel_magnitude_resized = cv2.resize(sobel_magnitude, (width, height))
        laplacian_abs_resized = cv2.resize(laplacian_abs, (width, height))
        prewitt_edges_resized = cv2.resize(prewitt_edges, (width, height))
        morph_edges_resized = cv2.resize(morph_edges, (width, height))

        # Display results in separate windows
        cv2.imshow("Canny Edges", edges_canny_resized)
        cv2.imshow("Sobel Edges", sobel_magnitude_resized)
        cv2.imshow("Laplacian Edges", laplacian_abs_resized)
        cv2.imshow("Prewitt Edges", prewitt_edges_resized)
        cv2.imshow("Morphological Edges", morph_edges_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
