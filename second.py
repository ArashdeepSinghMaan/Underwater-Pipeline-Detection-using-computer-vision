**FINE TUNING PIPELINE**

ADAPTIVEW THRESHOLD FOR CANNY EDGE DETECTOR


import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Apply Gaussian filter with default kernel size (5x5)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image to grayscale
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny edge detector
    edges = cv2.Canny(adaptive_thresh, 20, 50)  # Adjust thresholds as needed

    # Apply probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a blank canvas to draw lines
    line_image = np.zeros_like(image)

    # Draw detected lines on the blank canvas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay the lines on the original image
    result_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # Display images individually
     # Display images individually
    cv2_imshow(image)  # Original Image
    cv2_imshow(blurred_image)  # Blurred Image
    cv2_imshow(adaptive_thresh)#"Adaptive Threshold",
    cv2_imshow(edges)  # Edges
    cv2_imshow(result_image)  # Original Image with Detected Lines



    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)


gaussain,Canny,Adaptive,Morphological,Hough

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Apply Gaussian filter with default kernel size (5x5)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image to grayscale
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny edge detector
    edges = cv2.Canny(adaptive_thresh, 20, 50)  # Adjust thresholds as needed

    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=3)
    edges = cv2.dilate(edges, kernel, iterations=3)

    # Apply probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a blank canvas to draw lines
    line_image = np.zeros_like(image)

    # Draw detected lines on the blank canvas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay the lines on the original image
    result_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # Display images individually
    cv2_imshow(image)  # Original Image
    cv2_imshow(blurred_image)  # Blurred Image
    cv2_imshow(adaptive_thresh)#"Adaptive Threshold",
    cv2_imshow(edges)  # Edges
    cv2_imshow(result_image)  # Original Image with Detected Lines

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpegg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)


# **gaussain,Canny,Adaptive,gaussian,Hough**





import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Apply Gaussian filter with default kernel size (5x5)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image to grayscale
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny edge detector
    edges = cv2.Canny(adaptive_thresh, 20, 50)  # Adjust thresholds as needed

    # Apply Gaussian blur to remove noise produced by Canny edge detector
    edges_blurred = cv2.GaussianBlur(edges, (5, 5), 0)

    # Apply probabilistic Hough transform on blurred edges
    lines = cv2.HoughLinesP(edges_blurred, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)

    # Create a blank canvas to draw lines
    line_image = np.zeros_like(image)

    # Draw detected lines on the blank canvas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay the lines on the original image
    result_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # Display images individually
    cv2_imshow( image)  # Original Image
    cv2_imshow( blurred_image)  # Blurred Image
    cv2_imshow( adaptive_thresh)  # Adaptive Threshold
    cv2_imshow( edges)  # Edges
    cv2_imshow( edges_blurred)  # Edges Blurred
    cv2_imshow( result_image)  # Original Image with Detected Lines

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpegg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)


White Balancing,Contrast Limited Adaptive Histogram Equalization,gaussain,Canny,Adaptive,gaussian,Hough

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

# Function to preprocess each image
def preprocess_image(image):
    # White balancing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wb = cv2.xphoto.createSimpleWB()
    gray_balanced = wb.balanceWhite(gray)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray_balanced)

    return clahe_output

# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Apply Gaussian filter with default kernel size (5x5)
    blurred_image = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Canny edge detector
    edges = cv2.Canny(adaptive_thresh, 20, 50)  # Adjust thresholds as needed

    # Apply Gaussian blur to remove noise produced by Canny edge detector
    edges_blurred = cv2.GaussianBlur(edges, (5, 5), 0)

    # Apply probabilistic Hough transform on blurred edges
    lines = cv2.HoughLinesP(edges_blurred, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)

    # Create a blank canvas to draw lines
    line_image = np.zeros_like(image)

    # Draw detected lines on the blank canvas
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay the lines on the original image
    result_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # Display images individually
    cv2_imshow(image)  # Original Image
    cv2_imshow(blurred_image)  # Blurred Image
    cv2_imshow(adaptive_thresh)  # Adaptive Threshold
    cv2_imshow(edges)  # Edges
    cv2_imshow(edges_blurred)  # Edges Blurred
    cv2_imshow(result_image)  # Original Image with Detected Lines

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpegg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)


After detecting edges, contours are found in the edge image using cv2.findContours.
Contours with areas below a certain threshold are filtered out.
Remaining contours are drawn on a blank canvas.
The Hough transform is applied on this canvas to detect lines associated with long objects.

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

# Function to preprocess each image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply white balancing
    wb = cv2.xphoto.createSimpleWB()
    gray_balanced = wb.balanceWhite(gray)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray_balanced)

    return clahe_output

# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Apply edge detection
    edges = cv2.Canny(preprocessed_image, 20, 50)  # Adjust thresholds as needed

    # Apply Hough transform on edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)

    # Draw detected lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display images individually
    cv2_imshow(image)  # Original Image with Detected Lines

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpegg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)


**Addition of Region of Interest**

import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

# Function to preprocess each image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply white balancing
    wb = cv2.xphoto.createSimpleWB()
    gray_balanced = wb.balanceWhite(gray)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray_balanced)

    return clahe_output

# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Define region of interest (e.g., a rectangular area)
    # Coordinates: (x1, y1) = top-left corner, (x2, y2) = bottom-right corner
    x1, y1, x2, y2 = 100, 100, 400, 300  # Example coordinates
    roi = image[y1:y2, x1:x2]

    # Preprocess the region of interest
    preprocessed_roi = preprocess_image(roi)

    # Apply edge detection to the region of interest
    edges_roi = cv2.Canny(preprocessed_roi, 20, 50)

    # Apply Hough transform on edges within the region of interest
    lines_roi = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=100, minLineLength=150, maxLineGap=10)

    # Draw detected lines on the original image within the region of interest
    if lines_roi is not None:
        for line in lines_roi:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the original image with the region of interest and detected lines
    cv2_imshow(roi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpegg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)


Modified Adaptive ROI + White Balancing,Contrast Limited Adaptive Histogram Equalization,gaussain,Canny,Adaptive,gaussian,Hough


from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import os

# Function to preprocess each image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply white balancing
    wb = cv2.xphoto.createSimpleWB()
    gray_balanced = wb.balanceWhite(gray)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray_balanced)

    return clahe_output

# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Apply edge detection
    edges = cv2.Canny(preprocessed_image, 20, 50)  # Adjust thresholds as needed

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define region of interest around each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]  # Define ROI around contour

        # Apply additional processing within the ROI (e.g., further edge detection, line detection)
        # Example: Apply Hough transform on edges within the ROI
        edges_roi = cv2.Canny(preprocessed_image[y:y+h, x:x+w], 20, 50)
        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=90, minLineLength=200, maxLineGap=10)

        # Draw detected lines on the original image within the ROI
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the original image with detected lines within the ROIs
    cv2_imshow(image)

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)  # Corrected path concatenation
        print("Processing:", image_path)
        process_image(image_path)


Modified Adaptive ROI + White Balancing,Contrast Limited Adaptive Histogram Equalization,gaussain,Canny,Adaptive,gaussian, Probalistic -Hough

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import os

# Function to preprocess each image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply white balancing
    wb = cv2.xphoto.createSimpleWB()
    gray_balanced = wb.balanceWhite(gray)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray_balanced)

    return clahe_output

# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Apply edge detection
    edges = cv2.Canny(preprocessed_image, 10, 40)  # Adjust thresholds as needed

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define region of interest around each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]  # Define ROI around contour

        # Apply additional processing within the ROI (e.g., further edge detection, line detection)
        # Example: Apply Probabilistic Hough transform on edges within the ROI
        edges_roi = cv2.Canny(preprocessed_image[y:y+h, x:x+w], 20, 50)
        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=60, minLineLength=200, maxLineGap=10)

        # Draw detected lines on the original image within the ROIs
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the original image with detected lines within the ROIs
    cv2_imshow( image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        image_path = os.path.join(folder_path, filename)  # Corrected path concatenation
        print("Processing:", image_path)
        process_image(image_path)


White balancing,Contrast limited Adaptive Histogram Equalization,Gaussian,Adaptive Theshold for canny edge detector, Again gausian,Probalistic Hough transform,Adaptive Region of Interest . Also a adpative feed back loop make this pipeline robust.


# ** FINAL**

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import os

# Function to preprocess each image
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply white balancing
    wb = cv2.xphoto.createSimpleWB()
    gray_balanced = wb.balanceWhite(gray)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_output = clahe.apply(gray_balanced)

    return clahe_output

# Function to process each image
def process_image(image_path):
    # Read input image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Apply Gaussian filtering
    blurred = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)

    # Adaptive thresholding for Canny edge detection
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define adaptive region of interest around each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]  # Define ROI around contour

        # Apply additional processing within the ROI (e.g., further edge detection, line detection)
        # Example: Apply Probabilistic Hough transform on edges within the ROI
        edges_roi = cv2.Canny(blurred[y:y+h, x:x+w], 50, 150)
        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=10)

        # Draw detected lines on the original image within the ROIs
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the original image with detected lines within the ROIs
    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE/'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):  # Consider only image files
        image_path = os.path.join(folder_path, filename)  # Corrected path concatenation
        print("Processing:", image_path)
        process_image(image_path)
