from google.colab import drive
drive.mount('/content/drive')


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert RGB image to LAB color space
def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab_image

# Step 2: Apply Gaussian blur filter
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is fixed to (3, 3)
    return blurred_image

# Step 3: Display the pre-processed image
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check if file is an image
            # Load the image
            image_path = os.path.join(folder_path, file)
            example_image = cv2.imread(image_path)
            example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Pre-processing steps
            lab_image = rgb_to_lab(example_image)
            preprocessed_image = apply_gaussian_blur(lab_image)

            # Display the original and pre-processed images
            plt.figure(figsize=(10, 10))
            plt.subplot(121), plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB))
            plt.title('LAB image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            plt.title('LAB with Gaussian Image'), plt.xticks([]), plt.yticks([])
            plt.show()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert RGB image to LAB color space
def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab_image

# Step 2: Apply Gaussian blur filter
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is fixed to (3, 3)
    return blurred_image

# Step 3: Apply Canny edge detection algorithm
def apply_canny_edge_detection(image, lower_threshold, upper_threshold):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# Step 4: Display the image
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check if file is an image
            # Load the image
            image_path = os.path.join(folder_path, file)
            example_image = cv2.imread(image_path)
            example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Pre-processing steps
            lab_image = rgb_to_lab(example_image)
            preprocessed_image = apply_gaussian_blur(lab_image)

            # Canny edge detection step
            lower_threshold = 10  # Adjust as needed
            upper_threshold = 50  # Adjust as needed
            edges_image = apply_canny_edge_detection(preprocessed_image, lower_threshold, upper_threshold)

            # Display the original, pre-processed, and edge-detected images
            plt.figure(figsize=(15, 5))
            plt.subplot(131), plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB))
            plt.title('LAB image'), plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            plt.title('Smooth LAB Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(133), plt.imshow(edges_image, cmap='gray')
            plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
            plt.show()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert RGB image to LAB color space
def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab_image

# Step 2: Apply Gaussian blur filter
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is fixed to (3, 3)
    return blurred_image

# Step 3: Apply Canny edge detection algorithm
def apply_canny_edge_detection(image, lower_threshold, upper_threshold):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# Step 4: Apply dilation morphological filter
def apply_dilation(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

# Step 5: Apply contour detection
def apply_contour_detection(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Step 6: Filter contours by hierarchy and minimum contour area threshold
def filter_contours(contours, min_area_threshold):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    return filtered_contours

# Step 7: Apply erosion morphological filter
def apply_erosion(image):
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

# Step 8: Display the image
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check if file is an image
            # Load the image
            image_path = os.path.join(folder_path, file)
            example_image = cv2.imread(image_path)
            example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Pre-processing steps
            lab_image = rgb_to_lab(example_image)
            preprocessed_image = apply_gaussian_blur(lab_image)

            # Canny edge detection step
            lower_threshold = 10  # Adjust as needed
            upper_threshold = 50  # Adjust as needed
            edges_image = apply_canny_edge_detection(preprocessed_image, lower_threshold, upper_threshold)

            # Dilation morphological filter step
            dilated_image = apply_dilation(edges_image)

            # Contour detection step
            contours = apply_contour_detection(dilated_image)

            # Filter contours step
            min_area_threshold = 20  # Adjust as needed
            filtered_contours = filter_contours(contours, min_area_threshold)

            # Create a binary image with filtered contours
            filtered_contours_image = np.zeros_like(dilated_image)
            cv2.drawContours(filtered_contours_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

            # Erosion morphological filter step
            eroded_image = apply_erosion(filtered_contours_image)

            # Display the original, pre-processed, edge-detected, dilated, eroded, and filtered contour images
            plt.figure(figsize=(15, 5))
            plt.subplot(161), plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB))
            plt.title('LAB image'), plt.xticks([]), plt.yticks([])
            plt.subplot(162), plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            plt.title('Preprocessed Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(163), plt.imshow(edges_image, cmap='gray')
            plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
            plt.subplot(164), plt.imshow(dilated_image, cmap='gray')
            plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(165), plt.imshow(eroded_image, cmap='gray')
            plt.title('Eroded Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(166), plt.imshow(filtered_contours_image, cmap='gray')
            plt.title('Filtered Contours Image'), plt.xticks([]), plt.yticks([])
            plt.show()


# **LAB+Gaussian+canny+Morphological+Probablistic Hough**

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert RGB image to LAB color space
def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab_image

# Step 2: Apply Gaussian blur filter
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is fixed to (3, 3)
    return blurred_image

# Step 3: Apply Canny edge detection algorithm
def apply_canny_edge_detection(image, lower_threshold, upper_threshold):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# Step 4: Apply dilation morphological filter
def apply_dilation(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

# Step 5: Apply contour detection
def apply_contour_detection(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Step 6: Filter contours by hierarchy and minimum contour area threshold
def filter_contours(contours, min_area_threshold):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    return filtered_contours

# Step 7: Apply erosion morphological filter
def apply_erosion(image):
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

# Step 8: Apply Hough transform
def apply_hough_transform(image, min_line_length=50, max_line_gap=50):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

# Step 9: Merge the detected lines based on minimum distance threshold
def merge_lines(lines, min_distance_threshold):
    merged_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            merged_lines.append([x1, y1, x2, y2])

    return merged_lines

# Step 10: Display the image
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check if file is an image
            # Load the image
            image_path = os.path.join(folder_path, file)
            original_image = cv2.imread(image_path)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Pre-processing steps
            lab_image = rgb_to_lab(original_image_rgb)
            preprocessed_image = apply_gaussian_blur(lab_image)

            # Canny edge detection step
            lower_threshold = 10  # Adjust as needed
            upper_threshold = 30  # Adjust as needed
            edges_image = apply_canny_edge_detection(preprocessed_image, lower_threshold, upper_threshold)

            # Dilation morphological filter step
            dilated_image = apply_dilation(edges_image)

            # Contour detection step
            contours = apply_contour_detection(dilated_image)

            # Filter contours step
            min_area_threshold = 20  # Adjust as needed
            filtered_contours = filter_contours(contours, min_area_threshold)

            # Create a binary image with filtered contours
            filtered_contours_image = np.zeros_like(dilated_image)
            cv2.drawContours(filtered_contours_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

            # Erosion morphological filter step
            eroded_image = apply_erosion(filtered_contours_image)

            # Apply Hough transform
            min_line_length = 20  # Minimum line length
            max_line_gap = 50  # Maximum gap between lines to be considered as single line
            detected_lines = apply_hough_transform(eroded_image, min_line_length, max_line_gap)

            # Merge the detected lines
            min_distance_threshold = 10  # Adjust as needed
            merged_lines = merge_lines(detected_lines, min_distance_threshold)

            # Draw detected lines on the original image
            output_image = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
            if merged_lines is not None:
                for line in merged_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines for detected lines

            # Display the images
            plt.figure(figsize=(15, 5))
            plt.subplot(131), plt.imshow(original_image_rgb)
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(cv2.cvtColor(lab_image, cv2.COLOR_BGR2RGB))
            plt.title('LAB image'), plt.xticks([]), plt.yticks([])
            plt.subplot(133), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.title('Output Image'), plt.xticks([]), plt.yticks([])
            plt.show()


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert RGB image to LAB color space
def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab_image

# Step 2: Apply Gaussian blur filter
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is fixed to (3, 3)
    return blurred_image

# Step 3: Apply Canny edge detection algorithm
def apply_canny_edge_detection(image, lower_threshold, upper_threshold):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# Step 4: Apply dilation morphological filter
def apply_dilation(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

# Step 5: Apply contour detection
def apply_contour_detection(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Step 6: Filter contours by hierarchy and minimum contour area threshold
def filter_contours(contours, min_area_threshold):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    return filtered_contours

# Step 7: Apply erosion morphological filter
def apply_erosion(image):
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

# Step 8: Apply Hough transform
def apply_hough_transform(image, min_line_length=50, max_line_gap=50):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

# Step 9: Merge the detected lines based on minimum distance threshold
def merge_lines(lines, min_distance_threshold):
    merged_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            merged_lines.append([x1, y1, x2, y2])

    return merged_lines

# Step 10: Display the image
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check if file is an image
            # Load the image
            image_path = os.path.join(folder_path, file)
            original_image = cv2.imread(image_path)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Pre-processing steps
            lab_image = rgb_to_lab(original_image_rgb)
            preprocessed_image = apply_gaussian_blur(lab_image)

            # Canny edge detection step
            lower_threshold = 10  # Adjust as needed
            upper_threshold = 40  # Adjust as needed
            edges_image = apply_canny_edge_detection(preprocessed_image, lower_threshold, upper_threshold)

            # Dilation morphological filter step
            dilated_image = apply_dilation(edges_image)

            # Contour detection step
            contours = apply_contour_detection(dilated_image)

            # Filter contours step
            min_area_threshold = 20  # Adjust as needed
            filtered_contours = filter_contours(contours, min_area_threshold)

            # Create a binary image with filtered contours
            filtered_contours_image = np.zeros_like(dilated_image)
            cv2.drawContours(filtered_contours_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

            # Erosion morphological filter step
            eroded_image = apply_erosion(filtered_contours_image)

            # Apply Hough transform
            min_line_length = 20  # Minimum line length
            max_line_gap = 50  # Maximum gap between lines to be considered as single line
            detected_lines = apply_hough_transform(eroded_image, min_line_length, max_line_gap)

            # Merge the detected lines
            min_distance_threshold = 10  # Adjust as needed
            merged_lines = merge_lines(detected_lines, min_distance_threshold)

            # Draw detected lines on the original image
            if merged_lines is not None:
                for line in merged_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines for detected lines

            # Display the original image with detected lines
            plt.figure(figsize=(10, 6))
            plt.imshow(original_image)
            plt.title('Original Image with Detected Lines')
            plt.axis('off')
            plt.show()


## **LAB+Gaussian+canny+Morphological+Probablistic Hough+ Histogram Equallization**

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Convert RGB image to LAB color space
def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab_image

# Step 2: Apply Gaussian blur filter
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is fixed to (3, 3)
    return blurred_image

# Step 3: Apply Canny edge detection algorithm
def apply_canny_edge_detection(image, lower_threshold, upper_threshold):
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges

# Step 4: Apply dilation morphological filter
def apply_dilation(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image

# Step 5: Apply contour detection
def apply_contour_detection(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Step 6: Filter contours by hierarchy and minimum contour area threshold
def filter_contours(contours, min_area_threshold):
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]
    return filtered_contours

# Step 7: Apply erosion morphological filter
def apply_erosion(image):
    kernel = np.ones((2, 2), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

# Step 8: Apply Hough transform
def apply_hough_transform(image, min_line_length=50, max_line_gap=50):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

# Step 9: Merge the detected lines based on minimum distance threshold
def merge_lines(lines, min_distance_threshold):
    merged_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            merged_lines.append([x1, y1, x2, y2])

    return merged_lines

# Step 10: Apply histogram equalization
def apply_histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Step 11: Display the image
def display_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

    # List all files in the folder
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check if file is an image
            # Load the image
            image_path = os.path.join(folder_path, file)
            original_image = cv2.imread(image_path)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Pre-processing steps
            lab_image = rgb_to_lab(original_image_rgb)
            preprocessed_image = apply_gaussian_blur(lab_image)

            # Histogram equalization step
            equalized_image = apply_histogram_equalization(preprocessed_image[:,:,0])  # Apply only on the L channel

            # Canny edge detection step
            lower_threshold = 10  # Adjust as needed
            upper_threshold = 50  # Adjust as needed
            edges_image = apply_canny_edge_detection(equalized_image, lower_threshold, upper_threshold)

            # Dilation morphological filter step
            dilated_image = apply_dilation(edges_image)

            # Contour detection step
            contours = apply_contour_detection(dilated_image)

            # Filter contours step
            min_area_threshold = 20  # Adjust as needed
            filtered_contours = filter_contours(contours, min_area_threshold)

            # Create a binary image with filtered contours
            filtered_contours_image = np.zeros_like(dilated_image)
            cv2.drawContours(filtered_contours_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

            # Erosion morphological filter step
            eroded_image = apply_erosion(filtered_contours_image)

            # Apply Hough transform
            min_line_length = 50  # Minimum line length
            max_line_gap = 20  # Maximum gap between lines to be considered as single line
            detected_lines = apply_hough_transform(eroded_image, min_line_length, max_line_gap)

            # Merge the detected lines
            min_distance_threshold = 10  # Adjust as needed
            merged_lines = merge_lines(detected_lines, min_distance_threshold)

            # Draw detected lines on the original image
            output_image = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
            if merged_lines is not None:
                for line in merged_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green lines for detected lines

            # Display the images
            plt.figure(figsize=(15, 5))
            plt.subplot(131), plt.imshow(original_image_rgb)
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(132), plt.imshow(equalized_image, cmap='gray')
            plt.title('Equalized LAB Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(133), plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.title('Output Image'), plt.xticks([]), plt.yticks([])
            plt.show()


### ***OUR PIPELINE***

**Image Processing with Gaussian Blur and Canny Edge Detection**

import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

# Specify the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

# Get list of files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file in the folder
for filename in file_list:
    # Check if the file is an image (you may want to add more robust checks here)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read input image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Apply Gaussian filter with default kernel size (5x5)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply Canny edge detector
        edges = cv2.Canny(blurred_image, 20, 50)  # Adjust thresholds as needed

        # Display final image with detected edges
        cv2_imshow(edges)
        cv2.waitKey(0)

        # Create and display a blank image (you can adjust the size as needed)
        blank_image = np.zeros_like(edges)
        cv2_imshow(blank_image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()




**Gaussian, Canny, Probablistic Hough**

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

    # Apply Canny edge detector
    edges = cv2.Canny(gray, 20, 50)  # Adjust thresholds as needed

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
    cv2_imshow(edges)  # Edges
    cv2_imshow(result_image)  # Original Image with Detected Lines
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the folder containing images
folder_path = '/content/drive/MyDrive/COMPUTER VISSION ALLIANCE'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png')or filename.endswith('.jpegg'):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        print("Processing:", image_path)
        process_image(image_path)
