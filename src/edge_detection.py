
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

