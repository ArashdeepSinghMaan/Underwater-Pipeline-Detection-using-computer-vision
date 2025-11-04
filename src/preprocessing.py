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
