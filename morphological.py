import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('racecar2.jpg', 0)  # Read the image in grayscale

# Define a kernel
kernel = np.ones((5, 5), np.uint8)

# Perform erosion
erosion = cv2.erode(image, kernel, iterations=1)

# Perform dilation
dilation = cv2.dilate(image, kernel, iterations=1)

# Display the original image, erosion, and dilation
plt.figure(figsize=(10, 5))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(132)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')

plt.subplot(133)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')

plt.tight_layout()
plt.show()