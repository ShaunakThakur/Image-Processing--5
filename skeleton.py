import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('racecar2.jpg', 0)  # Read the image in grayscale

# Threshold the image (convert to binary)
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Invert the binary image
thresh = cv2.bitwise_not(thresh)

# Apply a structural element for the skeletonization
skel = np.zeros(thresh.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Skeletonization process
while cv2.countNonZero(thresh) > 0:
    eroded = cv2.erode(thresh, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(thresh, temp)
    skel = cv2.bitwise_or(skel, temp)
    thresh = eroded.copy()

# Display the original image and its skeleton
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(skel, cmap='gray')
plt.title('Skeletonized Image')

plt.tight_layout()
plt.show()
