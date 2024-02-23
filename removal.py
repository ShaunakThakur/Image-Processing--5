import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('racecar2.jpg')  # Read the image

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to create a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform morphological closing to fill small holes
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Perform morphological opening to remove small objects
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a threshold area (adjust as needed)
threshold_area = 100  # Change this value to alter the minimum area of the object to keep

# Filter contours by area, removing smaller ones
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold_area]

# Create a mask of the filtered contours
mask = np.zeros_like(opening)
cv2.drawContours(mask, filtered_contours, -1, 255, -1)

# Bitwise AND to remove smaller objects
result = cv2.bitwise_and(image, image, mask=mask)

# Display the original image and the processed result
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(122)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Image with Small Objects Removed')

plt.tight_layout()
plt.show()
