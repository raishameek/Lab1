import cv2
import numpy as np

# Reading the image
image = cv2.imread('download.jfif')

# Get the width and height of the image
height, width = image.shape[:2]

# Get the center of the image to create the rotation matrix
center = (width / 2, height / 2)

# Create the rotation matrix with 45 degrees angle and scale 1
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

# Rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

# Get tx and ty values for translation (you can adjust these as needed)
tx, ty = width / 4, height / 4

# Create the translation matrix using tx and ty
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)

# Apply the translation to the rotated image
translated_image = cv2.warpAffine(src=rotated_image, M=translation_matrix, dsize=(width, height))

# Display the original, rotated, and translated images
cv2.imshow('Original image', image)
cv2.imshow('Rotated image', rotated_image)
cv2.imshow('Translated image', translated_image)

# Wait indefinitely, press any key to exit
cv2.waitKey(0)

# Save the rotated and translated images to disk
cv2.imwrite('rotated_image.jpg', rotated_image)
cv2.imwrite('translated_image.jpg', translated_image)

# Close the windows
cv2.destroyAllWindows()
