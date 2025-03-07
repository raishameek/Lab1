import cv2
import numpy as np
img = cv2.imread('istockphoto-1442179368-612x612.jpg')

h, w, channels = img.shape

half_w = w // 2
half_h = h // 2

# Split the image into four quadrants
left_part = img[:, :half_w]
right_part = img[:, half_w:]
top_part = img[:half_h, :]
bottom_part = img[half_h:, :]

# Resize the parts if necessary to make sure all parts align
# This should be unnecessary as long as the image is evenly divisible
# by half, but it's a good safeguard
left_part_resized = cv2.resize(left_part, (half_w, half_h))
right_part_resized = cv2.resize(right_part, (half_w, half_h))
top_part_resized = cv2.resize(top_part, (half_w, half_h))
bottom_part_resized = cv2.resize(bottom_part, (half_w, half_h))

# Combine the quadrants into one single image (top-left, top-right, bottom-left, bottom-right)
top_row = np.hstack((left_part_resized, right_part_resized))  # Combine left and right parts horizontally
bottom_row = np.hstack((top_part_resized, bottom_part_resized))  # Combine top and bottom parts horizontally

# Stack the two rows vertically to create the full image
combined_image = np.vstack((top_row, bottom_row))

# Display the combined image in one window
cv2.imshow('Combined Image', combined_image)

# Save all individual parts
cv2.imwrite('top.jpg', top_part)
cv2.imwrite('bottom.jpg', bottom_part)
cv2.imwrite('right.jpg', right_part)
cv2.imwrite('left.jpg', left_part)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
