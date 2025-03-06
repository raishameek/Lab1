import cv2
import numpy as np

# 1. Reading an Image
img = cv2.imread('image.jpg')

# 2. Displaying the Original Image
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# 3. Grayscale Conversion
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_img)
cv2.waitKey(0)

# 4. Image Resizing
resized_img = cv2.resize(img, (400, 400))  
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)

# 5. Image Cropping
cropped_img = img[100:300, 100:300]  
cv2.imshow('Cropped Image', cropped_img)
cv2.waitKey(0)

# 6. Image Rotation
rows, cols = img.shape[:2]
matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_img = cv2.warpAffine(img, matrix, (cols, rows))
cv2.imshow('Rotated Image', rotated_img)
cv2.waitKey(0)

# 7. Thresholding
_, thresholded_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', thresholded_img)
cv2.waitKey(0)

# 8. Blurring (Gaussian Blur)
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Blurred Image', blurred_img)
cv2.waitKey(0)

# 9. Edge Detection (Canny)
edges = cv2.Canny(img, 100, 200)
cv2.imshow('Edge Detection (Canny)', edges)
cv2.waitKey(0)

# 10. Image Sharpening
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_img = cv2.filter2D(img, -1, kernel)
cv2.imshow('Sharpened Image', sharpened_img)
cv2.waitKey(0)

# 11. Histogram Equalization
equalized_img = cv2.equalizeHist(gray_img)
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)

# 12. Morphological Operations (Dilation & Erosion)
kernel = np.ones((5, 5), np.uint8)
dilated_img = cv2.dilate(thresholded_img, kernel, iterations=1)
eroded_img = cv2.erode(thresholded_img, kernel, iterations=1)
cv2.imshow('Dilated Image', dilated_img)
cv2.waitKey(0)
cv2.imshow('Eroded Image', eroded_img)
cv2.waitKey(0)

# 13. Contour Detection
contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)

# 14. Color Space Conversion (BGR to HSV)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_img)
cv2.waitKey(0)

# 15. Drawing on an Image (Rectangle & Circle)
img_copy = img.copy()
cv2.rectangle(img_copy, (50, 50), (200, 200), (255, 0, 0), 2)  
cv2.circle(img_copy, (300, 300), 50, (0, 255, 0), 2)  
cv2.imshow('Drawing on Image', img_copy)
cv2.waitKey(0)

# Cleanup all windows
cv2.destroyAllWindows()
