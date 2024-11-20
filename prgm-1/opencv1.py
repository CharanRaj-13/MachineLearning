'''
pip install opencv-python
'''

import cv2

# Example: Loading and displaying an image
image = cv2.imread('C:\\img\\path')
cv2.imshow('Sample Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Primarily for image processing, computer vision tasks like image classification, object detection, etc.