import cv2
import numpy as np

SCALING_INT = 5

image = cv2.imread('images/2.jpg')
print(image.shape)

# image = cv2.resize(image, (300, 800))
image = cv2.resize(image, (image.shape[1] // SCALING_INT,
                           image.shape[0] // SCALING_INT))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 100, 100)

kernel = np.ones((5, 5), np.uint8())

image = cv2.dilate(image, kernel, 1)  # Розширює світлі області на зоображені.
image = cv2.erode(image, kernel, 1)  # Розширює темні області на зоображені.

image = cv2.imwrite('car.jpg', image)

cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
