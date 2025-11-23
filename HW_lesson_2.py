import cv2

SCALING_INT = 2

img_email = cv2.imread('images/3-email.jpg')
img_self = cv2.imread('images/4-self.jpg')

img_email = cv2.resize(img_email, (img_email.shape[1] // SCALING_INT,
                                   img_email.shape[0] // SCALING_INT))
img_self = cv2.resize(img_self, (img_self.shape[1] // SCALING_INT,
                                 img_self.shape[0] // SCALING_INT))

img_email = cv2.cvtColor(img_email, cv2.COLOR_BGR2GRAY)
img_self = cv2.cvtColor(img_self, cv2.COLOR_BGR2GRAY)

img_email = cv2.Canny(img_email, 100, 100)
img_self = cv2.Canny(img_self, 100, 100)

cv2.imwrite('result/email.jpg', img_email)
cv2.imwrite('result/self.jpg', img_self)
