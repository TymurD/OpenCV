import cv2
import numpy as np


COLORS = {
    "purple": [(np.array([125, 50, 50]), np.array([155, 255, 255]))],
    "pink": [(np.array([155, 100, 60]), np.array([180, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
}

IMAGE_PATH = "images/7_candies.jpg"
image_original = cv2.imread(IMAGE_PATH)
image_original = cv2.resize(image_original, (640, 480))

hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

image_masked = np.zeros(hsv.shape[:2], dtype="uint8")

contours = {}

for color_name, ranges in COLORS.items():
    for (lower, upper) in ranges:
        mask = cv2.inRange(hsv, lower, upper)
        image_masked = cv2.bitwise_or(image_masked, mask)
        contours[f"{color_name}"] = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[0]

# kernel = np.ones((5, 5), np.uint8)
# image_mask = cv2.morphologyEx(image_masked, cv2.MORPH_CLOSE, kernel)

# image_result = cv2.bitwise_and(image_original,
#                                image_original,
#                                mask=image_mask)
colors_simple = {
    "purple": (255, 0, 255),
    "pink": (0, 0, 255),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255)
}

for color_name, color_contours in contours.items():
    for cnts in color_contours:
        if cv2.contourArea(cnts) < 600:
            continue
        x, y, w, h = cv2.boundingRect(cnts)
        cv2.rectangle(image_original, (x, y), (x + w, y + h),
                      colors_simple[color_name], 2)
        cv2.putText(image_original, f"{color_name} candy", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors_simple[color_name], 2)
        cv2.drawContours(image_original,
                         [cnts], -1,
                         colors_simple[color_name], 2)

cv2.imshow("Original Image", image_original)
cv2.imwrite("result/output_candies.png", image_original)


print("Press 'q' to exit the application.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
