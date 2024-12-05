import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

# Перетворюємо на сіре зображення
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Застосовуємо фільтр Гауса для згладжування
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Використовуємо детектор Canny для виявлення країв
edges = cv2.Canny(blurred_image, 50, 150, apertureSize=3)

# Застосовуємо перетворення Хафа для виявлення ліній
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Малюємо лінії на зображенні
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Відображаємо результати
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Line Detection using Hough Transform')
plt.axis('off')
plt.show()
