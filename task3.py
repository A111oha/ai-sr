import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Застосовуємо фільтри Шарра для виявлення контурів
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Горизонтальні контури
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Вертикальні контури

# Об'єднуємо результати двох фільтрів
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Перетворюємо результат в формат uint8 для відображення
sobel_combined = np.uint8(np.absolute(sobel_combined))

# Відображаємо результат
plt.figure(figsize=(10, 10))
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')
plt.show()
