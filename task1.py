import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантажуємо зображення
image = cv2.imread('image2.jpg', cv2.IMREAD_COLOR)

# Перетворюємо на сіре зображення
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Застосовуємо фільтр Гауса для згладжування зображення
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 2. Використовуємо фільтр Canny для виявлення контурів
edges = cv2.Canny(blurred_image, 100, 200)

# Відображаємо результати з більш великими зображеннями
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Параметр figsize для фігури, збільшуємо тільки по ширині
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(blurred_image, cmap='gray')
axes[1].set_title('Gaussian Blurred Image')
axes[1].axis('off')

axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Canny Edges')
axes[2].axis('off')


for ax in axes:
    ax.set_aspect('auto')

plt.tight_layout()
plt.show()

