import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh từ file
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Ảnh âm tính
def negative_image(image):
    return 255 - image

# 2. Tăng độ tương phản ảnh
def contrast_image(image, alpha=1.5, beta=0):
    # alpha: độ tương phản, beta: độ sáng
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 3. Biến đổi log
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    return np.array(log_image, dtype=np.uint8)

# 4. Cân bằng Histogram
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Hiển thị ảnh gốc và các ảnh đã qua xử lý
def display_images(original, neg_img, contrast_img, log_img, hist_img):
    titles = ['Original Image', 'Negative Image', 'Contrast Image', 'Log Transform', 'Histogram Equalization']
    images = [original, neg_img, contrast_img, log_img, hist_img]
    
    plt.figure(figsize=(10, 7))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# Gọi các hàm xử lý ảnh
neg_img = negative_image(image)
contrast_img = contrast_image(image)
log_img = log_transform(image)
hist_img = histogram_equalization(image)

# Hiển thị kết quả
display_images(image, neg_img, contrast_img, log_img, hist_img)